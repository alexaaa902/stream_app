%%writefile main.py
from __future__ import annotations
from pydantic import ConfigDict

# ---------- Imports ----------
from typing import Optional, Dict, Any
from pathlib import Path
import json
import math
import os
import hashlib

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

# ---------- Optional schemas ----------
try:
    from .schemas import PredictRequest, PredictResponse  # if you have them
except Exception:
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        # βασικά
        tender_country: Optional[str] = None
        tender_mainCpv: Optional[str] = None
        tender_year: Optional[int] = None

        # πρόσθετα που αναφέρονται στο features.json
        tender_procedureType: Optional[str] = None
        tender_supplyType: Optional[str] = None
        buyer_buyerType: Optional[str] = None
        buyer_country: Optional[str] = None

        tender_estimatedPrice_EUR: Optional[float] = None
        tender_indicator_score_INTEGRITY: Optional[float] = None
        tender_indicator_score_ADMINISTRATIVE: Optional[float] = None
        tender_indicator_score_TRANSPARENCY: Optional[float] = None
        lot_bidsCount: Optional[float] = None

        tender_estimatedPrice_EUR_log: Optional[float] = None
        lot_bidsCount_log: Optional[float] = None

        # ⚠️ ΔΕΝ συμπεριλαμβάνουμε target_duration (leakage)

    class PredictResponse(BaseModel):
        model_config = ConfigDict(protected_namespaces=())  # για να φύγει το warning με model_used

        predicted_days: float
        risk_flag: bool
        model_used: str
        tau_days: float
        p_long: float
        tau_prob: float
        stage_used: str
        pred_short: float
        pred_long: float
        build: str


# ---------- App ----------
app = FastAPI(title="ProcureSight API", version="1.0")

BUILD_ID = "BUILD_2025_12_21_A"

# ---------- Globals / defaults ----------
features: Dict[str, Any] = {}
meta: Dict[str, Any] = {}
clf = None
reg_short = None
reg_long = None
LONG_THR_DEFAULT = 720.0

# ---------- LightGBM import ----------
try:
    import lightgbm as lgb
except Exception:
    lgb = None


# ---------- Health / debug ----------
@app.get("/health")
def health_check() -> dict:
    return {
        "ok": True,
        "build": BUILD_ID,
        "file": __file__,
        "cwd": os.getcwd(),
    }


@app.get("/")
def root() -> dict:
    try:
        _ensure_models_loaded()
        model_ok = bool(clf and reg_short and reg_long)
        return {
            "name": "ProcureSight API",
            "version": "1.0",
            "model_loaded": model_ok,
            "build": BUILD_ID,
            "file": __file__,
        }
    except Exception as e:
        return {
            "name": "ProcureSight API",
            "version": "1.0",
            "model_loaded": False,
            "error": str(e),
            "build": BUILD_ID,
            "file": __file__,
        }


@app.get("/where")
def where():
    return {
        "ok": True,
        "build": BUILD_ID,
        "file": __file__,
        "cwd": os.getcwd(),
    }


# ---------- Helpers ----------
def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _derive_cpv_parts(s: pd.Series):
    """Return cpv_div2, cpv_grp3 as numeric from an 8-digit cleaned CPV string."""
    s = s.astype(str).str.replace(r"[^\d]", "", regex=True).str.zfill(8)
    return _safe_num(s.str[:2]), _safe_num(s.str[:3])


def _combine_hard(p_long: float, y_short: float, y_long: float, tau_prob: float) -> float:
    """
    2-stage HARD:
      - if p_long >= tau_prob => long reg
      - else => short reg
    clamp to [1,1800]
    """
    CAP_MIN, CAP_MAX = 1.0, 1800.0
    use_long = float(p_long) >= float(tau_prob)
    out = float(y_long) if use_long else float(y_short)
    return float(np.clip(out, CAP_MIN, CAP_MAX))


def _year_bump_from_meta(year: Optional[int]) -> float:
    """
    Reads poly coeffs from meta["year_bump_info"]["poly_coeffs"].
    In training you stored constant term first.
    Evaluate: a0 + a1*y + a2*y^2 ...
    """
    if year is None or not meta:
        return 0.0

    bump = meta.get("bump", {}) or {}
    if (bump.get("mode") != "year-aware") or (meta.get("year_bump_info") is None):
        return 0.0

    info = meta.get("year_bump_info") or {}
    coeffs = info.get("poly_coeffs", []) or []
    max_b = float(info.get("max_b", bump.get("year_bump_max", 5.0)))

    if not coeffs:
        return 0.0

    y = float(year)
    val = 0.0
    for i, a in enumerate(coeffs):
        val += float(a) * (y ** i)

    return float(np.clip(val, 0.0, max_b))


def _apply_year_bump(yhat: float, tender_year: Optional[int]) -> float:
    """One-sided triangular bump around ~720 using meta poly (same logic as training)."""
    b = _year_bump_from_meta(tender_year)

    LOW, HIGH, CENTER, SNAP_FROM, SNAP_TO = 670.0, 720.0, 720.0, 715.0, 720.0
    yh = float(yhat)

    # bump only in [LOW, HIGH)
    if yh < LOW or yh >= HIGH or b <= 0:
        return yh

    w = (yh - LOW) / max(CENTER - LOW, 1e-9)  # 0..1
    yh2 = yh + b * max(min(w, 1.0), 0.0)

    if SNAP_FROM <= yh2 < SNAP_TO:
        yh2 = SNAP_TO

    return float(yh2)


def _align_to_booster(X: pd.DataFrame, booster) -> pd.DataFrame:
    """
    Align X to LightGBM booster:
      - add missing cols as NaN
      - drop extras
      - enforce exact order
    """
    try:
        exp = list(booster.feature_name())
    except Exception:
        exp = None

    if not exp or any((name is None) or (name == "") for name in exp):
        return X

    missing = [c for c in exp if c not in X.columns]
    for c in missing:
        X[c] = np.nan

    return X[exp]


# ---------- Loader ----------
def _load_artifacts() -> dict:
    global clf, reg_short, reg_long, features, meta, LONG_THR_DEFAULT

    model_dir = Path(__file__).resolve().parent / "model"
    f_clf = model_dir / "stage1_classifier.txt"
    f_rs = model_dir / "stage2_reg_short.txt"
    f_rl = model_dir / "stage2_reg_long.txt"
    f_feat = model_dir / "features.json"
    f_meta = model_dir / "meta.json"

    if not model_dir.exists():
        raise RuntimeError(f"Model dir not found: {model_dir}")
    if lgb is None:
        raise RuntimeError("LightGBM is not installed. Please `pip install lightgbm`.")

    clf = lgb.Booster(model_file=str(f_clf))
    reg_short = lgb.Booster(model_file=str(f_rs))
    reg_long = lgb.Booster(model_file=str(f_rl))

    with open(f_feat, "r", encoding="utf-8") as f:
        features = json.load(f)

    if f_meta.exists():
        with open(f_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        LONG_THR_DEFAULT = float(meta.get("long_threshold_days", LONG_THR_DEFAULT))

    return {
        "clf": clf,
        "reg_short": reg_short,
        "reg_long": reg_long,
        "features": features,
        "meta": meta,
        "LONG_THR_DEFAULT": LONG_THR_DEFAULT,
    }


def _ensure_models_loaded():
    global clf, reg_short, reg_long, features, meta, LONG_THR_DEFAULT
    if clf is not None and reg_short is not None and reg_long is not None and features:
        return
    res = _load_artifacts()
    clf = res.get("clf") or clf
    reg_short = res.get("reg_short") or reg_short
    reg_long = res.get("reg_long") or reg_long
    features = res.get("features") or features
    meta = res.get("meta") or meta
    LONG_THR_DEFAULT = res.get("LONG_THR_DEFAULT", LONG_THR_DEFAULT)


# ---------- Build X ----------
def _build_dataframe(req: PredictRequest) -> pd.DataFrame:
    """
    Build a 1-row dataframe matching training features.json
    and compute derived CPV parts + log1p transforms.
    """
    d = req.model_dump()

    # Normalize base fields
    d["tender_country"] = (d.get("tender_country") or "").upper().strip()
    if d.get("tender_mainCpv") is not None:
        d["tender_mainCpv"] = str(d["tender_mainCpv"]).strip()

    X = pd.DataFrame([d])

    # Derived CPV parts
    if "tender_mainCpv" in X.columns and X["tender_mainCpv"].notna().any():
        cpv_div2, cpv_grp3 = _derive_cpv_parts(X["tender_mainCpv"])
        X["cpv_div2"], X["cpv_grp3"] = cpv_div2, cpv_grp3

    # Ensure all training features exist & order (features.json)
    feat_list = list(features.get("features", []))
    cat_list = set(features.get("categorical", []))

    for c in feat_list:
        if c not in X.columns:
            X[c] = np.nan
    if feat_list:
        X = X[feat_list]

    # Auto-compute *_log fields (training used log1p)
    if "tender_estimatedPrice_EUR" in X.columns and "tender_estimatedPrice_EUR_log" in X.columns:
        base = pd.to_numeric(X["tender_estimatedPrice_EUR"], errors="coerce")
        approx_ln = np.log1p(base)
        given = pd.to_numeric(X["tender_estimatedPrice_EUR_log"], errors="coerce")
        need_fix = given.isna() | ~np.isfinite(given) | (np.abs(given - approx_ln) > 2.5)
        X.loc[need_fix, "tender_estimatedPrice_EUR_log"] = approx_ln

    if "lot_bidsCount" in X.columns and "lot_bidsCount_log" in X.columns:
        base = pd.to_numeric(X["lot_bidsCount"], errors="coerce")
        approx_ln = np.log1p(base)
        given = pd.to_numeric(X["lot_bidsCount_log"], errors="coerce")
        need_fix = given.isna() | ~np.isfinite(given) | (np.abs(given - approx_ln) > 0.5)
        X.loc[need_fix, "lot_bidsCount_log"] = approx_ln

    # ⚠️ Καμία λογική default για target_duration (δεν υπάρχει πλέον)

    # Dtypes policy (respect categorical list)
    for c in X.columns:
        if c in cat_list:
            X[c] = X[c].astype("string").str.strip().astype("category")
        else:
            if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

    # If something became category but shouldn't, coerce to numeric
    for c in X.select_dtypes(include=["category"]).columns:
        if c not in cat_list:
            X[c] = pd.to_numeric(X[c].astype("string"), errors="coerce")

    if "tender_year" in X.columns:
        X["tender_year"] = pd.to_numeric(X["tender_year"], errors="coerce")

    return X


# ---------- model_info endpoint (hashes / sizes) ----------
def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


@app.get("/model_info")
def model_info():
    _ensure_models_loaded()
    model_dir = Path(__file__).resolve().parent / "model"
    files = {
        "stage1_classifier.txt": model_dir / "stage1_classifier.txt",
        "stage2_reg_short.txt": model_dir / "stage2_reg_short.txt",
        "stage2_reg_long.txt": model_dir / "stage2_reg_long.txt",
        "features.json": model_dir / "features.json",
        "meta.json": model_dir / "meta.json",
    }

    info: Dict[str, Any] = {}
    for k, p in files.items():
        info[k] = {
            "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else None,
            "mtime": p.stat().st_mtime if p.exists() else None,
            "sha1": _sha1(p) if p.exists() else None,
        }

    return {
        "build": BUILD_ID,
        "file": __file__,
        "cwd": os.getcwd(),
        "long_thr_default": float(LONG_THR_DEFAULT),
        "tau_prob": float(meta.get("tau", 0.5)) if meta else None,
        "n_features": len(features.get("features", [])) if features else None,
        "files": info,
    }


# ---------- Predict endpoints ----------
@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    tau_prob: Optional[float] = Query(
        None, description="Probability cutoff for risk_flag (e.g., 0.686). Default: model tau from meta."
    ),
    tau_days: Optional[float] = Query(
        None, description="Days threshold for reference/reporting (e.g., 720). Default: LONG_THR_DEFAULT."
    ),
):
    """
    - tau_prob: controls the *risk flag* (probability-based early warning)
    - tau_days: optional reference threshold in days (does NOT control risk_flag)
    Internals:
      - tau_prob_model (meta['tau']) routes short vs long regressor
      - p_long is stage-1 probability P(long>=720)
    """
    try:
        _ensure_models_loaded()
        if not (clf and reg_short and reg_long):
            raise HTTPException(status_code=503, detail="Models not loaded")

        # --- defaults ---
        DEFAULT_TAU_DAYS = float(LONG_THR_DEFAULT)
        tau_days_val = DEFAULT_TAU_DAYS if tau_days is None else float(tau_days)

        tau_prob_model = float(meta.get("tau", 0.5))  # routing threshold (training)
        tau_prob_val = tau_prob_model if tau_prob is None else float(tau_prob)
        tau_prob_val = float(np.clip(tau_prob_val, 0.0, 1.0))

        # 1) build X
        X = _build_dataframe(req)

        # 2) align
        Xc = _align_to_booster(X.copy(), clf)
        Xs = _align_to_booster(X.copy(), reg_short)
        Xl = _align_to_booster(X.copy(), reg_long)

        # 3) predict components
        p = float(clf.predict(Xc)[0])
        y_short = float(reg_short.predict(Xs)[0])
        y_long = float(reg_long.predict(Xl)[0])

        if (p is None) or (not math.isfinite(p)):
            p = 0.0

        # 4) combine (routing based on MODEL threshold, not user cutoff)
        yhat = _combine_hard(p, y_short, y_long, tau_prob_model)

        # 5) year-aware bump
        yhat = _apply_year_bump(yhat, req.tender_year)

        # outputs
        stage_used = "long_reg" if (p >= tau_prob_model) else "short_reg"

        # ✅ risk flag is PROBABILITY-BASED (what you want)
        risk_flag_point = (p >= tau_prob_val)

        return PredictResponse(
            predicted_days=float(yhat),
            risk_flag=bool(risk_flag_point),
            model_used="lgbm_2stage",
            tau_days=float(tau_days_val),     # reference only
            p_long=float(p),
            tau_prob=float(tau_prob_val),     # the cutoff used for risk_flag
            stage_used=str(stage_used),
            pred_short=float(y_short),
            pred_long=float(y_long),
            build=BUILD_ID,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict_debug")
def predict_debug(req: PredictRequest):
    _ensure_models_loaded()

    X = _build_dataframe(req)
    Xc = _align_to_booster(X.copy(), clf)
    Xs = _align_to_booster(X.copy(), reg_short)
    Xl = _align_to_booster(X.copy(), reg_long)

    def row_dict(df: pd.DataFrame):
        v = df.iloc[0].to_dict()
        return {k: (None if (pd.isna(val) or val is pd.NA) else val) for k, val in v.items()}

    return {
        "raw_X_cols": list(X.columns),
        "raw_X_row": row_dict(X),

        "clf_expected": list(clf.feature_name()) if clf is not None else None,
        "short_expected": list(reg_short.feature_name()) if reg_short is not None else None,
        "long_expected": list(reg_long.feature_name()) if reg_long is not None else None,

        "clf_cols": list(Xc.columns),
        "short_cols": list(Xs.columns),
        "long_cols": list(Xl.columns),

        "clf_row": row_dict(Xc),
        "short_row": row_dict(Xs),
        "long_row": row_dict(Xl),
    }
