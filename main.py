%%writefile main.py
from __future__ import annotations
from pydantic import ConfigDict

# ---------- Imports ----------
from typing import Optional, Dict, Any
from pathlib import Path
import json
import math

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
        target_duration: Optional[float] = None

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

import os

BUILD_ID = "BUILD_2025_12_21_A"

# Health check: ελαφρύ, για Render/monitors
@app.get("/health")
def health_check() -> dict:
    return {
        "ok": True,
        "build": BUILD_ID,
        "file": __file__,
        "cwd": os.getcwd(),
    }

# Root: δείχνει αν τα μοντέλα είναι φορτωμένα
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

# Debug endpoint: δείχνει ξεκάθαρα ποιο αρχείο τρέχει
@app.get("/where")
def where():
    return {
        "ok": True,
        "build": BUILD_ID,
        "file": __file__,
        "cwd": os.getcwd(),
    }


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
    In your training script you wrote:
      "poly_coeffs": [float(c) for c in coeffs[::-1]]
    i.e. constant term first.
    So we evaluate: a0 + a1*y + a2*y^2 ...
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

    # add missing
    missing = [c for c in exp if c not in X.columns]
    for c in missing:
        X[c] = np.nan

    # enforce order + drop extras
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

    # Safeguard if target_duration accidentally is in features list
    if "target_duration" in X.columns and X["target_duration"].isna().any():
        X["target_duration"] = 700.0

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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, tau: Optional[float] = Query(None)):
    """
    Query param:
      - tau: DAYS threshold for risk_flag (e.g. 720)

    Model internals:
      - tau_prob: probability threshold (meta["tau"]) for routing short vs long
      - p_long:  probability from stage-1 classifier
    """
    try:
        _ensure_models_loaded()
        if not (clf and reg_short and reg_long):
            raise HTTPException(status_code=503, detail="Models not loaded")

        # 1) tau_days: risk threshold in DAYS (UI param)
        DEFAULT_TAU_DAYS = float(LONG_THR_DEFAULT)
        try:
            tau_days = float(tau) if tau is not None else DEFAULT_TAU_DAYS
        except Exception:
            tau_days = DEFAULT_TAU_DAYS

        # 2) tau_prob: routing threshold in PROBABILITY (from training artifacts)
        tau_prob = float(meta.get("tau", 0.5))

        # 3) build X
        X = _build_dataframe(req)

        # 4) align columns/order for each booster
        Xc = _align_to_booster(X.copy(), clf)
        Xs = _align_to_booster(X.copy(), reg_short)
        Xl = _align_to_booster(X.copy(), reg_long)

        # 5) predict components
        p = float(clf.predict(Xc)[0])
        y_short = float(reg_short.predict(Xs)[0])
        y_long = float(reg_long.predict(Xl)[0])

        if p is None or not math.isfinite(p):
            p = 0.0

        # 6) combine (routing based on probability threshold)
        yhat = _combine_hard(p, y_short, y_long, tau_prob)

        # 7) apply year-aware bump (if enabled in meta)
        yhat = _apply_year_bump(yhat, req.tender_year)

        # 8) outputs
        stage_used = "long_reg" if (p >= tau_prob) else "short_reg"
        risk_flag_point = (yhat >= tau_days)

        return PredictResponse(
            predicted_days=float(yhat),
            risk_flag=bool(risk_flag_point),
            model_used="lgbm_2stage",
            tau_days=float(tau_days),
            p_long=float(p),
            tau_prob=float(tau_prob),
            stage_used=str(stage_used),
            pred_short=float(y_short),
            pred_long=float(y_long),
            build="BUILD_2025_12_21_A",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
