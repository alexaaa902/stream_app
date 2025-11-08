%%writefile main.py
from __future__ import annotations

# ---------- Imports ----------
from typing import Optional, Dict, Any
from pathlib import Path
import json
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
        tender_year:    Optional[int]  = None
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
        predicted_days: float
        risk_flag: bool
        model_used: str
        tau: float

# ---------- App ----------
app = FastAPI(title="ProcureSight API", version="1.0")

# Health check: ελαφρύ, για Render/monitors
@app.get("/health")
def health_check() -> dict:
    return {"ok": True}

# Root: δείχνει αν τα μοντέλα είναι φορτωμένα
@app.get("/")
def root() -> dict:
    try:
        _ensure_models_loaded()
        model_ok = bool(clf and reg_short and reg_long)
        return {"name": "ProcureSight API", "version": "1.0", "model_loaded": model_ok}
    except Exception as e:
        return {"name": "ProcureSight API", "version": "1.0", "model_loaded": False, "error": str(e)}

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

def _combine_hard(p, y_short, y_long, tau):
    """Use long reg if prob>=tau; clamp to [1,1800]."""
    CAP_MIN, CAP_MAX = 1.0, 1800.0
    use_long = (np.asarray(p, float) >= float(tau))
    out = np.where(use_long, np.asarray(y_long, float), np.asarray(y_short, float))
    return np.clip(out, CAP_MIN, CAP_MAX)

def _year_bump_from_meta(year: Optional[int]) -> float:
    if year is None or not meta:
        return 0.0
    bump = meta.get("bump", {}) or {}
    if (bump.get("mode") != "year-aware") or (meta.get("year_bump_info") is None):
        return 0.0
    info = meta["year_bump_info"]
    coeffs = info.get("poly_coeffs", [])
    max_b = float(info.get("max_b", bump.get("year_bump_max", 5.0)))
    if not coeffs:
        return 0.0
    y = float(year); val = 0.0
    for i, a in enumerate(coeffs):
        val += float(a) * (y ** i)
    return float(np.clip(val, 0.0, max_b))

def _apply_year_bump(yhat: float, tender_year: Optional[int]) -> float:
    """One-sided triangular bump around 720 using meta poly."""
    b = _year_bump_from_meta(tender_year)
    LOW, HIGH, CENTER, SNAP_FROM, SNAP_TO = 670.0, 720.0, 720.0, 715.0, 720.0
    yh = float(yhat)
    if yh < LOW or yh >= HIGH or b <= 0:
        return yh
    w = (yh - LOW) / max(CENTER - LOW, 1e-9)
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
        return X  # nothing to do

    # add missing
    missing = [c for c in exp if c not in X.columns]
    for c in missing:
        X[c] = np.nan

    # (optional) you can log extras if needed:
    # extras = [c for c in X.columns if c not in exp]
    # if extras: print("DEBUG extras dropped:", extras)

    return X[exp]

# ---------- Loader ----------
def _load_artifacts() -> dict:
    global clf, reg_short, reg_long, features, meta, LONG_THR_DEFAULT
    model_dir = Path(__file__).resolve().parent / "model"
    f_clf = model_dir / "stage1_classifier.txt"
    f_rs  = model_dir / "stage2_reg_short.txt"
    f_rl  = model_dir / "stage2_reg_long.txt"
    f_feat= model_dir / "features.json"
    f_meta= model_dir / "meta.json"

    if not model_dir.exists():
        raise RuntimeError(f"Model dir not found: {model_dir}")
    if lgb is None:
        raise RuntimeError("LightGBM is not installed. Please `pip install lightgbm`.")

    clf = lgb.Booster(model_file=str(f_clf))
    reg_short = lgb.Booster(model_file=str(f_rs))
    reg_long  = lgb.Booster(model_file=str(f_rl))

    with open(f_feat, "r", encoding="utf-8") as f:
        features = json.load(f)
    if f_meta.exists():
        with open(f_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        LONG_THR_DEFAULT = float(meta.get("long_threshold_days", LONG_THR_DEFAULT))

    return {"clf": clf, "reg_short": reg_short, "reg_long": reg_long,
            "features": features, "meta": meta, "LONG_THR_DEFAULT": LONG_THR_DEFAULT}

def _ensure_models_loaded():
    global clf, reg_short, reg_long, features, meta, LONG_THR_DEFAULT
    if clf is not None and reg_short is not None and reg_long is not None and features:
        return
    res = _load_artifacts()
    if isinstance(res, dict):
        clf       = res.get("clf")       or clf
        reg_short = res.get("reg_short") or reg_short
        reg_long  = res.get("reg_long")  or reg_long
        features  = res.get("features")  or features
        meta      = res.get("meta")      or meta
        LONG_THR_DEFAULT = res.get("LONG_THR_DEFAULT", LONG_THR_DEFAULT)

# ---------- Build X ----------
def _build_dataframe(req: PredictRequest) -> pd.DataFrame:
    d = req.model_dump()

    # Normalize
    d["tender_country"] = (d.get("tender_country") or "").upper().strip()
    if d.get("tender_mainCpv") is not None:
        d["tender_mainCpv"] = str(d["tender_mainCpv"]).strip()

    X = pd.DataFrame([d])

    # CPV derived features (numeric)
    if "tender_mainCpv" in X.columns and X["tender_mainCpv"].notna().any():
        cpv_div2, cpv_grp3 = _derive_cpv_parts(X["tender_mainCpv"])
        X["cpv_div2"], X["cpv_grp3"] = cpv_div2, cpv_grp3

    # Ensure all training features exist & order
    feat_list = list(features.get("features", []))
    cat_list  = set(features.get("categorical", []))
    for c in feat_list:
        if c not in X.columns:
            X[c] = np.nan
    if feat_list:
        X = X[feat_list]

    # Auto-compute *_log fields (training used ln(1+x))
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

    # Target safeguard (αν υπάρχει μέσα στα features από λάθος)
    if "target_duration" in X.columns and X["target_duration"].isna().any():
        X["target_duration"] = 700.0

    # Dtypes policy (respect features.json)
    for c in X.columns:
        if c in cat_list:
            X[c] = X[c].astype("string").str.strip().astype("category")
        else:
            if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

    # guard: if any category not in cat_list → back to numeric
    for c in X.select_dtypes(include=["category"]).columns:
        if c not in cat_list:
            X[c] = pd.to_numeric(X[c].astype("string"), errors="coerce")

    if "tender_year" in X.columns:
        X["tender_year"] = pd.to_numeric(X["tender_year"], errors="coerce")

    return X

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, tau: Optional[float] = Query(None)):
    import math
    try:
        _ensure_models_loaded()
        if not (clf and reg_short and reg_long):
            raise HTTPException(status_code=503, detail="Models not loaded")

        # --- ensure numeric tau ---
        DEFAULT_TAU = LONG_THR_DEFAULT  # ή 720
        try:
            tau = float(tau) if tau is not None else DEFAULT_TAU
        except Exception:
            tau = DEFAULT_TAU

        # --- build dataframe as usual ---
        X = _build_dataframe(req)

        # --- ensure log features exist (αν λείπουν) ---
        if "tender_estimatedPrice_EUR_log" not in X.columns and "tender_estimatedPrice_EUR" in X.columns:
            X["tender_estimatedPrice_EUR_log"] = X["tender_estimatedPrice_EUR"].apply(
                lambda v: math.log(v) if v and v > 0 else 0
            )
        if "lot_bidsCount_log" not in X.columns and "lot_bidsCount" in X.columns:
            X["lot_bidsCount_log"] = X["lot_bidsCount"].apply(
                lambda v: math.log(v) if v and v > 0 else 0
            )

        # --- Align with boosters (order/shape) ---
        X = _align_to_booster(X, clf)
        X = _align_to_booster(X, reg_short)
        X = _align_to_booster(X, reg_long)

        # --- DEBUG (safe) ---
        try:
            print("\nDEBUG_X (values):\n", X.T)
            print("DEBUG dtypes:", {c: str(t) for c, t in X.dtypes.items()})
            print("DEBUG NaNs per col:", X.isna().sum().to_dict())
            fn = clf.feature_name()
            print("DEBUG Booster feature order (first 25):", fn[:25], "...")
        except Exception:
            pass

        # --- Predict ---
        p       = float(clf.predict(X)[0])
        y_short = float(reg_short.predict(X)[0])
        y_long  = float(reg_long.predict(X)[0])

        # --- sanitize probability ---
        if p is None or not math.isfinite(p):
            p = 0.0

        chosen_tau = float(meta.get("tau", DEFAULT_TAU)) if tau is None else float(tau)
        yhat = float(_combine_hard(p, y_short, y_long, chosen_tau))
        yhat = _apply_year_bump(yhat, getattr(req, "tender_year", None))

        # --- flags ---
        risk_flag_point = yhat >= chosen_tau
        prob_cut = 0.5
        risk_flag_prob = p >= prob_cut

        # --- Return PredictResponse (όπως πριν, απλώς πιο πλήρες) ---
        return PredictResponse(
    predicted_days=yhat,
    risk_flag=bool(yhat >= chosen_tau),   # σημειακό flag: predicted_days vs τ
    model_used="lgbm_2stage",
    tau=chosen_tau,
)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
