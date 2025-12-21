from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import json, math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

# ---------- Schemas ----------
class PredictRequest(BaseModel):
    tender_country: Optional[str] = None
    tender_mainCpv: Optional[str] = None
    tender_year: Optional[int] = None
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
    # (προαιρετικό) για να φύγει το warning για model_used/model_
    model_config = ConfigDict(protected_namespaces=())

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

@app.get("/health")
def health_check():
    return {"ok": True}

# ---------- Globals ----------
features: Dict[str, Any] = {}
meta: Dict[str, Any] = {}
clf = reg_short = reg_long = None
LONG_THR_DEFAULT = 720.0
BUILD = "BUILD_2025_12_21_A"

try:
    import lightgbm as lgb
except Exception:
    lgb = None

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _derive_cpv_parts(s: pd.Series):
    s = s.astype(str).str.replace(r"[^\d]", "", regex=True).str.zfill(8)
    return _safe_num(s.str[:2]), _safe_num(s.str[:3])

def _align_to_booster(X: pd.DataFrame, booster) -> pd.DataFrame:
    try:
        exp = list(booster.feature_name())
    except Exception:
        exp = None
    if not exp:
        return X
    for c in exp:
        if c not in X.columns:
            X[c] = np.nan
    return X[exp]

def _combine_hard(p_long: float, y_short: float, y_long: float, tau_prob: float) -> float:
    use_long = float(p_long) >= float(tau_prob)
    out = float(y_long) if use_long else float(y_short)
    return float(np.clip(out, 1.0, 1800.0))

def _load_artifacts():
    global clf, reg_short, reg_long, features, meta, LONG_THR_DEFAULT
    model_dir = Path(__file__).resolve().parent / "model"

    if lgb is None:
        raise RuntimeError("LightGBM is not installed.")
    if not model_dir.exists():
        raise RuntimeError(f"Model dir not found: {model_dir}")

    clf = lgb.Booster(model_file=str(model_dir / "stage1_classifier.txt"))
    reg_short = lgb.Booster(model_file=str(model_dir / "stage2_reg_short.txt"))
    reg_long = lgb.Booster(model_file=str(model_dir / "stage2_reg_long.txt"))

    with open(model_dir / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)

    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        LONG_THR_DEFAULT = float(meta.get("long_threshold_days", LONG_THR_DEFAULT))

def _ensure_models_loaded():
    if clf is not None and reg_short is not None and reg_long is not None and features:
        return
    _load_artifacts()

@app.get("/")
def root():
    try:
        _ensure_models_loaded()
        return {"name": "ProcureSight API", "model_loaded": True, "build": BUILD}
    except Exception as e:
        return {"name": "ProcureSight API", "model_loaded": False, "error": str(e), "build": BUILD}

def _build_dataframe(req: PredictRequest) -> pd.DataFrame:
    d = req.model_dump()
    d["tender_country"] = (d.get("tender_country") or "").upper().strip()
    if d.get("tender_mainCpv") is not None:
        d["tender_mainCpv"] = str(d["tender_mainCpv"]).strip()
    X = pd.DataFrame([d])

    if "tender_mainCpv" in X.columns and X["tender_mainCpv"].notna().any():
        div2, grp3 = _derive_cpv_parts(X["tender_mainCpv"])
        X["cpv_div2"], X["cpv_grp3"] = div2, grp3

    feat_list = list(features.get("features", []))
    cat_list = set(features.get("categorical", []))

    for c in feat_list:
        if c not in X.columns:
            X[c] = np.nan
    if feat_list:
        X = X[feat_list]

    # log fixes
    if "tender_estimatedPrice_EUR" in X.columns and "tender_estimatedPrice_EUR_log" in X.columns:
        base = pd.to_numeric(X["tender_estimatedPrice_EUR"], errors="coerce")
        X["tender_estimatedPrice_EUR_log"] = np.log1p(base)

    if "lot_bidsCount" in X.columns and "lot_bidsCount_log" in X.columns:
        base = pd.to_numeric(X["lot_bidsCount"], errors="coerce")
        X["lot_bidsCount_log"] = np.log1p(base)

    for c in X.columns:
        if c in cat_list:
            X[c] = X[c].astype("string").str.strip().astype("category")
        else:
            if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

    return X

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, tau: Optional[float] = Query(None)):
    _ensure_models_loaded()
    if not (clf and reg_short and reg_long):
        raise HTTPException(status_code=503, detail="Models not loaded")

    tau_days = float(tau) if tau is not None else float(LONG_THR_DEFAULT)
    tau_prob = float(meta.get("tau", 0.5))

    X = _build_dataframe(req)
    p = float(clf.predict(_align_to_booster(X.copy(), clf))[0])
    y_s = float(reg_short.predict(_align_to_booster(X.copy(), reg_short))[0])
    y_l = float(reg_long.predict(_align_to_booster(X.copy(), reg_long))[0])

    if not math.isfinite(p):
        p = 0.0

    yhat = _combine_hard(p, y_s, y_l, tau_prob)
    stage_used = "long_reg" if (p >= tau_prob) else "short_reg"
    risk_flag_point = (yhat >= tau_days)

    return PredictResponse(
        predicted_days=float(yhat),
        risk_flag=bool(risk_flag_point),
        model_used="lgbm_2stage",
        tau_days=float(tau_days),
        p_long=float(p),
        tau_prob=float(tau_prob),
        stage_used=stage_used,
        pred_short=float(y_s),
        pred_long=float(y_l),
        build=BUILD,
    )
