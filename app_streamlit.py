# app_streamlit.py — Early warnings & aggregated risk summaries — ProcureSight
import os, io, zipfile, re, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import traceback

st.set_page_config(page_title="Risk alerts & summaries — ProcureSight", layout="wide")

st.warning("RUNNING: stream_app/app_streamlit.py — BUILD 2025-12-29")


# ---------- helpers ----------
def _slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')

st.set_page_config(page_title="Risk alerts & summaries — ProcureSight", layout="wide")

# --- JSON safety: αντικατάσταση NaN/±inf με None ---
def rows_json_safe_from_list(rows: list[dict]) -> list[dict]:
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notna(df), None)  # NaN -> None
    # Αν έχεις datetime, κάν’ τες strings:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype("datetime64[ns]").astype(str).where(df[c].notna(), None)
    return df.to_dict(orient="records")

# ================== CONFIG ==================
DEFAULT_BASE = "data"
MIN_COUNT_DEFAULT = 100
MIN_SINGLE_EST_PRICE = 221_000
MIN_SINGLE_YEAR      = 2008
MAX_SINGLE_YEAR      = 2025
MIN_SINGLE_BIDS      = 1
MAX_SINGLE_BIDS      = 200
TOP_K_DEFAULT     = 15
HIST_BINS_DEFAULT = 40
CORR_MIN_ABS      = 0.30
PREVIEW_STYLE_MAX_CELLS = 260_000

DEFAULTS = dict(
    base_dir=DEFAULT_BASE, topk=TOP_K_DEFAULT, min_count=MIN_COUNT_DEFAULT,
    api_base="http://127.0.0.1:8000", tau=720
)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ================== THEME ==================
BRAND = {"name":"ProcureSight","primary":"#0B5FFF","secondary":"#1E2A44","accent":"#00C2A8",
         "bg":"#F6F8FB","card":"#FFFFFF","text":"#121619","muted":"#5C6B7D"}
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [BRAND["primary"], BRAND["accent"], "#7A78FF", "#FF7A59", "#2BC255"]

# ================== FASTAPI CONFIG (dynamic) ==================
def _get_api_base() -> str:
    import os, re, streamlit as st

    # 1️⃣ Αν τρέχεις σε Colab (με Cloudflare tunnel)
    if os.path.exists("/content"):
        for path in [
            "/content/cf_api_bg.log",
            "/content/cf_api_log.txt",
            "/content/cf_api.log"
        ]:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                m = re.search(r"https://[0-9A-Za-z\.-]+\.trycloudflare\.com", text)
                if m:
                    return m.group(0).rstrip("/")
            except Exception:
                pass

    # 2️⃣ Αν έχει οριστεί μεταβλητή περιβάλλοντος ή secret
    api_env = os.getenv("API_BASE")
    if api_env:
        return api_env.rstrip("/")
    try:
        if hasattr(st, "secrets") and "API_BASE" in st.secrets:
            return st.secrets["API_BASE"].strip().rstrip("/")
    except Exception:
        pass

    # 3️⃣ Προεπιλεγμένο (Render)
    return "https://procuresight-api.onrender.com"

DEFAULT_API_BASE = _get_api_base()

# 1) cache με dependency στο current_api_base
@st.cache_resource
def get_api_base(current_api_base: str) -> str:
    return current_api_base

def _current_api_base() -> str:
    # παίρνουμε το API από session_state ή από DEFAULT_API_BASE
    return st.session_state.get("api_base", DEFAULT_API_BASE)

# 2) Χρήση ΠΑΝΤΑ με όρισμα
def api_predict(payload: dict, tau: float | None = None) -> dict:
    base = get_api_base(_current_api_base())
    params = {"tau": tau} if tau is not None else {}
    r = requests.post(f"{base}/predict", json=payload, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def api_predict_batch(rows: list[dict], tau: float | None = None) -> list[dict]:
    base = get_api_base(_current_api_base())
    params = {"tau": tau} if tau is not None else {}
    try:
        r = requests.post(f"{base}/predict_batch", json=rows, params=params, timeout=90)
        if r.status_code == 404:
            raise requests.HTTPError("batch endpoint not found", response=r)
        r.raise_for_status()
        return r.json()
    except Exception:
        return [api_predict(d, tau=tau) for d in rows]


# ========= MAPPINGS =========
CPV_MAPPING = {
    "03":"Agricultural products","09":"Petroleum products","14":"Mining products","15":"Food products",
    "18":"Clothing / leather","30":"Machinery / equipment","31":"Electrical machinery","33":"Medical equipment",
    "34":"Transport equipment","35":"Security / defence","38":"Lab / measuring","39":"Furniture / supplies",
    "45":"Construction works","50":"Repair / maintenance services","60":"Transport services",
    "70":"Real estate services","71":"Architectural / engineering","72":"IT services",
    "73":"Research / consulting","79":"Education / training","80":"Health / social services",
    "90":"Sewage / refuse / environment","98":"Miscellaneous services",
}
COUNTRY_MAP = {
    "LT":"Lithuania","IT":"Italy","RO":"Romania","ES":"Spain","BG":"Bulgaria","GR":"Greece","PT":"Portugal",
    "PL":"Poland","FR":"France","DE":"Germany","CZ":"Czechia","SK":"Slovakia","HU":"Hungary","NL":"Netherlands",
    "BE":"Belgium","AT":"Austria","IE":"Ireland","DK":"Denmark","SE":"Sweden","FI":"Finland","EE":"Estonia",
    "LV":"Latvia","SI":"Slovenia","HR":"Croatia","LU":"Luxembourg","MT":"Malta","CY":"Cyprus"
}
NAME_TO_CODE = {v.strip().upper(): k for k, v in COUNTRY_MAP.items()}

def normalize_iso2_country(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if not s: return np.nan
    s_up = s.upper()
    if len(s_up) == 2 and s_up.isalpha(): return s_up
    return NAME_TO_CODE.get(s_up, s_up)

PROC_MAP = {
    "OPEN": "Open","RESTRICTED": "Restricted","NEGOTIATED_WITH_PUBLICATION": "Negotiated with publication",
    "NEGOTIATED_WITHOUT_PUBLICATION": "Negotiated without publication","COMPETITIVE_DIALOG": "Competitive dialogue",
    "APPROACHING_BIDDERS": "Approaching bidders","NEGOTIATED": "Negotiated",
    "OUTRIGHT_AWARD": "Outright award","OTHER": "Other","CONCESSION": "Concession",
}

# ================== UI helpers ==================
def banner(msg: str, variant: str = "info"):
    colors = {
        "info":("#E8F0FF","#CFE0FF","#0B3BAA"),
        "ok":  ("#E7FAF6","#C7F2EA","#0B7D70"),
        "warn":("#FFF7E6","#FFE4B3","#8A5A00"),
        "error":("#FDE8E8","#FACDCD","#8A1111"),
    }.get(variant, ("#E8F0FF","#CFE0FF","#0B3BAA"))
    st.markdown(
        f"""<div style="background:{colors[0]};border:1px solid {colors[1]};color:{colors[2]};
        padding:.75rem 1rem;border-radius:10px;font-size:.95rem;">{msg}</div>""",
        unsafe_allow_html=True
    )

def risk_pct(s: pd.Series) -> float: return 100.0 * s.astype(bool).mean()
def coerce_bool_col(s: pd.Series) -> pd.Series: return s.astype(str).str.lower().isin(["true","1","yes","y","t"])

def safe_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _safe_labels(df: pd.DataFrame, cols) -> pd.Series:
    if isinstance(cols, str):
        cols = [cols]
    elif not isinstance(cols, (list, tuple, pd.Index)):
        cols = list(cols) if cols is not None else []
    if len(cols) == 0:
        return pd.Series([f"Group {i+1}" for i in range(len(df))], index=df.index)
    sub = df[cols]
    if isinstance(sub, pd.Series):
        sub = sub.to_frame()
    return sub.astype(str).agg(" — ".join, axis=1)

def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "tender_country" in df.columns:
        df["tender_country"] = df["tender_country"].apply(normalize_iso2_country).astype(str).str.strip()
        df["country_name"] = df["tender_country"].map(COUNTRY_MAP).fillna(df["tender_country"])
    else:
        df["country_name"] = "Unknown"
    if "buyer_country" in df.columns:
        df["buyer_country"] = df["buyer_country"].apply(normalize_iso2_country)
        df["buyer_country_name"] = df["buyer_country"].map(COUNTRY_MAP).fillna(df["buyer_country"])
    if "cpv_div2" in df.columns:
        df["cpv_div2"] = pd.to_numeric(df["cpv_div2"], errors="coerce").astype("Int64")
        df["cpv_div2_str"] = df["cpv_div2"].astype("Int64").astype(str).str.zfill(2)
        df["cpv_category"] = df["cpv_div2_str"].map(CPV_MAPPING).fillna("Other/Unknown")
    else:
        df["cpv_category"] = "Other/Unknown"
    if "tender_procedureType" in df.columns:
        df["procedure_label"] = df["tender_procedureType"].map(PROC_MAP).fillna(df["tender_procedureType"])
    else:
        df["procedure_label"] = "Unknown"
    if "tender_supplyType" not in df.columns:
        if "cpv_div2" in df.columns:
            div = df["cpv_div2"].fillna(-1).astype(int)
            cond_works = div.eq(45); cond_sup = div.between(3,44); cond_serv = (~cond_works)&(~cond_sup)&div.ge(50)
            df["inferred_supplyType"] = np.select([cond_works,cond_sup,cond_serv],
                                                  ["WORKS","SUPPLIES","SERVICES"], default="UNKNOWN")
        else:
            df["inferred_supplyType"] = "UNKNOWN"
    if {"cpv_category","cpv_div2"}.issubset(df.columns):
        df["cpv_div2_label"] = df.apply(
            lambda r: f"{r['cpv_category']} (div {int(r['cpv_div2']):02d})" if pd.notna(r.get("cpv_div2"))
            else r["cpv_category"], axis=1
        )
    else:
        df["cpv_div2_label"] = df.get("cpv_category", "Other/Unknown")
    if {"cpv_category","cpv_div2","cpv_grp3"}.issubset(df.columns):
        def _grp_lbl(r):
            d = f"(div {int(r['cpv_div2']):02d})" if pd.notna(r.get("cpv_div2")) else ""
            g = f"grp {int(r['cpv_grp3']):03d}" if pd.notna(r.get("cpv_grp3")) else "grp —"
            return f"{r['cpv_category']} {d} / {g}"
        df["cpv_grp3_label"] = df.apply(_grp_lbl, axis=1)
    else:
        df["cpv_grp3_label"] = df.get("cpv_category", "Other/Unknown")
    return df

def status_pill(text: str, variant: str = "info"):
    colors = {"ok":("#E7FAF6","#0B7D70"), "warn":("#FFF7E6","#8A5A00"), "error":("#FDE8E8","#8A1111"), "info":("#E8F0FF","#0B3BAA")}.get(variant, ("#E8F0FF","#0B3BAA"))
    return st.markdown(
        f"""<span style="display:inline-flex;align-items:center;gap:.45rem;padding:.25rem .55rem;border-radius:999px;
        background:{colors[0]};color:{colors[1]};font-weight:700;font-size:.85rem;border:1px solid #00000010;">
        <span style="width:.55rem;height:.55rem;border-radius:50%;background:{colors[1]};display:inline-block"></span>{text}</span>""",
        unsafe_allow_html=True
    )

def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""<div style="background:{BRAND['card']};border:1px solid #e9edf5;border-radius:14px;padding:14px 16px;box-shadow:0 1px 2px #00000008;">
        <div style="font-size:.80rem;color:{BRAND['muted']};font-weight:700">{title}</div>
        <div style="font-size:1.6rem;font-weight:800;color:{BRAND['text']};margin:.15rem 0">{value}</div>
        <div style="font-size:.80rem;color:{BRAND['muted']}">{sub}</div></div>""",
        unsafe_allow_html=True
    )

def chip(text: str):
    st.markdown(f"""<span style="background:#f3f6ff;border:1px solid #dfe6ff;color:#1b3ea6;
    padding:.20rem .55rem;border-radius:999px;font-weight:700;font-size:.8rem">{text}</span>""", unsafe_allow_html=True)

st.markdown("""
<style>
.sticky-toolbar{position:sticky;top:0;z-index:10;background:rgba(246,248,251,.88);
backdrop-filter:saturate(180%) blur(6px);border-bottom:1px solid #eef1f6;padding:.4rem .6rem;margin:-.5rem -1rem .75rem}
</style>
""", unsafe_allow_html=True)

# ---------- aggregated labeling ----------
def add_cpv_labels_for_aggregated(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df2 = df.copy()
    if "cpv_div2" not in df2.columns and "cpv_grp3" in df2.columns:
        _grp = pd.to_numeric(df2["cpv_grp3"], errors="coerce").astype("Int64")
        df2["cpv_div2"] = (_grp // 10).astype("Int64")
    if "cpv_div2" in df2.columns:
        df2["cpv_div2"] = pd.to_numeric(df2["cpv_div2"], errors="coerce").astype("Int64")
        div_str = df2["cpv_div2"].astype("Int64").astype(str).str.zfill(2)
        div_name = div_str.map(CPV_MAPPING).fillna("Other/Unknown")
        df2["CPV Division"] = div_name + " (div " + div_str + ")"
    if "cpv_grp3" in df2.columns:
        df2["cpv_grp3"] = pd.to_numeric(df2["cpv_grp3"], errors="coerce").astype("Int64")
        grp_str = df2["cpv_grp3"].astype("Int64").astype(str).str.zfill(3)
        base = df2["CPV Division"] if "CPV Division" in df2.columns else pd.Series("CPV", index=df2.index)
        df2["CPV Group"] = base + " / grp " + grp_str
    display_cols: list[str] = []
    if "CPV Group" in df2.columns: display_cols.append("CPV Group")
    if "CPV Division" in df2.columns and "CPV Group" not in display_cols: display_cols.append("CPV Division")
    ignore = {"cpv_div2","cpv_grp3","RiskPct","riskpct","risk_pct","risk%","Count","count"}
    for c in df2.columns:
        if c in ignore: continue
        if c not in display_cols and not np.issubdtype(df2[c].dtype, np.number):
            display_cols.append(c)
    return df2, display_cols

def rank_table(df: pd.DataFrame, by_cols, min_count: int) -> pd.DataFrame:
    if isinstance(by_cols, str): by_cols = [by_cols]
    out = (
        df.groupby(by_cols, dropna=False)
          .agg(RiskPct=("risk_flag", risk_pct), Count=("risk_flag", "size"))
          .query("Count >= @min_count")
          .sort_values(["RiskPct","Count"], ascending=[False, False])
    )
    return out

def plot_risk_vs_count(tbl: pd.DataFrame, title="Risk% & Count (Top-K)", count_in_thousands: bool = True):
    dfp = tbl.reset_index()
    idx_cols = [c for c in dfp.columns if c not in ("RiskPct","Count")]
    x = _safe_labels(dfp, idx_cols)
    risk = dfp["RiskPct"].astype(float)
    cnt  = dfp["Count"].astype(float) / (1000.0 if count_in_thousands else 1.0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=risk, name="Risk%", yaxis="y1",
                         hovertemplate="<b>%{x}</b><br>Risk%: %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=cnt,  name=f"Count{' (K)' if count_in_thousands else ''}", yaxis="y2", opacity=0.60,
                         hovertemplate="<b>%{x}</b><br>Count: %{y:,.1f}"+("K" if count_in_thousands else "")+"<extra></extra>"))
    fig.update_layout(title=title, xaxis=dict(title="Category"), yaxis=dict(title="Risk %", range=[0,100]),
                      yaxis2=dict(title=f"Count{' (K)' if count_in_thousands else ''}", overlaying="y", side="right"),
                      barmode="group", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{_slug(title)}")

def build_zip_of_tables(tables: dict, zip_name: str = "summaries.zip") -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, df_tbl in tables.items():
            csv_bytes = df_tbl.reset_index().to_csv(index=False).encode("utf-8")
            zf.writestr(f"{key}.csv", csv_bytes)
    bio.seek(0)
    return bio.read()

FRIENDLY_COLS = {
    "tender_id":"Tender ID", "tender_year":"Year", "tender_country":"Country code",
    "buyer_country":"Buyer country", "country_name":"Country","buyer_country_name":"Buyer country (name)",
    "p_long_ge720":"P(long≥720)", "risk_flag":"High-risk?", "predicted_days":"Predicted days",
    "threshold_tau":"Threshold", "stage_used":"Stage",
    "tender_supplyType":"SupplyType", "inferred_supplyType":"SupplyType",
    "buyer_buyerType":"BuyerType", "tender_mainCpv":"Main CPV",
    "tender_procedureType":"Procedure (raw)", "procedure_label":"Procedure",
    "cpv_grp3":"CPV Group (code)", "cpv_div2":"CPV Division (code)", "cpv_category":"CPV Category Name",
    "cpv_div2_label":"CPV Division", "cpv_grp3_label":"CPV Group",
    "riskpct":"Risk%", "risk_pct":"Risk%", "risk%":"Risk%", "count":"Count",
}
def friendly_rename_df(df: pd.DataFrame) -> pd.DataFrame:
    m = {c: FRIENDLY_COLS.get(c, FRIENDLY_COLS.get(c.lower(), c)) for c in df.columns}
    return df.rename(columns=m)

# ---------- detectors ----------
def is_model_output_like(df) -> bool:
    needles = ["predicted_days", "risk_flag", "p_long", "threshold_tau", "stage_used", "tau", "model_used"]
    cols_low = {c.lower() for c in df.columns}
    hits = sum(1 for n in needles if any(n in c for c in cols_low))
    return hits >= 2

def is_aggregated_like(df) -> bool:
    cols_low = {c.lower() for c in df.columns}
    has_riskpct = any(k in cols_low for k in ["riskpct","risk%","risk_pct"])
    has_count   = "count" in cols_low
    return has_riskpct and has_count

# ---- File listing & CSV reading ----
def list_csvs(base_dir: str, recursive: bool) -> list[str]:
    out = []
    if not base_dir or not os.path.isdir(base_dir): return out
    if recursive:
        for root, _, files in os.walk(base_dir):
            for name in files:
                if name.lower().endswith(".csv") and not name.startswith(("~",".")):
                    out.append(os.path.join(root, name))
    else:
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if os.path.isfile(full) and name.lower().endswith(".csv") and not name.startswith(("~",".")):
                out.append(full)
    return sorted(out, key=lambda p: os.path.relpath(p, base_dir).lower())

@st.cache_data(show_spinner=False)
def _read_csv_cached_from_bytes(b: bytes):
    return pd.read_csv(io.BytesIO(b), sep=None, engine="python", encoding_errors="ignore")

@st.cache_data(show_spinner=False)
def _read_csv_cached_from_path(path: str, mtime: float):
    return pd.read_csv(path, sep=None, engine="python", encoding_errors="ignore")

def read_csv_any_cached(payload, is_bytes: bool) -> pd.DataFrame:
    return _read_csv_cached_from_bytes(payload) if is_bytes else _read_csv_cached_from_path(payload, os.path.getmtime(payload))

# ---- Correlation (auto) ----
def show_correlation_auto(df: pd.DataFrame, title: str, min_abs: float = CORR_MIN_ABS):
    num = df.select_dtypes(include=[np.number]).copy()
    num.drop(columns=["predicted_days_display"], inplace=True, errors="ignore")
    if num.shape[1] < 3 or num.dropna(how="all").empty: return False
    st.markdown(f"### {title}")
    corr = num.corr(numeric_only=True)
    strength = (corr.abs().sum(axis=1) - 1.0).sort_values(ascending=False)
    corr = corr.loc[strength.index, strength.index]
    z = corr.values.copy()
    mask_lower = np.tril(np.ones_like(z, dtype=bool), k=0)
    z[mask_lower] = np.nan
    z[np.abs(z) < min_abs] = np.nan
    xlab = [FRIENDLY_COLS.get(c, FRIENDLY_COLS.get(c.lower(), c)) for c in corr.columns]
    ylab = [FRIENDLY_COLS.get(c, FRIENDLY_COLS.get(c.lower(), c)) for c in corr.index]
    text = np.where(np.isnan(z), "", np.round(z, 2).astype(str))
    heat = go.Heatmap(z=z, x=xlab, y=ylab, colorscale="RdBu", zmin=-1, zmax=1, showscale=True,
                      colorbar=dict(title="corr"), text=text, texttemplate="%{text}",
                      hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.2f}<extra></extra>")
    fig = go.Figure(heat); fig.update_layout(margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= min_abs:
                pairs.append({"Feature A": FRIENDLY_COLS.get(cols[i], FRIENDLY_COLS.get(cols[i].lower(), cols[i])),
                              "Feature B": FRIENDLY_COLS.get(cols[j], FRIENDLY_COLS.get(cols[j].lower(), cols[j])),
                              "Correlation": float(val), "|Correlation|": float(abs(val))})
    if pairs:
        pairs = sorted(pairs, key=lambda d: d["|Correlation|"], reverse=True)
        st.caption(f"Top correlated feature pairs (|corr| ≥ {min_abs:.2f}).")
        st.dataframe(pd.DataFrame(pairs)[["Feature A","Feature B","Correlation"]], use_container_width=True)
    return True

# ================== UI HEADER ==================
c1, c2 = st.columns([1,8])
with c1:
    st.markdown(
        f'<div style="width:64px;height:64px;border-radius:14px;background:{BRAND["primary"]};'
        f'display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:22px;">PS</div>',
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"<div style='margin-left:.5rem'><h1 style='margin-bottom:.2rem'>Risk alerts & summaries — {BRAND['name']}</h1>"
        f"<div class=\"small-muted\">EU procurement analytics demo</div></div>", unsafe_allow_html=True
    )
st.write("")

# ---------- Sidebar: Data source ----------
if "base_dir" not in st.session_state: st.session_state["base_dir"] = DEFAULT_BASE
if "scan_subfolders" not in st.session_state: st.session_state["scan_subfolders"] = True
with st.sidebar:
    st.header("Data source")
    base_dir = st.text_input("Base directory", value=st.session_state["base_dir"], help="Folder to scan for CSV files.")
    scan_recursive = st.checkbox("Scan subfolders", value=st.session_state["scan_subfolders"])
    show_agg = st.checkbox("Include aggregated tables in picker", value=False)
    st.session_state["base_dir"] = base_dir
    st.session_state["scan_subfolders"] = scan_recursive

def is_aggregated_csv(path: str) -> bool:
    try:
        small = pd.read_csv(path, nrows=3)
        cols = {c.lower() for c in small.columns}
        return bool({"riskpct","risk%","risk_pct","count"} & cols)
    except Exception:
        return False

file_list_all = list_csvs(base_dir, recursive=scan_recursive)
file_list = file_list_all if show_agg else [p for p in file_list_all if not is_aggregated_csv(p)]
if not file_list:
    st.info("No CSV files found in the folder. Change the 'Base directory' or enable scanning subfolders.")
def _format_path(p: str) -> str:
    if p == "— Select from folder —": return p
    try: return os.path.relpath(p, base_dir)
    except Exception: return os.path.basename(p)

# ---------- STYLE: ίδιες επικεφαλίδες και «card» εμφάνιση ----------
st.markdown("""
<style>
.section-title {font-size:1.05rem; font-weight:700; margin:.5rem 0 .35rem;}
.block {padding:.75rem; border:1px solid #e9edf5; border-radius:10px; background:#fff;}
</style>
""", unsafe_allow_html=True)

# ---------- PICK FROM FOLDER ----------
st.markdown('<div class="section-title">Pick a CSV from folder</div>', unsafe_allow_html=True)
pick = st.selectbox(
    label="",
    options=["— Select from folder —"] + file_list,
    index=0,
    format_func=_format_path,
    key="pick_csv_from_folder"
)

# ---------- UPLOAD ----------
st.markdown('<div class="section-title">Upload your CSV file</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(label="", type=["csv"], key="upload_csv")

# ---------- Επιλογή πηγής (uploaded έχει προτεραιότητα) ----------
has_uploaded = uploaded_file is not None
has_folder   = bool(pick and pick != "— Select from folder —")

if not (has_uploaded or has_folder):
    st.info("Please upload a CSV file **or** pick one from the folder to begin.")
    st.stop()

# ---------- Διαβάζουμε CSV ----------
label = None
if has_uploaded:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine="python", encoding_errors="ignore")
        label = uploaded_file.name
    except Exception as e:
        st.error(f"❌ Failed to read CSV file: {e}")
        st.stop()
else:
    try:
        df_raw = read_csv_any_cached(pick, is_bytes=False)
        label = _format_path(pick)
    except Exception as e:
        st.error(f"❌ Could not read CSV from disk: {e}")
        st.stop()

if df_raw.empty:
    banner("CSV is empty.", "warn")
    st.stop()

banner(f"Loaded: <b>{label}</b> — rows: <b>{len(df_raw):,}</b>", "ok")

# ===== CSV schema validation =====
cols_lower = {c.lower() for c in df_raw.columns}
looks_like_outputs   = any(k in cols_lower for k in ["predicted_days", "risk_flag", "p_long", "tau", "model_used"])
looks_like_aggregated= ({"riskpct","risk%","risk_pct"} & cols_lower) and ("count" in cols_lower)

if not looks_like_outputs and not looks_like_aggregated:
    REQUIRED_MIN = {"tender_country","tender_mainCpv","tender_year"}
    NICE_TO_HAVE = {"tender_procedureType","tender_supplyType","tender_estimatedPrice_EUR","lot_bidsCount"}

    missing = [c for c in REQUIRED_MIN if c not in df_raw.columns]
    if missing:
        banner(
            "❌ This CSV doesn’t match the expected procurement inputs.<br>"
            f"Missing required columns: <b>{', '.join(missing)}</b>.<br>"
            "Please upload a CSV with the required fields (e.g., tender_country, tender_mainCpv, tender_year).",
            "error",
        )
        st.stop()

    missing_nice = [c for c in NICE_TO_HAVE if c not in df_raw.columns]
    if missing_nice:
        banner(
            "ℹ️ Some useful columns are missing and will be assumed/derived if possible: "
            f"<b>{', '.join(missing_nice)}</b>.",
            "warn",
        )

# --- classify dataset type ---
is_early_warning = False; is_early = False; looks_agg = False
cols_lower_list = [c.lower() for c in df_raw.columns]
if any(("p_long" in c) or ("risk_flag" in c) for c in cols_lower_list):
    is_early_warning = True; is_early = True
elif any(("riskpct" in c) or ("risk%" in c) for c in cols_lower_list):
    looks_agg = True
is_early = ("predicted_days" in df_raw.columns) or ("risk_flag" in df_raw.columns)
lower = {c.lower(): c for c in df_raw.columns}
looks_agg = looks_agg or bool((lower.get("riskpct") or lower.get("risk%") or lower.get("risk_pct")) and lower.get("count"))

# ======== SAFE DEFAULTS + SECTIONS (χωρίς KeyError) ========
for k, v in {
    "show_preview": True,
    "show_rankings": True,
    "show_hist": False,
    "show_pairplots": False,
    "show_corr": False,
}.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.header("Sections")

    if is_early_warning:
        show_preview  = st.checkbox("Show preview table",                 key="show_preview")
        show_rankings = st.checkbox("Show rankings & charts",             key="show_rankings")

        st.caption("Early-warning dataset detected: choose which advanced charts to show.")
        show_hist      = st.checkbox("Show histogram/ECDF",               key="show_hist")
        show_pairplots = st.checkbox("Show Predicted-by Country/Procedure", key="show_pairplots")
        show_corr      = st.checkbox("Show correlation matrix",           key="show_corr")
    else:
        st.caption("Advanced visualizations available only for early-warning datasets.")
        st.checkbox("Show histogram/ECDF", value=False, disabled=True)
        st.checkbox("Show Predicted-by Country/Procedure", value=False, disabled=True)
        st.checkbox("Show correlation matrix", value=False, disabled=True)

        show_preview, show_rankings = True, True
        show_hist = show_pairplots = show_corr = False

        st.session_state.update(
            show_preview=True, show_rankings=True,
            show_hist=False, show_pairplots=False, show_corr=False
        )

    if st.button("Reset"):
        st.session_state.update({
            "base_dir": DEFAULT_BASE,
            "pick_file_path": None,
            "scan_subfolders": True,
            "topk": TOP_K_DEFAULT,
            "mincnt": MIN_COUNT_DEFAULT,
            "bins": HIST_BINS_DEFAULT,
            "show_preview": True,
            "show_rankings": True,
            "show_hist": False,
            "show_pairplots": False,
            "show_corr": False,
        })
        st.rerun()

show_preview   = st.session_state.get("show_preview", True)
show_rankings  = st.session_state.get("show_rankings", True)
show_hist      = st.session_state.get("show_hist", False)
show_pairplots = st.session_state.get("show_pairplots", False)
show_corr      = st.session_state.get("show_corr", False)

def _top_title(base: str, total: int, k: int) -> str:
    k = int(k)
    return f"{base} (All)" if k >= int(total) else f"{base} (Top-{k})"

# ================== EARLY-WARNING WORKFLOW ==================
if is_early:
    df = df_raw.copy()
    if "risk_flag" in df.columns and df["risk_flag"].dtype != bool:
        df["risk_flag"] = coerce_bool_col(df["risk_flag"])
    if "risk_flag" not in df.columns:
        df["risk_flag"] = (pd.to_numeric(df.get("predicted_days", pd.Series(index=df.index)), errors="coerce") >= 720)
    df = safe_numeric(df, ["predicted_days","tender_year","cpv_div2","cpv_grp3"])
    df = derive_labels(df)
    if "predicted_days" in df.columns:
        df["predicted_days_display"] = df["predicted_days"].round().astype("Int64")
    if "buyer_country" in df.columns and "tender_country" in df.columns:
        same = (df["buyer_country"].fillna("").astype(str).str.upper()
                == df["tender_country"].fillna("").astype(str).str.upper()).all()
        if same:
            df.drop(columns=["buyer_country"], inplace=True, errors="ignore")
    for c in ["country_name","cpv_category","procedure_label","cpv_div2_label","cpv_grp3_label"]:
        if c in df.columns: df[c] = df[c].astype("category")

    st.divider(); st.subheader("Filters")
    st.caption("Refine the dataset before computing rankings and charts.")
    c1, c2, c3, c4 = st.columns([1.3,1.3,1.3,1])
    with c1:
        countries = sorted(df["country_name"].dropna().astype(str).str.strip().unique().tolist()) if "country_name" in df else []
        sel_countries = st.multiselect("Countries", countries, default=[])
    with c2:
        cpv_cats = sorted(df["cpv_category"].dropna().unique().tolist()) if "cpv_category" in df else []
        sel_cpv = st.multiselect("CPV categories (2-digit Divisions)", cpv_cats, default=[])
    with c3:
        procs = sorted(df["procedure_label"].dropna().unique().tolist()) if "procedure_label" in df else []
        sel_proc = st.multiselect("Procedure types", procs, default=[])
    with c4:
        only_risk = st.toggle("High-risk only (≥ 720 days)", value=False)

    opt = st.selectbox("Quick preset (Supply type)", ["— none —","Supplies","Services","Works"], index=0)
    preset_mask = pd.Series(True, index=df.index)
    if opt != "— none —":
        col = "tender_supplyType" if "tender_supplyType" in df.columns else "inferred_supplyType"
        want = opt.upper()
        preset_mask &= df[col].astype(str).str.upper().eq(want)

    mask = pd.Series(True, index=df.index)
    if sel_countries: mask &= df["country_name"].isin(sel_countries)
    if sel_cpv:       mask &= df["cpv_category"].isin(sel_cpv)
    if sel_proc:      mask &= df["procedure_label"].isin(sel_proc)
    if only_risk:     mask &= df["risk_flag"].astype(bool)

    df_f = df.loc[mask & preset_mask].copy()
    if df_f.empty:
        banner("No rows match the current filters. Relax the selections.", "warn"); st.stop()

    st.divider(); st.subheader("Key indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Filtered rows", f"{len(df_f):,}")
    with k2: st.metric("Risk (share of high-risk rows)", f"{(100.0*df_f['risk_flag'].mean()):.2f}%")
    with k3: st.metric("Median predicted days", f"{df_f['predicted_days'].median():.0f}" if "predicted_days" in df_f else "—")
    with k4: st.metric("Count ≥ 720 days", f"{int(df_f['risk_flag'].sum()):,}")

    if show_preview:
        st.subheader("Preview of filtered records")
        st.caption("Showing all N rows. Use the download to export everything.")
        drop_cols_redundant = ["tender_procedureType","cpv_div2_str","cpv_div2_label","cpv_grp3_label","CPV Division","CPV Group"]
        df_prev = df_f.drop(columns=[c for c in drop_cols_redundant if c in df_f.columns], errors="ignore")
        order = ["tender_id","tender_year","tender_country","buyer_country","buyer_country_name","country_name",
                 "p_long_ge720","p_long_ge_tau","risk_flag","predicted_days","threshold_tau","stage_used",
                 "tender_supplyType","inferred_supplyType","buyer_buyerType","procedure_label",
                 "tender_mainCpv","cpv_grp3","cpv_div2","cpv_category"]
        front = [c for c in order if c in df_prev.columns]; rest = [c for c in df_prev.columns if c not in front]
        df_prev = df_prev[front + rest]
        disp = friendly_rename_df(df_prev.copy())
        if "Year" in disp.columns:
            disp["Year"] = pd.to_numeric(disp["Year"], errors="coerce").astype("Int64")
        for c in ["P(long≥720)","P(long≥τ)","p_long_ge720","p_long_ge_tau"]:
            if c in disp.columns:
                disp.drop(columns=[c], inplace=True, errors="ignore")
        if "predicted_days_display" in df_prev.columns:
            disp["Predicted days"] = df_prev["predicted_days_display"]
            disp.drop(columns=["predicted_days_display"], inplace=True, errors="ignore")

        def _show_df(df_to_show: pd.DataFrame):
            if "Predicted days" in df_to_show.columns:
                styler = df_to_show.style.set_properties(
                    subset=["Predicted days"],
                    **{"background-color":"#FFF5B3","font-weight":"800","border":"1px solid #E5C365"}
                )
                st.dataframe(styler, use_container_width=True)
            else:
                st.dataframe(df_to_show, use_container_width=True)

        total_cells = int(disp.shape[0] * disp.shape[1])
        if total_cells > PREVIEW_STYLE_MAX_CELLS:
            max_rows = max(1, PREVIEW_STYLE_MAX_CELLS // max(1, disp.shape[1])); _show_df(disp.head(max_rows))
            st.caption(f"Preview limited to {max_rows:,} rows for performance. Download includes all {len(disp):,} rows.")
        else:
            _show_df(disp)
        csv_bytes = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="filtered_early_warning.csv",
                           mime="text/csv", use_container_width=True)

        # ---------------- Rankings & charts (EARLY ONLY) ----------------
    if show_rankings:
        st.divider()
        st.subheader("Rankings & charts")
        st.caption(
            "Risk% = percentage of high-risk rows per category. "
            "Count = number of rows per category (minimum count applies)."
        )
        # πόσα distinct groups υπάρχουν ΜΕΤΑ τα φίλτρα
        # CPV groups
        n_cpv_groups = 0
        if {"cpv_category", "cpv_div2", "cpv_grp3"}.issubset(df_f.columns):
            n_cpv_groups = int(
                df_f[["cpv_category", "cpv_div2", "cpv_grp3"]]
                .drop_duplicates()
                .shape[0]
            )
        elif "cpv_category" in df_f.columns:
            n_cpv_groups = int(df_f["cpv_category"].nunique())

        # χώρες
        n_countries = int(df_f["country_name"].nunique()) if "country_name" in df_f.columns else 0

        # διαδικασίες
        n_procs = int(df_f["procedure_label"].nunique()) if "procedure_label" in df_f.columns else 0

        # ✅ helper: παίρνει topk/mincnt σαν ορίσματα, ΔΕΝ χρησιμοποιεί ew_topk/ew_mincnt
        def render_section(
            section_title,
            by_cols,
            explain=None,
            rename_cols=None,
            topk_val=None,
            mincnt_val=None,
        ):
            st.markdown(f"### {section_title}")
            if explain:
                st.caption(explain)

            mc = int(mincnt_val if mincnt_val is not None else MIN_COUNT_DEFAULT)
            tbl = rank_table(df_f, by_cols, mc)
            if tbl.empty:
                st.info("No categories meet the minimum count threshold.")
                return

            df_disp = tbl.reset_index()
            if rename_cols:
                df_disp = df_disp.rename(columns=rename_cols)
            df_disp["RiskPct"] = df_disp["RiskPct"].map(lambda v: f"{v:.2f}%")
            st.dataframe(friendly_rename_df(df_disp).head(100), use_container_width=True)

            k_eff = min(int(topk_val if topk_val is not None else TOP_K_DEFAULT), len(tbl))
            top_tbl = tbl.head(k_eff).copy().iloc[::-1]
            plot_risk_vs_count(
                top_tbl,
                title=f"{section_title}: " + _top_title("Risk% & Count", len(tbl), k_eff),
            )

        # ---- CPV display settings ----
        st.markdown("#### CPV display settings")
        colA1, colA2 = st.columns(2)
        with colA1:
            max_cpv = max(1, n_cpv_groups)
            default_topk_cpv = min(TOP_K_DEFAULT, max_cpv)
            ew_topk_cpv = st.number_input(
                "Top-K (CPV)",
                min_value=1,
                max_value=max_cpv,
                value=default_topk_cpv,
                step=1,
                key="ew_topk_cpv",
            )

        with colA2:
            ew_mincnt_cpv = st.number_input(
                "Min count (CPV)",
                1,
                100_000,
                MIN_COUNT_DEFAULT,
                step=10,
                key="ew_mincnt_cpv",
            )

        render_section(
            "CPV sectors & groups (Division 2-digit, Group 3-digit)",
            ["cpv_category", "cpv_div2", "cpv_grp3"],
            "- Division (2-digit) maps to sector (e.g., 33 → Medical equipment).\n"
            "- Group (3-digit) within that division (e.g., 331 → Medical imaging).",
            {"cpv_category": "CPV Category Name",
             "cpv_div2": "CPV Division (code)",
             "cpv_grp3": "CPV Group (code)"},
            topk_val=ew_topk_cpv,
            mincnt_val=ew_mincnt_cpv,
        )

        # ---- Countries display settings ----
        st.markdown("#### Countries display settings")
        colB1, colB2 = st.columns(2)
        with colB1:
            max_cty = max(1, n_countries)
            default_topk_cty = min(TOP_K_DEFAULT, max_cty)
            ew_topk_cty = st.number_input(
                "Top-K (Countries)",
                min_value=1,
                max_value=max_cty,
                value=default_topk_cty,
                step=1,
                key="ew_topk_cty",
            )

        with colB2:
            ew_mincnt_cty = st.number_input(
                "Min count (Countries)",
                1,
                100_000,
                MIN_COUNT_DEFAULT,
                step=10,
                key="ew_mincnt_cty",
            )

        render_section(
            "Countries",
            ["country_name"],
            "Country of the tender (mapped from ISO code).",
            {"country_name": "Country"},
            topk_val=ew_topk_cty,
            mincnt_val=ew_mincnt_cty,
        )

        # ---- Procedures display settings ----
        st.markdown("#### Procedures display settings")
        colC1, colC2 = st.columns(2)
        with colC1:
            max_proc = max(1, n_procs)
            default_topk_proc = min(TOP_K_DEFAULT, max_proc)
            ew_topk_proc = st.number_input(
                "Top-K (Procedures)",
                min_value=1,
                max_value=max_proc,
                value=default_topk_proc,
                step=1,
                key="ew_topk_proc",
            )

        with colC2:
            ew_mincnt_proc = st.number_input(
                "Min count (Procedures)",
                1,
                100_000,
                MIN_COUNT_DEFAULT,
                step=10,
                key="ew_mincnt_proc",
            )

        render_section(
            "Procedure types",
            ["procedure_label"],
            "Standard TED procedure types.",
            {"procedure_label": "Procedure"},
            topk_val=ew_topk_proc,
            mincnt_val=ew_mincnt_proc,
        )

    # ---------------- Histogram + ECDF (early-warning only) ----------------
    if show_hist and "predicted_days" in df_f.columns:
        st.markdown("### Distribution of predicted duration (days)")
        st.caption("Binned distribution of predicted days (clipped to [0, 1800]).")

        bins = st.number_input(
            "Histogram bins",
            min_value=10, max_value=200, value=HIST_BINS_DEFAULT, step=5, key="ew_bins"
        )

        cuts = pd.cut(df_f["predicted_days"].clip(0, 1800), bins=int(bins))
        hist = cuts.value_counts().sort_index()
        labels = [f"[{int(iv.left)}–{int(iv.right)}]" for iv in hist.index.to_list()]
        hist_df = pd.DataFrame({"bin": labels, "count": hist.values}).set_index("bin")
        st.bar_chart(hist_df["count"], use_container_width=True)

        st.markdown("#### Cumulative distribution (ECDF)")
        st.caption(
            "The **ECDF** shows the fraction of tenders with predicted duration ≤ X days. "
            "It rises from 0 to 1; a steeper rise means predictions are concentrated."
        )
        vals = np.sort(df_f["predicted_days"].dropna().values)
        if len(vals) > 0:
            y = np.arange(1, len(vals)+1) / len(vals)
            fig_ecdf = go.Figure(go.Scatter(
                x=vals, y=y, mode="lines",
                hovertemplate="Predicted: %{x:.2f} days<br>ECDF: %{y:.2f}<extra></extra>"
            ))
            fig_ecdf.update_layout(
                xaxis=dict(title="Predicted days"),
                yaxis=dict(title="ECDF", range=[0, 1]),
                margin=dict(t=20)
            )
            st.plotly_chart(fig_ecdf, use_container_width=True)
            p10, p50, p90 = np.percentile(vals, [10, 50, 90])
            st.caption(f"p10={p10:.0f}, p50={p50:.0f}, p90={p90:.0f}; mean={np.mean(vals):.1f}")

        # --- helper έξω από το if ---
        def plot_predicted_by_category(df_in: pd.DataFrame, cat_col: str, top_k: int = 12):
            if "predicted_days" not in df_in.columns or cat_col not in df_in.columns:
                return
            tmp = df_in[[cat_col, "predicted_days"]].dropna()
            if tmp.empty:
                return
            top_vals = tmp[cat_col].astype(str).value_counts().head(top_k).index.tolist()
            tmp = tmp[tmp[cat_col].astype(str).isin(top_vals)].copy()
            tmp[cat_col] = tmp[cat_col].astype(str)

            fig = px.box(tmp, x=cat_col, y="predicted_days", points=False)
            med = tmp.groupby(cat_col)["predicted_days"].median().reindex(tmp[cat_col].unique())
            fig.add_trace(go.Scatter(
                x=med.index.tolist(), y=med.values.tolist(), mode="lines+markers",
                name="Median", hovertemplate="<b>%{x}</b><br>Median: %{y:.0f} days<extra></extra>"
            ))
            x_title = "Country" if cat_col == "country_name" else ("Procedure" if cat_col == "procedure_label" else cat_col)
            fig.update_layout(xaxis_title=x_title, yaxis_title="Predicted days", margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True, key=f"pairplot_{_slug(cat_col)}")

        # --- Country/Procedure pairplots ---
        if show_pairplots and "predicted_days" in df_f.columns:
            st.markdown("### Predicted days by Country / Procedure")
            col1, col2 = st.columns(2)
            n_countries = int(df_f["country_name"].nunique()) if "country_name" in df_f.columns else 0
            n_procs     = int(df_f["procedure_label"].nunique()) if "procedure_label" in df_f.columns else 0

            with col1:
                topk_countries = st.number_input(
                    "Top-K Countries (by rows)", min_value=1, max_value=max(1, n_countries),
                    value=max(1, min(12, n_countries)), step=1, key="topk_countries"
                )
                st.caption("Shows distribution of predicted days for the most frequent countries.")
                if n_countries > 0:
                    plot_predicted_by_category(df_f, "country_name", top_k=int(topk_countries))
                else:
                    st.info("No countries available in the current filter.")

            with col2:
                topk_procs = st.number_input(
                    "Top-K Procedures (by rows)", min_value=1, max_value=max(1, n_procs),
                    value=max(1, min(12, n_procs)), step=1, key="topk_procs"
                )
                st.caption("Shows distribution of predicted days for the most frequent procedures.")
                if n_procs > 0:
                    plot_predicted_by_category(df_f, "procedure_label", top_k=int(topk_procs))
                else:
                    st.info("No procedures available in the current filter.")

        # --- Correlation matrix ---
        if show_corr:
            _ = show_correlation_auto(df_f, title="Correlation matrix", min_abs=CORR_MIN_ABS)


# ================== AGGREGATED WORKFLOW ==================
elif looks_agg:
    lower = {c.lower(): c for c in df_raw.columns}
    rcol = lower.get("riskpct") or lower.get("risk%") or lower.get("risk_pct")
    ccol = lower.get("count")
    df = df_raw.copy()
    df[rcol] = pd.to_numeric(df[rcol], errors="coerce")
    df[ccol] = pd.to_numeric(df[ccol], errors="coerce")
    if pd.notna(df[rcol].max()) and df[rcol].max() <= 1.5:
        df[rcol] = df[rcol] * 100.0

    df2, cat_cols = add_cpv_labels_for_aggregated(df)
    if cat_cols is None: cat_cols = []

    st.divider(); st.subheader("Summary KPIs")
    st.caption("Number of groups, weighted overall Risk%, and highest group Risk%.")
    groups = int(len(df2))
    total_cnt = df2[ccol].sum(skipna=True)
    weighted = (df2[rcol] * df2[ccol]).sum(skipna=True) / total_cnt if pd.notna(total_cnt) and total_cnt != 0 else np.nan
    topgrp = df2[rcol].max(skipna=True)
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Groups", f"{groups:,}")
    with k2: st.metric("Weighted Risk%", f"{weighted:.2f}%" if pd.notna(weighted) else "—")
    with k3: st.metric("Top group Risk%", f"{topgrp:.2f}%" if pd.notna(topgrp) else "—")

    df_rank = df2.copy().sort_values([rcol, ccol], ascending=[False, False]).reset_index(drop=True)
    df_rank.index = df_rank.index + 1
    total_riskw = (df_rank[rcol] * df_rank[ccol]).sum(skipna=True)
    df_rank["Risk Weighted"] = (df_rank[rcol] * df_rank[ccol])
    df_rank["Share Risk"] = (df_rank["Risk Weighted"] / total_riskw).fillna(0.0)
    df_rank["Cum Share Risk"] = df_rank["Share Risk"].cumsum().clip(upper=1.0)
    df_rank["Pareto 80%"] = (df_rank["Cum Share Risk"] <= 0.80)

    st.markdown("### Ranked table (simple)")
    st.caption(
        "Each row represents a category sorted by **Risk%** and **Count**. "
        "The **Pareto 80%** column marks groups that together account for roughly 80% of the total risk — "
        "helping you identify the most influential categories."
    )
    code_col = None; name_col = None
    if "cpv_grp3" in df_rank.columns and "CPV Group" in df_rank.columns:
        code_col, name_col = "cpv_grp3", "CPV Group"
    elif "cpv_div2" in df_rank.columns and "CPV Division" in df_rank.columns:
        code_col, name_col = "cpv_div2", "CPV Division"
    if code_col is not None:
        df_basic = pd.DataFrame({
            "Selected Category (code)": df_rank[code_col],
            "Selected Category (name)": df_rank[name_col],
            "Count (K)": (df_rank[ccol] / 1000.0).round(2),
            "Risk%": df_rank[rcol].map(lambda v: f"{v:.2f}%"),
            "Pareto 80%": df_rank["Pareto 80%"].astype(bool),
        })
    else:
        other_cats = [c for c in df_rank.columns if c not in [rcol, ccol, "Risk Weighted", "Share Risk", "Cum Share Risk", "Pareto 80%"]
                      and not np.issubdtype(df_rank[c].dtype, np.number)]
        labels = _safe_labels(df_rank, other_cats)
        df_basic = pd.DataFrame({
            "Selected Category": labels,
            "Count (K)": (df_rank[ccol] / 1000.0).round(2),
            "Risk%": df_rank[rcol].map(lambda v: f"{v:.2f}%"),
            "Pareto 80%": df_rank["Pareto 80%"].astype(bool),
        })
    st.dataframe(df_basic, use_container_width=True)
    st.download_button("⬇️ Download ranked table (simple CSV)",
                       data=df_basic.to_csv(index=False).encode("utf-8"),
                       file_name="ranked_simple.csv", mime="text/csv", use_container_width=True)

    st.markdown("### Risk vs Count (scatter)")
    logx = st.toggle("Log scale for Count", value=True)
    xvals = df_rank[ccol].astype(float); yvals = df_rank[rcol].astype(float)
    labels = _safe_labels(df_rank, cat_cols)
    fig_sc = go.Figure(go.Scatter(x=xvals, y=yvals, mode="markers", text=labels,
                                  marker=dict(size=9, color=BRAND["primary"], opacity=0.7),
                                  hovertemplate="<b>%{text}</b><br>Count: %{x:,.0f}<br>Risk%: %{y:.2f}%<extra></extra>"))
    fig_sc.update_layout(xaxis=dict(title="Count", type="log" if logx else "linear"),
                         yaxis=dict(title="Risk %", range=[0, 100]), margin=dict(t=40))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption("Each point represents a group; hover to see details.")

    # --- Groups by Risk% & Count section ---
    st.markdown("### Groups by Risk% & Count")
    st.caption("Shows top categories by their risk percentage and record count.")

    # πόσες ομάδες υπάρχουν συνολικά στο aggregated dataset
    n_groups = int(len(df_rank))
    default_topk = min(TOP_K_DEFAULT, n_groups) if n_groups > 0 else TOP_K_DEFAULT

    st.markdown("##### Chart display settings")
    st.markdown("""
<style>
[data-testid="stNumberInput"] > div > div { padding-bottom: .25rem; }
[data-testid="stNumberInput"] label { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)
    c_topk, c_mincnt, _ = st.columns([1, 1, 1], gap="small")

    with c_topk:
        topk = st.number_input(
            "Top-K (max groups to show)",
            min_value=1,
            max_value=max(1, n_groups),
            value=default_topk,
            step=1,
            key="agg_topk",
        )
    with c_mincnt:
        mincnt = st.number_input(
            "Min count (rows)",
            min_value=1, max_value=100_000, value=MIN_COUNT_DEFAULT, step=10, key="agg_mincnt"
        )

    df_rank_f = df_rank[df_rank[ccol] >= int(mincnt)].copy()
    k_eff   = min(int(topk), len(df_rank_f))
    top_tbl = df_rank_f.sort_values([rcol, ccol], ascending=[False, False]).head(k_eff).iloc[::-1]

    if cat_cols:
        sub = top_tbl[cat_cols]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame()
        x = sub.astype(str).agg(" — ".join, axis=1)
    else:
        x = pd.Series([f"Group {i+1}" for i in range(len(top_tbl))], index=top_tbl.index)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=top_tbl[rcol], name="Risk%", yaxis="y1",
                     hovertemplate="<b>%{x}</b><br>Risk%: %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=(top_tbl[ccol]/1000.0), name="Count (K)", yaxis="y2", opacity=0.60,
                     hovertemplate="<b>%{x}</b><br>Count: %{y:,.1f}K<extra></extra>"))
    fig.update_layout(
        xaxis=dict(title="Group"),
        yaxis=dict(title="Risk %", range=[0,100]),
        yaxis2=dict(title="Count (K)", overlaying="y", side="right"),
        barmode="group",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(t=60)
    )
    st.plotly_chart(fig, use_container_width=True, key="chart_groups_risk_count")

    st.markdown("### Cumulative risk vs cumulative count (Pareto view)")
    st.caption("""
    **What this shows (80/20 idea):** how concentrated the total risk is across groups.
    The Pareto curve shows how total risk accumulates across categories. Curves close to the equality line indicate a more even risk distribution, 
    while steeper curves reveal higher risk concentration. 
    The **Gini coefficient** below measures this concentration — higher values mean stronger inequality.
    """)

    # --- Risk-weighted contribution ---
    df_rank["Risk Weighted"] = df_rank[rcol].astype(float) * df_rank[ccol].astype(float)

    # --- Shares (guard against divide-by-zero) ---
    cnt_sum = float(df_rank[ccol].sum()) if pd.notna(df_rank[ccol].sum()) else 0.0
    riskw_sum = float(df_rank["Risk Weighted"].sum()) if pd.notna(df_rank["Risk Weighted"].sum()) else 0.0

    df_rank["_share_cnt"]  = (df_rank[ccol] / cnt_sum).fillna(0.0) if cnt_sum > 0 else 0.0
    df_rank["_share_risk"] = (df_rank["Risk Weighted"] / riskw_sum).fillna(0.0) if riskw_sum > 0 else 0.0

    # ✅ Lorenz curve wants sorting from LOW → HIGH contribution
    df_lorenz = df_rank.sort_values("_share_risk", ascending=True).copy()

    # --- Cumulative shares ---
    df_lorenz["Cum Share Count"] = df_lorenz["_share_cnt"].cumsum().clip(upper=1.0)
    df_lorenz["Cum Share Risk"]  = df_lorenz["_share_risk"].cumsum().clip(upper=1.0)

    # --- Curve arrays ---
    xg = np.r_[0.0, df_lorenz["Cum Share Count"].values]
    yg = np.r_[0.0, df_lorenz["Cum Share Risk"].values]

    # --- Gini (area between equality line and Lorenz curve) ---
    auc = np.trapezoid(yg, xg)
    gini = float(np.clip(1.0 - 2.0 * auc, 0.0, 1.0))

    # labels πρέπει να αντιστοιχούν στη ΣΕΙΡΑ του df_lorenz
    labels_pareto = ["(start)"] + _safe_labels(df_lorenz, cat_cols).tolist()

    # --- Plot ---
    fig_l = go.Figure()
    fig_l.add_trace(go.Scatter(
        x=xg, y=yg, mode="lines+markers", name="Cumulative risk",
        text=labels_pareto,
        hovertemplate="<b>%{text}</b><br>Cum Count: %{x:.1%}<br>Cum Risk: %{y:.1%}<extra></extra>"
    ))
    fig_l.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Equality line",
        line=dict(dash="dash")
    ))
    fig_l.update_layout(
        xaxis=dict(title="Cumulative share of Count", range=[0, 1]),
        yaxis=dict(title="Cumulative share of Risk",  range=[0, 1]),
        margin=dict(t=40)
    )
    st.plotly_chart(fig_l, use_container_width=True)
    st.metric("Gini (risk concentration)", f"{gini:.3f}")

    # (προαιρετικά) correlation για aggregated που δεν είναι “μόνο CPV”
    if show_corr:
        lower_cols = [c.lower() for c in df2.columns]
        cpv_only = (
            ("cpv_div2" in lower_cols or "cpv_grp3" in lower_cols)
            and set(lower_cols).issubset({
                "cpv_div2", "cpv_grp3",
                rcol.lower(), ccol.lower(),
                "cpv division", "cpv group"
            })
        )
        if not cpv_only:
            _ = show_correlation_auto(df2, title="Correlation matrix", min_abs=CORR_MIN_ABS)

# ================== RAW BATCH INPUTS ==================
else:
    st.divider()
    st.subheader("Preview (batch inputs)")
    st.caption("We show a small sample from the batch file you selected. It will be sent to the API from the 'Batch from CSV' tab.")
    prev = friendly_rename_df(df_raw.head(10).copy())
    rename_lite = {"Procedure (raw)": "Procedure","SupplyType":"Supply type","BuyerType":"Buyer type",
                   "Buyer country":"Buyer country","tender_country":"Country code","tender_year":"Year",
                   "tender_mainCpv":"Main CPV","tender_estimatedPrice_EUR":"Estimated price (EUR)",
                   "lot_bidsCount":"Bids count","tender_indicator_score_INTEGRITY":"Integrity score",
                   "tender_indicator_score_ADMINISTRATIVE":"Administrative score",
                   "tender_indicator_score_TRANSPARENCY":"Transparency score"}
    prev = prev.rename(columns={c: rename_lite.get(c, c) for c in prev.columns})
    st.dataframe(prev, use_container_width=True)

# ================== FASTAPI INTEGRATION UI ==================
st.divider()
st.header("⚡ Try the API: get live predictions! ")
api_base_in = st.text_input("API Base URL", value=DEFAULT_API_BASE)

if st.session_state.get("_last_api_base") != api_base_in:
    st.session_state["_last_api_base"] = api_base_in
    try:
        get_api_base.clear()      # Streamlit 1.25+
    except Exception:
        st.cache_resource.clear() # fallback
st.session_state["api_base"] = api_base_in

c_conn = st.columns([1,1,2])
with c_conn[0]:
    if st.button("Test API", use_container_width=True):
        try:
            info = requests.get(f"{api_base_in}/", timeout=10).json()
            ok = bool(info.get("model_loaded", False))
            st.session_state["_api_ok"] = ok
            st.session_state["_api_info"] = info
        except Exception as e:
            st.session_state["_api_ok"] = False
            st.session_state["_api_info"] = {"error": str(e)}
with c_conn[1]:
    def _probe(url: str, tries: int = 3, timeout: float = 8.0) -> tuple[bool, dict]:
        import time
        last_err = None
        for _ in range(tries):
            try:
                r = requests.get(url.rstrip("/") + "/", timeout=timeout)
                if not r.ok:
                    last_err = {"status": r.status_code, "text": r.text[:200]}
                    time.sleep(0.8); continue
                try:
                    js = r.json()
                except Exception:
                    return True, {"note": "reachable (non-JSON response)"}
                return bool(js.get("model_loaded", False)), js
            except Exception as e:
                last_err = {"error": str(e)}; time.sleep(0.8)
        return False, (last_err or {})
    ok, info = _probe(api_base_in)
    status_pill("ready" if ok else "unreachable", "ok" if ok else "error")
with c_conn[2]:
    st.link_button("📘 Open /docs", f"{api_base_in.rstrip('/')}/docs", use_container_width=True)

# Decide when batch is allowed (only for raw CSVs)
looks_early = is_model_output_like(df_raw)
looks_agg   = is_aggregated_like(df_raw)
can_batch   = not (looks_early or looks_agg)

t1, t2, t3 = st.tabs(["🔮 Quick single", "📦 Batch from CSV", "🔧 Connection"])

# ================== Tab 1: Quick single ==================
with t1:
    st.caption("⚡ Get a fresh prediction, independent of the file you opened.")
    col1, col2 = st.columns(2)
    country_opts = sorted(COUNTRY_MAP.keys())

    with col1:
        tender_country = st.selectbox(
            "Country (tender_country)",
            [f"{c} — {COUNTRY_MAP[c]}" for c in country_opts],
            index=country_opts.index("IT") if "IT" in country_opts else 0,
        ).split(" — ")[0]

        PROC_REV = {v: k for k, v in PROC_MAP.items()}
        proc_labels = list(PROC_REV.keys())
        default_label = PROC_MAP.get("OPEN", proc_labels[0])
        proc_label_sel = st.selectbox(
            "Procedure type",
            proc_labels,
            index=proc_labels.index(default_label),
        )
        tender_procedureType = PROC_REV[proc_label_sel]

        tender_supplyType = st.selectbox("Supply type", ["WORKS", "SUPPLIES", "SERVICES"])

    with col2:
        tender_year = st.number_input("Year", value=2023, step=1)
        if not (MIN_SINGLE_YEAR <= tender_year <= MAX_SINGLE_YEAR):
            st.warning(f"Year should be between {MIN_SINGLE_YEAR} and {MAX_SINGLE_YEAR} for this demo.")

        tender_estimatedPrice_EUR = st.number_input(
            "Estimated price (EUR)",
            min_value=0.0,
            max_value=50_000_000.0,
            value=3_000_000.0,
        )
        if tender_estimatedPrice_EUR < MIN_SINGLE_EST_PRICE:
            st.warning(f"Estimated price must be at least {MIN_SINGLE_EST_PRICE:,.0f} EUR.")

        lot_bidsCount = st.number_input("Bids count", value=4, step=1)
        if not (MIN_SINGLE_BIDS <= lot_bidsCount <= MAX_SINGLE_BIDS):
            st.warning(f"Bids count should be between {MIN_SINGLE_BIDS} and {MAX_SINGLE_BIDS} for this demo.")

        tau_val = st.number_input("τ (threshold, days)", 100, 1200, 720)

    st.divider()
    st.markdown("#### Tender CPV category")

    def cpv_label(code: str) -> str:
        c = str(code).strip()
        div = c[:2].zfill(2)
        name = CPV_MAPPING.get(div, "CPV")
        return f"{c} — {name}"

    cpv_codes = [
        "45200000", "45100000", "30200000", "33100000", "09100000",
        "71300000", "80500000", "72000000", "90400000", "34900000",
        "79900000", "50500000",
    ]
    cpv_mode = st.selectbox("Main CPV", ["Custom…"] + [cpv_label(c) for c in cpv_codes], index=1)
    tender_mainCpv = (
        st.text_input("Enter CPV code (8 digits)", "45200000")
        if cpv_mode == "Custom…" else cpv_mode.split(" — ")[0]
    )

    def infer_supply_from_cpv(cpv: str) -> str:
        c = "".join(ch for ch in str(cpv) if ch.isdigit())[:8].ljust(2, "0")
        div = int(c[:2]) if c[:2].isdigit() else -1
        if div == 45:
            return "WORKS"
        if 3 <= div <= 44:
            return "SUPPLIES"
        if div >= 50:
            return "SERVICES"
        return "UNKNOWN"

    inferred_supply = infer_supply_from_cpv(tender_mainCpv)
    need_fix = inferred_supply != "UNKNOWN" and inferred_supply != tender_supplyType
    fix_on = st.toggle(f"Auto-fix supply type to {inferred_supply}", value=True) if need_fix else False
    if need_fix and not fix_on:
        st.error("Supply type and CPV disagree. Enable Auto-fix or change one of them to continue.")
        st.stop()
    if fix_on:
        tender_supplyType = inferred_supply

    # OPTIONAL: keep ONLY if you still want the demo override.
    # If you don't want it at all: delete this toggle + delete the override block below.
    force_long = st.toggle("🔧 Demo: Force long model (override router)", value=False)

    left, right = st.columns([1, 3])
    with left:
        run_single = st.button("🔮 Predict", use_container_width=True)
    with right:
        st.caption("")

    if run_single:
        errors: list[str] = []
        if tender_estimatedPrice_EUR < MIN_SINGLE_EST_PRICE:
            errors.append(f"Estimated price must be at least {MIN_SINGLE_EST_PRICE:,.0f} EUR.")
        if not (MIN_SINGLE_YEAR <= tender_year <= MAX_SINGLE_YEAR):
            errors.append(f"Year must be between {MIN_SINGLE_YEAR} and {MAX_SINGLE_YEAR} (got {int(tender_year)}).")
        if not (MIN_SINGLE_BIDS <= lot_bidsCount <= MAX_SINGLE_BIDS):
            errors.append(f"Bids count must be between {MIN_SINGLE_BIDS} and {MAX_SINGLE_BIDS} (got {int(lot_bidsCount)}).")

        if errors:
            st.error("Please fix the following before requesting a prediction:\n\n- " + "\n- ".join(errors))
            st.stop()

        price = float(tender_estimatedPrice_EUR) if tender_estimatedPrice_EUR is not None else None
        bids  = float(lot_bidsCount) if lot_bidsCount is not None else None

        payload = {
            "tender_country": tender_country,
            "tender_procedureType": tender_procedureType,
            "tender_supplyType": tender_supplyType,
            "tender_mainCpv": str(tender_mainCpv).strip(),
            "tender_year": int(tender_year),
            "tender_estimatedPrice_EUR": price,
            "lot_bidsCount": bids,
            # important engineered features
            "tender_estimatedPrice_EUR_log": float(np.log1p(price)) if price is not None else None,
            "lot_bidsCount_log": float(np.log1p(bids)) if bids is not None else None,
        }

        try:
            res = api_predict(payload, tau=float(tau_val))

            # ====== Parse fields safely ======
            pred = float(res.get("predicted_days", float("nan")))
            tau_days = float(res.get("tau_days", tau_val))  # prefer what API says
            stage = str(res.get("stage_used", "—"))

            ps = res.get("pred_short", None)
            pl = res.get("pred_long", None)
            ps = float(ps) if ps is not None else None
            pl = float(pl) if pl is not None else None

            p_long = float(res.get("p_long", float("nan")))
            tau_prob = float(res.get("tau_prob", float("nan")))

            flag = bool(res.get("risk_flag", pred >= tau_days))

            # OPTIONAL: keep this ONLY if you still want the demo override
            # If you don't want it at all, delete the toggle earlier + delete this whole block.
            if force_long and (pl is not None):
                pred = pl
                stage = "long_reg (forced)"
                flag = bool(pred >= tau_days)

            # ====== Top summary (4 metrics) ======
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Final (used) days", "—" if not np.isfinite(pred) else f"{pred:,.0f}")
                st.caption(f"Threshold τ = {tau_days:,.0f} days")

            with c2:
                st.metric("Short estimate", "—" if ps is None else f"{ps:,.0f}")
                st.caption("Assuming the short-duration regime")

            with c3:
                st.metric("Long estimate", "—" if pl is None else f"{pl:,.0f}")
                st.caption("Assuming the long-duration regime")

            with c4:
                st.metric("Long-duration probability", "—" if not np.isfinite(p_long) else f"{p_long*100:.1f}%")
                st.caption("Confidence this case belongs to the long-duration regime")

            st.divider()

            # ===== Tabs below results (2) =====
            tab_explain, tab_signals, tab_debug = st.tabs(["ℹ️ Explanation", "📊 Model signals", "🧪 Debug"])

            with tab_explain:
                st.markdown(
                    "**What these estimates mean**\n\n"
                    "- **Final (used):** the prediction the model actually selected via its routing logic.\n"
                    "- **Short estimate:** prediction assuming a short-duration regime.\n"
                    "- **Long estimate:** prediction assuming a long-duration regime.\n"
                )

                st.markdown("**How the model chooses Short vs Long**")
                if stage == "long_reg (forced)":
                    st.warning("Long model used because manual override is enabled.")
                else:
                    st.info(
                        "The model chooses between two specialized predictors. "
                        "It uses the **Long** model only when it is sufficiently confident the case belongs to the "
                        "**long-duration regime**; otherwise it uses the **Short** model."
                    )

            with tab_signals:
                st.markdown("### Routing signal (confidence vs cutoff)")
                st.caption("This is the signal the router uses to decide whether to use the Long model.")

                # Show probability clearly
                st.metric(
                    "Confidence of long-duration regime",
                    "—" if not np.isfinite(p_long) else f"{p_long*100:.1f}%",
                )

                # Show cutoff clearly (avoid raw variable names)
                if np.isfinite(tau_prob):
                    st.metric("Routing cutoff", f"{tau_prob*100:.1f}%")
                else:
                    st.metric("Routing cutoff", "—")

                # Model choice line (ONLY here, as requested)
                if stage == "long_reg (forced)":
                    st.warning("Model choice: Long model (manual override).")
                else:
                    used = "Long model" if stage.startswith("long_reg") else "Short model"
                    st.info(
                        f"Model choice: **{used}**. "
                        f"The Long model is selected only when confidence is above the routing cutoff."
                    )
                    if np.isfinite(p_long) and np.isfinite(tau_prob):
                        if p_long >= tau_prob:
                            st.success("Routing outcome: confidence ≥ cutoff → Long model selected.")
                        else:
                            st.info("Routing outcome: confidence < cutoff → Short model selected.")
                        
            with tab_debug:
                # Keep debug clean: do NOT show force_long unless you want it
                st.json(
                    {
                        "predicted_days_final_used": pred,
                        "tau_days": tau_days,
                        "stage_used": stage,
                        "pred_short": ps,
                        "pred_long": pl,
                        "risk_flag": flag,
                        "p_long": p_long,
                        "tau_prob": tau_prob,
                        "build": res.get("build", None),
                    }
                )
                st.caption("Raw API response")
                st.json(res)

        except requests.HTTPError as e:
            st.error(f"API error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ================== Tab 2: Batch from CSV ==================
with t2:
    if not can_batch:
        banner(
            "This tab works only with raw tender records/CSVs (no model outputs, e.g. predicted_days, "
            "risk_flag, or aggregated summaries, e.g. Risk% and Count per group).<br><br>"
            "Open a procurement file with the original inputs, e.g. one row per tender, "
            "containing at least:<br>"
            "<b>tender_country</b>, <b>tender_mainCpv</b>, <b>tender_year</b> "
            "(optionally also procedure type, supply type, estimated price, bids count).",
            "warn",
        )
    else:
        st.caption(
            "Generate **multiple predictions at once** using the CSV file you uploaded. "
            "Each row in your file will receive a model prediction — ideal for testing many tenders together."
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            tau_batch = st.number_input("Override τ (batch, days)", 100, 1200, 720, step=10)
        with col_b:
            top_k = st.number_input("Top-K (charts)", min_value=5, max_value=50, value=12, step=1)

        def _prepare_rows(df_src: pd.DataFrame) -> list[dict]:
            """
            Prepare rows for /predict_batch
            - works with new & old features.json
            - NEVER sends target_duration
            """
            feat_path = "procuresight_api/model/features.json"

            # --- read feature list safely ---
            feat_cols = None
            if os.path.exists(feat_path):
                try:
                    with open(feat_path, "r", encoding="utf-8") as f:
                        FEATS = json.load(f)

                    feat_cols = (
                        FEATS.get("features")
                        or FEATS.get("feature_columns")
                        or FEATS.get("featureColumns")
                    )
                except Exception:
                    feat_cols = None

            if not feat_cols:
                feat_cols = list(df_src.columns)

            # 🔒 hard stop: never allow target_duration
            feat_cols = [c for c in feat_cols if c != "target_duration"]

            rows: list[dict] = []
            for _, r in df_src.iterrows():
                d = {c: r[c] for c in feat_cols if c in df_src.columns}

                # double safety
                d.pop("target_duration", None)

                # --- normalize country codes ---
                if "tender_country" in d and pd.notna(d["tender_country"]):
                    d["tender_country"] = str(d["tender_country"]).upper().strip()

                if "buyer_country" in d and pd.notna(d["buyer_country"]):
                    d["buyer_country"] = str(d["buyer_country"]).upper().strip()

                # --- normalize CPV ---
                if "tender_mainCpv" in d and pd.notna(d["tender_mainCpv"]):
                    cpv = str(d["tender_mainCpv"]).strip()
                    if cpv.endswith(".0"):
                        cpv = cpv[:-2]
                    cpv = "".join(ch for ch in cpv if ch.isdigit())[:8]
                    d["tender_mainCpv"] = cpv

                # --- add logs ONLY if model expects them ---
                if "tender_estimatedPrice_EUR_log" in feat_cols:
                    base = pd.to_numeric(d.get("tender_estimatedPrice_EUR"), errors="coerce")
                    d["tender_estimatedPrice_EUR_log"] = None if pd.isna(base) else float(np.log1p(base))

                if "lot_bidsCount_log" in feat_cols:
                    base = pd.to_numeric(d.get("lot_bidsCount"), errors="coerce")
                    d["lot_bidsCount_log"] = None if pd.isna(base) else float(np.log1p(base))

                rows.append(d)

            return rows

        # ---- display + call API ----
        with st.expander("Preview & send", expanded=True):
            try:
                df_in = df_raw.copy()

                preview_rename = {
                    "tender_procedureType": "Procedure (raw)",
                    "procedure_label": "Procedure",
                    "tender_country": "Country code",
                    "country_name": "Country",
                    "buyer_country": "Buyer country",
                    "buyer_buyerType": "Buyer type",
                    "tender_year": "Year",
                    "tender_mainCpv": "Main CPV",
                    "tender_supplyType": "Supply type",
                    "tender_estimatedPrice_EUR": "Estimated price (EUR)",
                    "lot_bidsCount": "Bids count",
                    "tender_indicator_score_INTEGRITY": "Integrity score",
                    "tender_indicator_score_ADMINISTRATIVE": "Administrative score",
                    "tender_indicator_score_TRANSPARENCY": "Transparency score",
                }
                prev = df_in.head(10).rename(columns=preview_rename).copy()

                for cand in ["Risk%", "RiskPct", "risk_pct", "risk%"]:
                    if cand in prev.columns:
                        s = pd.to_numeric(prev[cand], errors="coerce")
                        if pd.notna(s.max()) and s.max() <= 1.5:
                            s = s * 100.0
                        prev["Risk %"] = s.map(lambda v: f"{v:.2f}%")
                        prev.drop(
                            columns=[
                                c
                                for c in ["Risk%", "RiskPct", "risk_pct", "risk%"]
                                if c in prev.columns
                            ],
                            inplace=True,
                            errors="ignore",
                        )
                        break

                st.caption(f"Rows to predict: {len(df_in):,}")
                st.dataframe(prev, use_container_width=True)

                rows = _prepare_rows(df_in)
                rows = rows_json_safe_from_list(rows)  # NaN/±inf -> None
                preds = api_predict_batch(rows, tau=float(tau_batch) if tau_batch else None)
                df_out = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(preds)], axis=1)

                if "predicted_days" in df_out.columns:
                    st.divider()
                    st.subheader("Quick rankings on batch predictions")
                    df_rank_src = derive_labels(df_out.copy())
                    tau_used = float(tau_batch) if tau_batch else 720.0
                    df_rank_src["risk_flag"] = pd.to_numeric(
                        df_rank_src["predicted_days"], errors="coerce"
                    ) >= tau_used

                    def _all_unknown(df: pd.DataFrame, cols) -> bool:
                        if cols is None:
                            return True
                        if isinstance(cols, (str, int)):
                            cols = [cols]
                        cols = [c for c in cols if c in df.columns]
                        if not cols:
                            return True
                        sub = df[cols]
                        if isinstance(sub, pd.Series):
                            sub = sub.to_frame()
                        norm = sub.astype(str).apply(lambda s: s.str.strip().str.upper())
                        return norm.apply(
                            lambda s: (s.nunique(dropna=False) == 1)
                            and (s.iloc[0] in ("UNKNOWN", "NAN", "NONE", "", "NA")),
                            axis=0,
                        ).all()

                    def _rank_and_plot(tbl: pd.DataFrame, by_candidates: list[str], title: str):
                        by_cols = [c for c in by_candidates if c in tbl.columns]
                        st.markdown(f"#### {title}")
                        if not by_cols:
                            st.info(f"Required columns missing for {title}.")
                            return
                        out = (
                            tbl.groupby(by_cols, dropna=False)
                            .agg(
                                RiskPct=("risk_flag", lambda s: float(100.0 * np.nanmean(s.astype(float)))),
                                Count=("risk_flag", "size"),
                            )
                            .sort_values(["RiskPct", "Count"], ascending=[False, False])
                            .reset_index()
                        )
                        if out.empty or _all_unknown(out, [c for c in out.columns if c not in ("RiskPct", "Count")]):
                            st.info(
                                "No rankings available: the selected CSV lacks raw columns (country/procedure/CPV) or they are all Unknown."
                            )
                            return

                        out_disp = out.copy()
                        out_disp["RiskPct"] = out_disp["RiskPct"].astype(float).map(lambda v: f"{v:.2f}%")
                        rename_map = {
                            c: FRIENDLY_COLS.get(c, FRIENDLY_COLS.get(str(c).lower(), c))
                            for c in out_disp.columns
                            if c not in ("RiskPct", "Count")
                        }
                        out_disp = out_disp.rename(columns=rename_map).rename(columns={"RiskPct": "Risk %"})
                        st.dataframe(out_disp.head(50), use_container_width=True)

                        idx_cols = [c for c in out.columns if c not in ("RiskPct", "Count")]
                        if not idx_cols:
                            x = pd.Series([f"Group {i+1}" for i in range(len(out))], index=out.index)
                        else:
                            sub = out[idx_cols]
                            if isinstance(sub, pd.Series):
                                sub = sub.to_frame()
                            x = sub.astype(str).agg(" — ".join, axis=1)

                        fig = go.Figure()
                        fig.add_trace(
                            go.Bar(
                                x=x,
                                y=out["RiskPct"],
                                name="Risk%",
                                yaxis="y1",
                                hovertemplate="<b>%{x}</b><br>Risk%: %{y:.2f}%<extra></extra>",
                            )
                        )
                        fig.add_trace(
                            go.Bar(
                                x=x,
                                y=(out["Count"] / 1000.0),
                                name="Count (K)",
                                yaxis="y2",
                                opacity=0.60,
                                hovertemplate="<b>%{x}</b><br>Count: %{y:,.1f}K<extra></extra>",
                            )
                        )
                        fig.update_layout(
                            xaxis_title="Group",
                            yaxis=dict(title="Risk %", range=[0, 100]),
                            yaxis2=dict(title="Count (K)", overlaying="y", side="right"),
                            barmode="group",
                            margin=dict(t=50),
                            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"batch_rank_{_slug(title)}")

                    _rank_and_plot(df_rank_src, ["country_name", "tender_country"], "By Country")
                    _rank_and_plot(df_rank_src, ["procedure_label", "tender_procedureType"], "By Procedure")

                    df_rank_src2 = df_rank_src.copy()
                    base_series = pd.Series([np.nan] * len(df_rank_src2), index=df_rank_src2.index)
                    df_rank_src2["cpv_div2"] = pd.to_numeric(
                        df_rank_src2.get("cpv_div2", base_series), errors="coerce"
                    )
                    if df_rank_src2["cpv_div2"].isna().all():
                        s_cpv = (
                            df_rank_src2["tender_mainCpv"].astype(str)
                            if "tender_mainCpv" in df_rank_src2.columns
                            else pd.Series([""] * len(df_rank_src2), index=df_rank_src2.index)
                        )
                        df_rank_src2["cpv_div2"] = pd.to_numeric(s_cpv.str.slice(0, 2), errors="coerce")
                    _rank_and_plot(df_rank_src2, ["cpv_div2"], "By CPV (division-2)")
                else:
                    st.info(
                        "No rankings available: predictions were not produced for this file (missing 'predicted_days')."
                    )

                st.markdown("### Predictions (joined with your batch file)")
                st.caption(
                    "Your original data is shown together with the new predictions — ready to review or download."
                )
                df_show = df_out.copy()
                if "predicted_days" in df_show.columns:
                    df_show["Predicted days"] = (
                        pd.to_numeric(df_show["predicted_days"], errors="coerce").round().astype("Int64")
                    )
                if "risk_flag" in df_show.columns:
                    df_show["High-risk?"] = df_show["risk_flag"].map({True: "Yes", False: "No"})
                if "RiskPct" in df_show.columns:
                    rp = pd.to_numeric(df_show["RiskPct"], errors="coerce")
                    if pd.notna(rp.max()) and rp.max() <= 1.5:
                        rp = rp * 100.0
                    df_show["Risk %"] = rp.map(lambda v: f"{v:.2f}%")

                show_rename = {
                    "tender_procedureType": "Procedure (raw)",
                    "tender_supplyType": "Supply type",
                    "buyer_buyerType": "Buyer type",
                    "buyer_country": "Buyer country",
                    "tender_estimatedPrice_EUR": "Estimated price (EUR)",
                    "lot_bidsCount": "Bids count",
                    "tender_indicator_score_INTEGRITY": "Integrity score",
                    "tender_indicator_score_ADMINISTRATIVE": "Administrative score",
                    "tender_indicator_score_TRANSPARENCY": "Transparency score",
                }
                df_show.rename(columns=show_rename, inplace=True)
                df_show = friendly_rename_df(df_show)
                for c in ["RiskPct", "Risk%", "predicted_days", "risk_flag"]:
                    if c in df_show.columns:
                        df_show.drop(columns=[c], inplace=True, errors="ignore")
                df_show = df_show.loc[:, ~pd.Index(df_show.columns).duplicated()].copy().reset_index(drop=True)

                front = [
                    c
                    for c in [
                        "Procedure",
                        "Procedure (raw)",
                        "Country",
                        "Country code",
                        "CPV Group",
                        "CPV Division",
                        "Risk %",
                        "Predicted days",
                        "High-risk?",
                        "model_used",
                        "tau",
                    ]
                    if c in df_show.columns
                ]
                df_show = df_show[front + [c for c in df_show.columns if c not in front]]

                def _hl(val):
                    if val == "Yes":
                        return "background-color:#ffb3b3; font-weight:700; text-align:center;"
                    if val == "No":
                        return "background-color:#b3ffcc; font-weight:700; text-align:center;"
                    return ""

                try:
                    styler = df_show.style
                    if "High-risk?" in df_show.columns:
                        styler = styler.applymap(_hl, subset=["High-risk?"])
                    if "Predicted days" in df_show.columns:
                        styler = styler.set_properties(
                            subset=["Predicted days"],
                            **{"background-color": "#fff8b3", "font-weight": "bold"},
                        )
                    st.dataframe(styler, use_container_width=True)
                except Exception:
                    st.dataframe(df_show, use_container_width=True)

                cdl, cdr = st.columns(2)
                with cdl:
                    st.download_button(
                        "⬇️ Download predictions CSV",
                        df_show.to_csv(index=False).encode("utf-8"),
                        file_name="predictions_batch_friendly.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with cdr:
                    st.download_button(
                        "⬇️ Download raw predictions CSV",
                        df_out.to_csv(index=False).encode("utf-8"),
                        file_name="predictions_batch_raw.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except requests.HTTPError as e:
                st.error(
                    f"API error: {e.status_code if hasattr(e, 'status_code') else ''} — "
                    f"{e.response.text if hasattr(e, 'response') else e}"
                )
            except Exception as e:
                st.error(f"Failed to run batch: {e}")


# ================== Tab 3: Connection / Debug ==================
with t3:
    st.caption("Connection & diagnostics.")
    cols = st.columns(3)
    with cols[0]:
        try:
            info = requests.get(f"{api_base_in}/", timeout=6).json()
            ok = bool(info.get("model_loaded", False))
            kpi_card("API reachability", "OK" if ok else "No", api_base_in)
        except Exception as e:
            kpi_card("API reachability", "No", str(e))
    with cols[1]:
        kpi_card("τ (default)", f"{int(st.session_state.get('tau', 720))}", "UI setting")
    with cols[2]:
        kpi_card("Theme", BRAND["name"], "ProcureSight")
    with st.expander("GET / payload"):
        try:
            st.json(requests.get(f"{api_base_in}/", timeout=6).json())
        except Exception as e:
            st.error(e)

st.markdown(
    """
    <hr style="margin:2rem 0; border-top:1px solid #e6e8ef"/>
    <div style="display:flex;justify-content:space-between;align-items:center;" class="small-muted">
      <div>© ProcurementProject — Streamlit demo</div>
      <div>Contact: <a href="mailto:alarsenoudh@gmail.com">alarsenoudh@gmail.com</a></div>
    </div>
    """,
    unsafe_allow_html=True,
)
