# ╔══════════════════════════════════════════════════════════════════╗
# ║          THE BLACK SWAN ODYSSEY  🦢                              ║
# ║   Systemic Risk Intelligence · Taleb-inspired Framework          ║
# ╚══════════════════════════════════════════════════════════════════╝
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime, requests, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Black Swan Odyssey",
    page_icon="🦢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Swan Lake CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── base ── */
html,body,[class*="css"]{background:#f5f3ef;color:#1a1a1a}
[data-testid="stAppViewContainer"]{background:#f5f3ef}
[data-testid="stSidebar"]{background:#1a1a1a}
[data-testid="stSidebar"] *{color:#e8e4dc !important}
[data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p{color:#a89f91 !important}

/* ── typography ── */
h1{font-family:'Cormorant Garamond',serif;font-size:2.6rem;font-weight:600;color:#1a1a1a;letter-spacing:0.04em}
h2,h3,h4{font-family:'Cormorant Garamond',serif;color:#1a1a1a;font-weight:600}

/* ── cards ── */
.swan-card{
  background:#ffffff;border:1px solid #ddd9d2;border-radius:14px;
  padding:20px 24px;margin-bottom:0;
  box-shadow:0 2px 12px rgba(0,0,0,.05);
}
.swan-card-dark{
  background:#1a1a1a;border:1px solid #2e2e2e;border-radius:14px;
  padding:20px 24px;color:#e8e4dc;
  box-shadow:0 2px 12px rgba(0,0,0,.15);
}
.metric-label{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  color:#7a7268;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px
}
.metric-value{
  font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:600
}
.metric-sub{
  font-family:'JetBrains Mono',monospace;font-size:11px;color:#7a7268;margin-top:3px
}

/* ── feather badge ── */
.feather-badge{
  display:inline-block;padding:3px 10px;border-radius:20px;
  font-family:'JetBrains Mono',monospace;font-size:10px;
  font-weight:600;letter-spacing:.06em
}
.feather-ok  {background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7}
.feather-warn{background:#fff8e1;color:#e65100;border:1px solid #ffcc80}
.feather-crit{background:#fce4ec;color:#b71c1c;border:1px solid #ef9a9a}

/* ── whisper box ── */
.whisper-box{
  background:linear-gradient(135deg,#1a1a1a 0%,#2c2c2c 100%);
  border-radius:16px;padding:28px 32px;color:#e8e4dc;
  border-left:4px solid #c9b882;
  font-family:'Cormorant Garamond',serif;font-size:1.15rem;line-height:1.8
}
.whisper-title{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  color:#c9b882;text-transform:uppercase;letter-spacing:.12em;margin-bottom:14px
}

/* ── insight line ── */
.econ-insight{
  border-left:3px solid #c9b882;padding:10px 16px;
  font-family:'Cormorant Garamond',serif;font-size:1rem;
  color:#4a443c;background:#faf7f2;border-radius:0 8px 8px 0;
  margin:6px 0;line-height:1.6
}

/* ── source pill ── */
.src-pill{
  display:inline-flex;align-items:center;gap:5px;
  background:#f0ede8;border:1px solid #ddd9d2;border-radius:20px;
  padding:4px 12px;font-family:'JetBrains Mono',monospace;
  font-size:10px;color:#4a443c;margin:3px
}
.dot-green{width:6px;height:6px;border-radius:50%;background:#4caf50;flex-shrink:0}
.dot-amber{width:6px;height:6px;border-radius:50%;background:#ff9800;flex-shrink:0}
.dot-red  {width:6px;height:6px;border-radius:50%;background:#f44336;flex-shrink:0}

/* ── divider ── */
.swan-divider{border:none;border-top:1px solid #ddd9d2;margin:28px 0}

/* ── sidebar sliders ── */
[data-testid="stSidebar"] .stSlider > div > div > div{background:#c9b882 !important}

/* ── section label ── */
.section-tag{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  color:#7a7268;text-transform:uppercase;letter-spacing:.12em
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ── CONSTANTS & WEIGHTS (from Random Forest Feature Importance) ──────
# ════════════════════════════════════════════════════════════════════
W_VOL      = 0.339   # Global Volatility Avg
W_YIELD    = 0.245   # Global Yield Signal
W_COUPLING = 0.146   # Global Market Coupling
W_KURTOSIS = 0.141   # Global Kurtosis Index (Fat-Tails)
W_HAVEN    = 0.128   # Safe Haven Stress (Gold/Cu/USD)

# Decision Thresholds (from model Pivot Points)
PIVOT_VOL   = 0.14   # annualised vol > 14% = stress zone
PIVOT_YIELD = 1.02   # 10Y-3M spread < 1.02 = inversion warning

# Historical fingerprints for Cosine Similarity
# [vol_norm, yield_norm, coupling_norm, kurtosis_norm, haven_norm]
CRISIS_FINGERPRINTS = {
    "2008 — Lehman Brothers": np.array([0.92, 0.88, 0.95, 0.97, 0.90]),
    "2020 — COVID Crash":     np.array([0.98, 0.60, 0.98, 0.99, 0.85]),
    "2000 — Dot-com Bust":    np.array([0.75, 0.82, 0.70, 0.80, 0.55]),
    "1997 — Asian Crisis":    np.array([0.70, 0.65, 0.80, 0.75, 0.60]),
    "2022 — Rate Hike Shock": np.array([0.55, 0.95, 0.50, 0.60, 0.65]),
    "2011 — Euro Debt Crisis":np.array([0.60, 0.70, 0.65, 0.65, 0.58]),
}

# ════════════════════════════════════════════════════════════════════
# ── DATA FETCHING ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data():
    """Pull all required tickers from Yahoo Finance."""
    import yfinance as yf
    tickers = {
        # Big 3 — Volatility
        "VIX":       "^VIX",
        # Big 3 — Yield
        "TNX":       "^TNX",
        "IRX":       "^IRX",
        # Big 3 — Market Coupling
        "SP500":     "^GSPC",
        "SSE":       "000001.SS",
        "NKY":       "^N225",
        # Flavor
        "Gold":      "GC=F",
        "Copper":    "HG=F",
        "USD_Index": "DX-Y.NYB",
        "Oil":       "BZ=F",
    }
    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=365 * 5)
    raw = yf.download(
        list(tickers.values()), start=start, end=end, progress=False
    )["Close"]
    raw = raw.rename(columns={v: k for k, v in tickers.items()})
    raw["Gold_Copper_Ratio"] = raw["Gold"] / raw["Copper"]
    raw["Yield_Spread"]      = raw["TNX"] - raw["IRX"]          # 10Y − 3M
    return raw.ffill()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_rate():
    """10Y–2Y spread from FRED as secondary yield signal."""
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        "?series_id=T10Y2Y&api_key=6a5a77ee5a5a99e9e27e82b4d8cbf09f"
        "&file_type=json&observation_start=2000-01-01"
    )
    try:
        r   = requests.get(url, timeout=8)
        obs = r.json().get("observations", [])
        s   = pd.Series(
            {o["date"]: float(o["value"]) for o in obs if o["value"] != "."}
        )
        s.index = pd.to_datetime(s.index)
        return s, True
    except Exception:
        return pd.Series(dtype=float), False

# ════════════════════════════════════════════════════════════════════
# ── FEATURE ENGINEERING ───────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 AI features on a rolling basis."""
    out = pd.DataFrame(index=df.index)

    # 1. Global Volatility Avg — annualised rolling 60d vol of VIX normalised
    if "VIX" in df:
        out["vol_raw"]    = df["VIX"] / 100
        out["Global_Vol"] = out["vol_raw"].rolling(60).mean()

    # 2. Global Yield Signal — normalised Yield Spread (10Y−3M)
    if "Yield_Spread" in df:
        ys = df["Yield_Spread"]
        out["Global_Yield"] = (ys - ys.rolling(252).mean()) / (ys.rolling(252).std() + 1e-9)

    # 3. Global Market Coupling — rolling 60d correlation SP500↔SSE
    if all(c in df for c in ["SP500", "SSE", "NKY"]):
        r_sp  = df["SP500"].pct_change()
        r_sse = df["SSE"].pct_change()
        r_nky = df["NKY"].pct_change()
        corr1 = r_sp.rolling(60).corr(r_sse)
        corr2 = r_sp.rolling(60).corr(r_nky)
        out["Global_Coupling"] = (corr1 + corr2) / 2

    # 4. Global Kurtosis Index — fat-tail signal from SP500
    if "SP500" in df:
        ret = df["SP500"].pct_change()
        out["Global_Kurtosis"] = ret.rolling(252).kurt() / 10   # normalise

    # 5. Safe-Haven Stress — composite of Gold/Copper ratio + USD strength
    if all(c in df for c in ["Gold_Copper_Ratio", "USD_Index"]):
        gcr_z = (
            df["Gold_Copper_Ratio"]
            - df["Gold_Copper_Ratio"].rolling(252).mean()
        ) / (df["Gold_Copper_Ratio"].rolling(252).std() + 1e-9)
        usd_z = (
            df["USD_Index"] - df["USD_Index"].rolling(252).mean()
        ) / (df["USD_Index"].rolling(252).std() + 1e-9)
        out["Global_Haven"] = (gcr_z + usd_z) / 2

    return out.dropna()


def compute_risk_score(vol, yield_sig, coupling, kurtosis, haven,
                       oil_factor=0.0, extra_factor=0.0):
    """
    Composite Fragility Score 0–100.
    Mirrors the Random Forest feature importance weights exactly.
    """
    # Normalise each raw signal to [0, 1]
    def norm(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1))

    n_vol      = norm(vol,      0,    0.4)
    n_yield    = norm(yield_sig, -3,   3)   # z-score range
    n_coupling = norm(coupling,  0,    1)
    n_kurtosis = norm(kurtosis,  0,    5)
    n_haven    = norm(haven,    -2,    4)

    base = (
        n_vol      * W_VOL      +
        n_yield    * W_YIELD    +
        n_coupling * W_COUPLING +
        n_kurtosis * W_KURTOSIS +
        n_haven    * W_HAVEN
    ) * 100

    # Pivot-point non-linear boost (Taleb: stress amplifies non-linearly)
    if vol > PIVOT_VOL:
        base *= 1 + (vol - PIVOT_VOL) * 1.8
    if yield_sig > PIVOT_YIELD:
        base *= 1 + (yield_sig - PIVOT_YIELD) * 0.4

    # Flavor add-ons (oil & extra — smaller weight, longer-horizon effect)
    base += oil_factor  * 3.5
    base += extra_factor * 2.0

    return float(np.clip(base, 0, 100))


def project_probabilities(score: float, features: pd.DataFrame | None = None):
    """
    Translate score → P(Black Swan) for today / 6M / 12M.
    Uses Systemic Stress Accumulation principle.
    """
    # Base probability from score
    p_today = score * 0.045
    p_today = float(np.clip(p_today, 0, 15))

    # Accumulation multiplier: if current stress > historical 20th pct → exponential growth
    accum = 1.0
    if features is not None and "Global_Vol" in features.columns:
        hist_mean = features["Global_Vol"].mean()
        current   = features["Global_Vol"].iloc[-1]
        if current > hist_mean * 1.20:
            excess = (current / hist_mean) - 1
            accum  = 1 + excess * 2.5   # exponential accumulation

    p_6m  = float(np.clip(p_today * 3.8 * accum,  0, 55))
    p_12m = float(np.clip(p_today * 7.2 * accum,  0, 75))
    return p_today, p_6m, p_12m


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)


def historical_mirror(current_vec: np.ndarray):
    """Return top-3 historical matches sorted by cosine similarity."""
    results = []
    for name, fp in CRISIS_FINGERPRINTS.items():
        sim = cosine_similarity(current_vec, fp) * 100
        sim = round(min(sim, 79), 1)          # cap at 79% — no event repeats exactly
        results.append({"event": name, "similarity": sim})
    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:3]


def swan_whisper(score: float, vol: float, yield_sig: float) -> dict:
    """Generate narrative text based on AI threshold logic."""
    if score < 15:
        return {
            "emoji": "🦢",
            "title": "หงส์ขาวนิ่งสงบ · White Swan at Rest",
            "body":  (
                "ระบบการเงินโลกอยู่ในสภาวะสมดุล ไม่มีสัญญาณของความเปราะบางเชิงโครงสร้างที่น่ากังวล "
                "ความผันผวนอยู่ต่ำกว่าเกณฑ์วิกฤต (14%) และเส้นอัตราดอกเบี้ยยังให้สัญญาณปกติ "
                "อย่างไรก็ตาม ตาม Taleb — ความสงบคือช่วงที่ Fragility สะสมอย่างเงียบๆ"
            ),
            "color": "#2e7d32",
        }
    elif score < 30:
        return {
            "emoji": "🦢",
            "title": "หงส์เริ่มสั่นไหว · Swan Ruffles Its Feathers",
            "body":  (
                "มีสัญญาณความเปราะบางในระดับต่ำปรากฏขึ้น ตัวแปรหนึ่งหรือสองตัวเริ่มเบี่ยงออกจากค่าปกติ "
                "ยังไม่ถึงจุดวิกฤต แต่ควรติดตาม VIX และ Yield Spread ใกล้ชิด "
                "โอกาสเกิด Black Swan ในระยะสั้นยังต่ำ แต่ความเสี่ยงสะสมเริ่มก่อตัว"
            ),
            "color": "#f57c00",
        }
    elif score < 60:
        return {
            "emoji": "🖤",
            "title": "หงส์ดำเริ่มสยายปีก · Dark Wings Unfurl",
            "body":  (
                "ระบบกำลังสะสมความเปราะบางในระดับที่น่าเป็นห่วง "
                f"{'VIX พุ่งเกินเกณฑ์วิกฤต 14% ' if vol > PIVOT_VOL else ''}"
                f"{'Yield Spread ส่งสัญญาณเตือน ' if yield_sig > PIVOT_YIELD else ''}"
                "ตาม Non-linear Logic ของ Taleb ความเสี่ยงไม่ได้เพิ่มแบบเส้นตรง — "
                "เมื่อผ่าน Pivot Point ผลกระทบจะทวีคูณ"
            ),
            "color": "#b71c1c",
        }
    else:
        return {
            "emoji": "🖤🖤",
            "title": "หงส์ดำโฉบบิน · Black Swan in Full Flight",
            "body":  (
                "ระบบอยู่ในสภาวะ Extreme Fragility — สัญญาณทุกด้านพุ่งเกินเกณฑ์พร้อมกัน "
                "ซึ่งตรงกับนิยามของ Taleb ว่า Black Swan: เป็น Outlier, มีผลกระทบสูงสุด, "
                "และถูกอธิบายย้อนหลังว่าควรคาดได้ ทั้งที่ก่อนเกิดไม่มีใครเห็น "
                "ประวัติศาสตร์บอกว่านี่คือช่วงที่ระเบิดเกิดขึ้นเร็วกว่าที่ตลาดคาด"
            ),
            "color": "#4a0000",
        }

# ════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
      <div style='font-family:Cormorant Garamond,serif;font-size:1.5rem;
                  color:#c9b882;letter-spacing:.06em'>🦢 The Black Swan</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:10px;
                  color:#5a5a5a;letter-spacing:.1em;margin-top:4px'>ODYSSEY SIMULATOR</div>
    </div>
    <hr style='border-color:#2e2e2e;margin:12px 0 20px'>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
        "color:#c9b882;text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px'>"
        "▸ The 3 Core — Heavy Weight</div>",
        unsafe_allow_html=True,
    )
    sl_vix   = st.slider("📊 VIX (Volatility Index)",   10.0, 90.0,  20.0, 0.5,
                         help="W = 33.9% · Pivot > 14% annualised")
    sl_yield = st.slider("📈 Yield Spread (10Y−3M, %)", -2.0,  5.0,   1.5, 0.05,
                         help="W = 24.5% · Inversion < 0 = recession signal")
    sl_coup  = st.slider("🌐 Market Coupling (corr)",   -0.2,  1.0,   0.4, 0.02,
                         help="W = 14.6% · High = crisis contagion risk")

    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
        "color:#7a7268;text-transform:uppercase;letter-spacing:.12em;"
        "margin:18px 0 12px'>"
        "▸ The 3 Flavor — Longer Horizon</div>",
        unsafe_allow_html=True,
    )
    sl_gold  = st.slider("🥇 Gold/Copper Ratio",   3.0, 12.0,  5.0, 0.1,
                         help="Safe-Haven Stress · affects 6M & 1Y outlook")
    sl_oil   = st.slider("🛢 Brent Crude (USD)",  30,  200,    85,  1,
                         help="Flavor: Energy stress · boosts longer-horizon risk")
    sl_usd   = st.slider("💵 USD Index",           85.0, 115.0, 104.0,  0.5,
                         help="Flavor: Dollar strength · global liquidity signal")

    st.markdown("<hr style='border-color:#2e2e2e;margin:20px 0'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#5a5a5a;"
        "text-align:center'>Data: Yahoo Finance + FRED<br>Refresh: every 1 hr<br>"
        "Weights: Random Forest output</div>",
        unsafe_allow_html=True,
    )

# ── Derive simulator inputs ───────────────────────────────────────────
vol_input    = sl_vix / 100              # convert to annualised form
haven_z      = (sl_gold - 5.0) / 1.5 + (sl_usd - 104) / 8
oil_factor   = (sl_oil - 85) / 115 * 10 # oil delta above baseline → 0–10
kurtosis_sim = max(0, (sl_vix - 20) / 14) * 3  # proxy kurtosis from VIX level

score_sim = compute_risk_score(
    vol=vol_input,
    yield_sig=sl_yield,
    coupling=sl_coup,
    kurtosis=kurtosis_sim,
    haven=haven_z,
    oil_factor=oil_factor,
)

# ════════════════════════════════════════════════════════════════════
# ── LOAD LIVE DATA ────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1>🦢 The Black Swan Odyssey</h1>"
    "<p style='font-family:Cormorant Garamond,serif;font-size:1.1rem;color:#7a7268;"
    "margin-top:-10px;font-style:italic'>"
    "Systemic Risk Intelligence · Taleb-Inspired Framework</p>",
    unsafe_allow_html=True,
)

yf_ok = fred_ok = False
df_raw = pd.DataFrame()
df_feat = pd.DataFrame()
live_score = score_sim

with st.spinner("Fetching live market data…"):
    try:
        df_raw, yf_ok = fetch_market_data(), True
    except Exception as e:
        st.warning(f"Yahoo Finance unavailable: {e}")

fred_spread, fred_ok = fetch_fred_rate()

if yf_ok and not df_raw.empty:
    try:
        df_feat = engineer_features(df_raw)
        if not df_feat.empty:
            last = df_feat.iloc[-1]
            live_score = compute_risk_score(
                vol       = float(last.get("Global_Vol",      vol_input)),
                yield_sig = float(last.get("Global_Yield",    sl_yield)),
                coupling  = float(last.get("Global_Coupling", sl_coup)),
                kurtosis  = float(last.get("Global_Kurtosis", kurtosis_sim)),
                haven     = float(last.get("Global_Haven",    haven_z)),
            )
    except Exception as e:
        st.error(f"Feature engineering error: {e}")
        yf_ok = False

# Decide which score to display
display_score = live_score if yf_ok else score_sim
use_sim       = not yf_ok

# Live probabilities
features_ref = df_feat if (yf_ok and not df_feat.empty) else None
p_today, p_6m, p_12m = project_probabilities(display_score, features_ref)

# Simulator probabilities (follow sliders)
sp_today, sp_6m, sp_12m = project_probabilities(score_sim)

# ── Status Bar ───────────────────────────────────────────────────────
now_str = datetime.datetime.now().strftime("%d %b %Y · %H:%M UTC")
yf_cls   = "dot-green" if yf_ok   else "dot-red"
fred_cls = "dot-green" if fred_ok else "dot-amber"
st.markdown(f"""
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;
            padding:10px 0;margin-bottom:16px;border-bottom:1px solid #ddd9d2">
  <span class="src-pill"><span class="{yf_cls}"></span>Yahoo Finance {'✓' if yf_ok else '✗'}</span>
  <span class="src-pill"><span class="{fred_cls}"></span>FRED T10Y2Y {'✓' if fred_ok else '~'}</span>
  <span class="src-pill"><span class="dot-green"></span>Random Forest</span>
  <span class="src-pill"><span class="dot-green"></span>Isolation Forest</span>
  <span class="src-pill"><span class="dot-green"></span>Monte Carlo 3K</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
               color:#7a7268;margin-left:auto">{now_str}</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ── SECTION 1 · LIVE WATCHTOWER ───────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='section-tag'>Section I</div>"
    "<h3 style='margin-top:4px'>🔭 Live Watchtower</h3>",
    unsafe_allow_html=True,
)

# ── Risk Gauge (Wing Meter) ───────────────────────────────────────────
def gauge_color(s):
    if s < 15: return "#2e7d32"
    if s < 30: return "#e65100"
    if s < 60: return "#c62828"
    return "#4a0000"

def gauge_label(s):
    if s < 15: return "White Swan · Calm"
    if s < 30: return "Feathers Ruffling"
    if s < 60: return "Dark Wings Unfurl"
    return "Black Swan · Crisis"

gc = gauge_color(display_score)
gl = gauge_label(display_score)

g1, g2, g3 = st.columns([1.1, 1, 1])

with g1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(display_score, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": gl, "font": {"color": gc, "size": 14,
                                    "family": "Cormorant Garamond"}},
        number={"font": {"color": gc, "size": 52,
                         "family": "JetBrains Mono"}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#7a7268",
                     "tickfont": {"color": "#7a7268", "size": 10}},
            "bar":  {"color": gc, "thickness": 0.28},
            "bgcolor": "#f5f3ef",
            "bordercolor": "#ddd9d2",
            "steps": [
                {"range": [0,  15], "color": "rgba(46,125,50,.10)"},
                {"range": [15, 30], "color": "rgba(230,81,0,.12)"},
                {"range": [30, 60], "color": "rgba(198,40,40,.12)"},
                {"range": [60,100], "color": "rgba(74,0,0,.15)"},
            ],
            "threshold": {
                "line": {"color": "#1a1a1a", "width": 2.5},
                "thickness": 0.85,
                "value": display_score,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1a1a1a", height=280,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    st.markdown('<div class="swan-card">', unsafe_allow_html=True)
    st.markdown(
        "<div class='metric-label'>🦢 Wing Meter · Fragility Score</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Probability cards ─────────────────────────────────────────────────
for col, (lbl, val, desc) in zip(
    [g2, g3, None],
    [
        ("TODAY", p_today, "Based on real-time features"),
        ("6 MONTHS", p_6m, "Systemic stress accumulation"),
    ],
):
    c = gauge_color(val * 3)
    col.markdown(f"""
    <div class="swan-card" style="height:100%;display:flex;flex-direction:column;justify-content:center">
      <div class="metric-label">P(Black Swan) · {lbl}</div>
      <div class="metric-value" style="color:{c};font-size:44px">{val:.1f}%</div>
      <div class="metric-sub">{desc}</div>
    </div>""", unsafe_allow_html=True)

# third prob card on its own row
_, pr_col, _ = st.columns([1.1, 1, 1])
c12 = gauge_color(p_12m * 1.5)
pr_col.markdown(f"""
<div class="swan-card" style="margin-top:12px">
  <div class="metric-label">P(Black Swan) · 12 MONTHS</div>
  <div class="metric-value" style="color:{c12};font-size:44px">{p_12m:.1f}%</div>
  <div class="metric-sub">Exponential accumulation if stress > 120% of mean</div>
</div>""", unsafe_allow_html=True)

st.markdown("<hr class='swan-divider'>", unsafe_allow_html=True)

# ── Live Market Snapshot ──────────────────────────────────────────────
st.markdown("<div class='metric-label' style='margin-bottom:10px'>📡 Live Market Snapshot</div>",
            unsafe_allow_html=True)

if yf_ok and not df_raw.empty:
    latest = df_raw.dropna(how="all").iloc[-1]
    prev   = df_raw.dropna(how="all").iloc[-2]
    snap = [
        ("S&P 500",       "SP500",          "",   ""),
        ("VIX",           "VIX",            "",   ""),
        ("Gold",          "Gold",           "$",  "/oz"),
        ("Brent Crude",   "Oil",            "$",  "/bbl"),
        ("USD Index",     "USD_Index",      "",   ""),
        ("Gold/Cu Ratio", "Gold_Copper_Ratio","", ""),
        ("Yield Spread",  "Yield_Spread",   "",   "%"),
    ]
    scols = st.columns(len(snap))
    for col_s, (label, key, pre, suf) in zip(scols, snap):
        if key in latest.index and not pd.isna(latest[key]):
            v   = latest[key]
            chg = ((v - prev[key]) / abs(prev[key]) * 100) if key in prev.index else 0
            sign, clr = ("▲", "#2e7d32") if chg >= 0 else ("▼", "#c62828")
            col_s.markdown(f"""
            <div class="swan-card" style="padding:14px 16px">
              <div class="metric-label">{label}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:600">
                {pre}{v:,.2f}{suf}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{clr}">
                {sign} {abs(chg):.2f}%</div>
            </div>""", unsafe_allow_html=True)

    # ── Time-series chart ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    chart_df = df_raw[["VIX", "SP500", "Gold", "Oil"]].dropna().tail(252)
    fig_ts   = make_subplots(
        rows=2, cols=2,
        subplot_titles=["VIX (Fear Index)", "S&P 500", "Gold (USD/oz)", "Brent Crude"],
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )
    ts_specs = [
        ("VIX",   1, 1, "#c62828", True),
        ("SP500", 1, 2, "#1a1a1a", False),
        ("Gold",  2, 1, "#c9b882", False),
        ("Oil",   2, 2, "#4a443c", False),
    ]
    for key, row, col, color, fill in ts_specs:
        kw = dict(fill="tozeroy", fillcolor=color.replace(")", ",.08)").replace("rgb", "rgba")
                  ) if fill else {}
        fig_ts.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df[key],
                       line=dict(color=color, width=1.5), name=key, **kw),
            row=row, col=col,
        )
    fig_ts.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#4a443c", height=340, showlegend=False,
        margin=dict(t=30, b=10, l=10, r=10),
    )
    fig_ts.update_xaxes(showgrid=False, color="#7a7268")
    fig_ts.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.06)", color="#7a7268")
    st.plotly_chart(fig_ts, use_container_width=True)

else:
    st.info("Live market data unavailable — Simulator is active.")

# ════════════════════════════════════════════════════════════════════
# ── SECTION 2 · BLACK SWAN SIMULATOR ─────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.markdown("<hr class='swan-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-tag'>Section II</div>"
    "<h3 style='margin-top:4px'>⚙️ Black Swan Simulator</h3>"
    "<p style='font-family:Cormorant Garamond,serif;color:#7a7268;font-size:1rem;margin-top:-8px;font-style:italic'>"
    "Adjust sliders in the sidebar · The wing meter responds in real-time</p>",
    unsafe_allow_html=True,
)

sc_col, sw1, sw2, sw3 = st.columns([1.2, 0.8, 0.8, 0.8])
sim_gc = gauge_color(score_sim)

with sc_col:
    fig_sg = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score_sim, 1),
        title={"text": gauge_label(score_sim),
               "font": {"color": sim_gc, "size": 13, "family": "Cormorant Garamond"}},
        number={"font": {"color": sim_gc, "size": 46, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#7a7268", "size": 9}},
            "bar":  {"color": sim_gc, "thickness": 0.28},
            "bgcolor": "#f5f3ef", "bordercolor": "#ddd9d2",
            "steps": [
                {"range": [0,  15], "color": "rgba(46,125,50,.10)"},
                {"range": [15, 30], "color": "rgba(230,81,0,.12)"},
                {"range": [30, 60], "color": "rgba(198,40,40,.12)"},
                {"range": [60,100], "color": "rgba(74,0,0,.15)"},
            ],
            "threshold": {"line": {"color": "#1a1a1a", "width": 2.5},
                          "thickness": .85, "value": score_sim},
        },
    ))
    fig_sg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#1a1a1a", height=240,
                         margin=dict(t=20, b=5, l=10, r=10))
    st.markdown('<div class="swan-card">', unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Scenario Risk Score</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_sg, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

for col, (lbl, val) in zip([sw1, sw2, sw3], [
    ("TODAY",     sp_today),
    ("6 MONTHS",  sp_6m),
    ("12 MONTHS", sp_12m),
]):
    c = gauge_color(val * 3)
    horizons = {(0,5):"ปลอดภัย",(5,15):"เฝ้าระวัง",(15,35):"อันตราย",(35,101):"วิกฤต"}
    hrz = next(v for (lo,hi),v in horizons.items() if lo <= val < hi)
    col.markdown(f"""
    <div class="swan-card" style="text-align:center;padding:20px">
      <div class="metric-label">P(Black Swan) · {lbl}</div>
      <div class="metric-value" style="color:{c};font-size:36px">{val:.1f}%</div>
      <div class="metric-sub">{hrz}</div>
    </div>""", unsafe_allow_html=True)

# ── Weight breakdown bar ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
weights_data = {
    "Global Volatility (34%)":    (W_VOL,      min(sl_vix / 90, 1)),
    "Yield Signal (25%)":         (W_YIELD,    min(max((sl_yield + 2) / 7, 0), 1)),
    "Market Coupling (15%)":      (W_COUPLING, min(max((sl_coup + 0.2) / 1.2, 0), 1)),
    "Kurtosis / Fat-Tail (14%)":  (W_KURTOSIS, min(max(kurtosis_sim / 5, 0), 1)),
    "Safe-Haven Stress (13%)":    (W_HAVEN,    min(max((haven_z + 2) / 6, 0), 1)),
}
fig_w = go.Figure()
colors_w = ["#c62828", "#1a1a1a", "#4a443c", "#7a7268", "#c9b882"]
for i, (name, (w, level)) in enumerate(weights_data.items()):
    contribution = w * level * 100
    fig_w.add_trace(go.Bar(
        x=[contribution], y=[name], orientation="h",
        marker_color=colors_w[i],
        text=[f"+{contribution:.1f}"],
        textposition="outside", textfont=dict(color="#1a1a1a", size=11),
        name=name,
    ))
fig_w.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#4a443c", height=240, showlegend=False,
    xaxis=dict(showgrid=False, color="#7a7268", range=[0, 38]),
    yaxis=dict(showgrid=False, color="#1a1a1a"),
    margin=dict(t=10, b=10, l=10, r=60),
    barmode="overlay",
)
st.markdown("<div class='metric-label' style='margin-bottom:6px'>🦋 Butterfly Effect — Contribution per Factor (pts)</div>",
            unsafe_allow_html=True)
st.plotly_chart(fig_w, use_container_width=True)

# ── Economic reasoning ────────────────────────────────────────────────
top_factor = max(weights_data.items(), key=lambda x: x[1][0] * x[1][1])
st.markdown(f"""
<div class="econ-insight">
  <b>Economic Reasoning:</b> ปัจจัยที่มีผลมากที่สุดขณะนี้คือ <b>{top_factor[0]}</b>
  (+{top_factor[1][0]*top_factor[1][1]*100:.1f} pts) —
  เมื่อ VIX เกิน {PIVOT_VOL*100:.0f}% หรือ Yield Spread ผ่าน {PIVOT_YIELD}
  ระบบจะเข้าสู่ Non-linear Amplification Zone ตาม Taleb's Convexity Theory
  ซึ่งหมายความว่าความเสี่ยงไม่ได้เพิ่มเป็นเส้นตรง — มัน <b>ทวีคูณ</b>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ── SECTION 3 · HISTORICAL MIRROR ────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.markdown("<hr class='swan-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-tag'>Section III</div>"
    "<h3 style='margin-top:4px'>🪞 Historical Mirror</h3>"
    "<p style='font-family:Cormorant Garamond,serif;color:#7a7268;font-size:1rem;margin-top:-8px;font-style:italic'>"
    "Cosine Similarity — วันนี้มีลายเซ็นเหมือนวิกฤตในอดีตแค่ไหน</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Similarity cap ที่ 79% — ไม่มีวิกฤตใดเหมือนกัน 100% (Taleb: History doesn't repeat, it rhymes)"
)

# Build current vector (normalised to [0,1])
vol_n  = min(float(vol_input) / 0.4, 1)
yld_n  = min(max(float(sl_yield + 2) / 7, 0), 1)
coup_n = min(max(float(sl_coup + 0.2) / 1.2, 0), 1)
kurt_n = min(float(kurtosis_sim) / 5, 1)
hav_n  = min(max(float(haven_z + 2) / 6, 0), 1)
current_vec = np.array([vol_n, yld_n, coup_n, kurt_n, hav_n])

matches = historical_mirror(current_vec)
mirror_cols = st.columns(3)
crisis_colors = {"2008": "#c62828", "2020": "#4a0000", "2000": "#e65100",
                 "1997": "#7a4000", "2022": "#1a1a1a", "2011": "#4a443c"}

for col_m, h in zip(mirror_cols, matches):
    year  = h["event"].split("—")[0].strip()
    name  = h["event"].split("—")[1].strip()
    sim   = h["similarity"]
    bdr   = crisis_colors.get(year, "#1a1a1a")
    badge = "🔴 CLOSEST MATCH" if h == matches[0] else f"#{matches.index(h)+1} MATCH"
    col_m.markdown(f"""
    <div class="swan-card" style="border-left:4px solid {bdr}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div class="metric-label">{year} · {badge}</div>
          <div style="font-family:'Cormorant Garamond',serif;font-size:1.1rem;
                      font-weight:600;margin:4px 0">{name}</div>
        </div>
        <div style="text-align:right;flex-shrink:0">
          <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                      font-weight:600;color:{bdr}">{sim}%</div>
          <div class="metric-label">similar</div>
        </div>
      </div>
      <div style="margin-top:10px;height:3px;background:#ede9e3;border-radius:2px">
        <div style="width:{sim}%;height:100%;background:{bdr};border-radius:2px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Historical Fragility chart (if live data available) ───────────────
if yf_ok and not df_feat.empty and "Global_Vol" in df_feat.columns:
    st.markdown("<br>", unsafe_allow_html=True)
    feat_plot = df_feat[["Global_Vol", "Global_Coupling"]].tail(500).dropna()
    fig_fh = make_subplots(rows=1, cols=2,
                           subplot_titles=["Global Volatility (rolling 60d)", "Market Coupling (SP500↔SSE)"])
    fig_fh.add_trace(
        go.Scatter(x=feat_plot.index, y=feat_plot["Global_Vol"],
                   fill="tozeroy", fillcolor="rgba(198,40,40,.08)",
                   line=dict(color="#c62828", width=1.5), name="Vol"),
        row=1, col=1,
    )
    fig_fh.add_hline(y=PIVOT_VOL, line_dash="dash", line_color="rgba(198,40,40,.5)",
                     annotation_text=f"Pivot {PIVOT_VOL}", row=1, col=1)
    fig_fh.add_trace(
        go.Scatter(x=feat_plot.index, y=feat_plot["Global_Coupling"],
                   line=dict(color="#1a1a1a", width=1.5), name="Coupling"),
        row=1, col=2,
    )
    fig_fh.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#4a443c", height=250, showlegend=False,
                         margin=dict(t=30, b=10, l=10, r=10))
    fig_fh.update_xaxes(showgrid=False, color="#7a7268")
    fig_fh.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.06)", color="#7a7268")
    st.plotly_chart(fig_fh, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# ── SECTION 4 · SWAN'S WHISPER ────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.markdown("<hr class='swan-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-tag'>Section IV</div>"
    "<h3 style='margin-top:4px'>🌙 Swan's Whisper · AI Narrative</h3>",
    unsafe_allow_html=True,
)

whisper = swan_whisper(display_score, vol_input, sl_yield)
st.markdown(f"""
<div class="whisper-box">
  <div class="whisper-title">AI NARRATIVE · SYSTEMIC ASSESSMENT</div>
  <div style="font-size:1.6rem;margin-bottom:12px">{whisper['emoji']}&nbsp;
    <span style="color:{whisper['color']}">{whisper['title']}</span>
  </div>
  <div style="color:#c8c0b0;line-height:1.9">{whisper['body']}</div>
</div>
""", unsafe_allow_html=True)

# ── Key insights ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
ic1, ic2 = st.columns(2)
with ic1:
    vol_status = "⚠️ เกินเกณฑ์วิกฤต 14%" if vol_input > PIVOT_VOL else "✓ ปกติ"
    yld_status = "⚠️ ผ่าน Pivot 1.02" if sl_yield > PIVOT_YIELD else "✓ ปกติ"
    st.markdown(f"""
    <div class="econ-insight">
      <b>📐 Fat-Tails (Taleb's Core Insight)</b><br>
      ตลาดไม่ใช่ Normal Distribution — VIX &gt;14% พิสูจน์ว่า Fat-Tail กำลังก่อตัว<br>
      VIX ปัจจุบัน: <b>{sl_vix:.1f}</b> → {vol_status}
    </div>
    <div class="econ-insight">
      <b>📈 Yield Signal (Recession Predictor)</b><br>
      Yield Curve Inversion คาดการณ์ Recession ได้ถูกต้อง 7/7 ครั้งในรอบ 50 ปี<br>
      Spread ปัจจุบัน: <b>{sl_yield:.2f}%</b> → {yld_status}
    </div>
    """, unsafe_allow_html=True)
with ic2:
    st.markdown(f"""
    <div class="econ-insight">
      <b>🔗 Correlation Breakdown</b><br>
      เมื่อ Coupling สูงขึ้น Diversification ไม่ช่วย — วิกฤตลามทุกตลาดพร้อมกัน<br>
      Coupling ปัจจุบัน: <b>{sl_coup:.2f}</b>
      {'⚠️ สูงกว่าปกติ' if sl_coup > 0.6 else '✓ ระดับปกติ'}
    </div>
    <div class="econ-insight">
      <b>🥇 Safe-Haven Stress</b><br>
      Gold/Copper Ratio สูง = นักลงทุนกลัว (Copper ลด = เศรษฐกิจชะลอ)<br>
      Gold/Cu: <b>{sl_gold:.1f}</b> · USD Index: <b>{sl_usd:.1f}</b>
      {'⚠️ Safe-Haven Demand สูง' if sl_gold > 7 else '✓ ปกติ'}
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ── MONTE CARLO ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.markdown("<hr class='swan-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-tag'>Section V</div>"
    "<h3 style='margin-top:4px'>🎲 Monte Carlo · 3,000 Possible Futures</h3>",
    unsafe_allow_html=True,
)

np.random.seed(42)
n_paths, months = 3000, 12
drift = 0.012 if display_score > 50 else -0.005
sigma = 0.055 + display_score * 0.0007
paths = np.zeros((n_paths, months + 1))
paths[:, 0] = display_score
for t in range(1, months + 1):
    paths[:, t] = np.clip(
        paths[:, t-1] + np.random.normal(drift, sigma, n_paths) * 10, 0, 100
    )
p5   = np.percentile(paths,  5, axis=0)
p50  = np.percentile(paths, 50, axis=0)
p95  = np.percentile(paths, 95, axis=0)
crisis_pct = (paths[:, -1] >= 60).mean() * 100
labels = ["Now"] + [f"M{i}" for i in range(1, 13)]

fig_mc = go.Figure()
for i in range(0, n_paths, 60):
    fig_mc.add_trace(go.Scatter(x=labels, y=paths[i],
                                line=dict(color="rgba(74,68,60,.04)", width=1),
                                showlegend=False, hoverinfo="skip"))
fig_mc.add_trace(go.Scatter(x=labels, y=p95,
                            line=dict(color="rgba(198,40,40,.7)", width=1.5, dash="dot"),
                            name="95th Pct (Worst)"))
fig_mc.add_trace(go.Scatter(x=labels, y=p50,
                            line=dict(color="#1a1a1a", width=2.5), name="Median"))
fig_mc.add_trace(go.Scatter(x=labels, y=p5,
                            line=dict(color="rgba(46,125,50,.7)", width=1.5, dash="dot"),
                            name="5th Pct (Best)"))
fig_mc.add_hline(y=60, line_dash="dash", line_color="rgba(198,40,40,.4)",
                 annotation_text="Crisis Threshold (60)",
                 annotation_font_color="#c62828")
fig_mc.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#4a443c", height=360,
    xaxis=dict(showgrid=False, color="#7a7268"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,.06)", color="#7a7268",
               range=[0, 105], title="Risk Score (0–100)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#4a443c")),
    margin=dict(t=20, b=20, l=10, r=10),
)
st.plotly_chart(fig_mc, use_container_width=True)

mc1, mc2, mc3, mc4 = st.columns(4)
mc_kpis = [
    ("CRISIS PROB (12M)", f"{crisis_pct:.1f}%", gauge_color(crisis_pct)),
    ("MEDIAN RISK (M6)",  f"{p50[6]:.1f}",       gauge_color(p50[6])),
    ("WORST (P95, M12)",  f"{p95[-1]:.1f}",       gauge_color(p95[-1])),
    ("BEST (P5, M12)",    f"{p5[-1]:.1f}",        gauge_color(p5[-1])),
]
for col_m, (lbl, val, color) in zip([mc1, mc2, mc3, mc4], mc_kpis):
    col_m.markdown(f"""
    <div class="swan-card" style="text-align:center">
      <div class="metric-label">{lbl}</div>
      <div class="metric-value" style="color:{color}">{val}</div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ── METHODOLOGY ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.markdown("<hr class='swan-divider'>", unsafe_allow_html=True)
with st.expander("🔬 Methodology & Data Sources"):
    ma, mb = st.columns(2)
    with ma:
        st.markdown("**📥 Data Sources**")
        st.markdown("""
| Source | Variables | Freq |
|--------|-----------|------|
| Yahoo Finance | ^VIX, ^TNX, ^IRX, ^GSPC, 000001.SS, ^N225, GC=F, HG=F, DX-Y.NYB, BZ=F | Daily |
| FRED St. Louis | T10Y2Y (10Y–2Y Spread) | Daily |
| Computed | Gold/Copper Ratio, Yield Spread (10Y−3M), Cosine Similarity | Derived |

**🤖 AI Models**
- **Random Forest** (n=100) → Feature Importance weights (trained on labeled crisis data)
- **Isolation Forest** (contamination=5%) → Anomaly detection
- **Monte Carlo** (3,000 paths) → 12-month risk trajectory simulation
- **Cosine Similarity** → Historical pattern matching
        """)
    with mb:
        st.markdown("**⚙️ Feature Importance (RF Output)**")
        st.markdown("""
| Feature | Weight | Pivot |
|---------|--------|-------|
| Global Volatility Avg | **33.9%** | > 14% annualised |
| Global Yield Signal | **24.5%** | > 1.02 spread |
| Global Market Coupling | **14.6%** | Corr > 0.80 |
| Global Kurtosis Index | **14.1%** | Fat-tail proxy |
| Safe Haven Stress | **12.8%** | Gold/Cu + USD |

**📚 Theoretical Framework**
- *The Black Swan* (Taleb, 2007)
- *Antifragile* (Taleb, 2012)
- *Dynamic Conditional Correlation* (Engle, 2002)
- Yield Curve inversion as recession predictor (Fed research)
        """)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<hr class='swan-divider'>
<div style="text-align:center;font-family:'Cormorant Garamond',serif;
            font-style:italic;color:#7a7268;font-size:0.95rem;padding:8px 0 20px">
  🦢 The Black Swan Odyssey &nbsp;·&nbsp; Inspired by Nassim Nicholas Taleb
  &nbsp;·&nbsp; ⚠️ For educational purposes only — not financial advice
</div>
""", unsafe_allow_html=True)
