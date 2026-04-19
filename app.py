import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# --- 1. SET PAGE & THEME (Light Blue Background + Anuphan Font) ---
st.set_page_config(page_title="Swan Odyssey: Global Watch", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #E3F2FD !important;
        font-family: 'Anuphan', sans-serif;
    }
    
    h1, h2, h3, b, strong, .stMarkdown {
        font-family: 'Anuphan', sans-serif;
        color: #01579B;
    }

    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #BBDEFB;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }

    [data-testid="stSidebar"] {
        background-color: #f8fbff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 The Black Swan Odyssey 🌍")
st.markdown("### *Global Market Watch: 6 Nations & Economic Indicators* 🐥")

# --- 2. DATA FETCHING FUNCTION ---
@st.cache_data(ttl=3600)
def fetch_global_data():
    tickers = {
        'NSE_India': '^NSEI', 
        'NYSE': '^NYA', 
        'SSE': '000001.SS', 
        'JPX': '^N225', 
        'Euronext': '^N100', 
        'LSE': '^FTSE',
        'VIX': '^VIX',
        'Gold': 'GC=F',
        'Crude_Oil': 'BZ=F',
        'Copper': 'HG=F',
        'USD_Index': 'DX-Y.NYB',
        '10Y_Bond': '^TNX',
        '2Y_Bond': '^IRX'
    }
    
    # ดึงข้อมูลย้อนหลัง 1 ปี
    df = yf.download(list(tickers.values()), period="1y")['Close']
    
    # ป้องกัน Error: indexer out-of-bounds หากดึงข้อมูลไม่ได้
    if df.empty or len(df) < 2:
        return None, None, None, None

    df = df.ffill().bfill()
    
    # เปลี่ยนชื่อ Column
    inv_tickers = {v: k for k, v in tickers.items()}
    df = df.rename(columns=inv_tickers)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Normalization (Base 100) สำหรับกราฟราคา
    price_cols = ['NSE_India', 'NYSE', 'SSE', 'JPX', 'Euronext', 'LSE', 'Gold', 'Crude_Oil', 'USD_Index']
    df_norm = (df[price_cols] / df[price_cols].iloc[0]) * 100
    
    return df, df_norm, latest, prev

# --- 3. EXECUTE FETCHING ---
df_raw, df_norm, latest, prev = fetch_global_data()

if df_raw is None:
    st.error("⚠️ ไม่สามารถดึงข้อมูลจากตลาดโลกได้ในขณะนี้ กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ตหรือรีเฟรชหน้าจอ 🦆")
    st.stop()

# --- 4. LOGIC & WEIGHTS ---
WEIGHTS = {'vol': 0.339, 'yield': 0.245, 'coupling': 0.146, 'other': 0.27}
curr_yield_spread = latest['10Y_Bond'] - latest['2Y_Bond']
curr_gold_cu = latest['Gold'] / latest['Copper']

def calc_risk(v, y, c=0.45):
    # AI Pivot Logic: VIX 14.3, Yield 1.02
    v_score = (v / 45) * WEIGHTS['vol']
    y_score = max(0, (1.1 - y/2)) * WEIGHTS['yield']
    c_score = c * WEIGHTS['coupling']
    return min(max((v_score + y_score + c_score + 0.1) * 100, 1.5), 98.5)

# --- 5. SECTION I: LIVE WATCHTOWER ---
st.header("🔭 Section I: Live Watchtower")

l_prob = calc_risk(latest['VIX'], curr_yield_spread)
p1, p2, p3 = st.columns(3)
p1.metric("Today's Risk 🦢", f"{l_prob:.2f}%", "Market Live")
p2.metric("6 Months Outlook 🦆", f"{min(l_prob*1.2, 99.0):.2f}%")
p3.metric("12 Months Outlook 🖤", f"{min(l_prob*1.6, 99.0):.2f}%")

st.subheader("🌎 Global Market Snapshot (6 Nations)")
m_cols = st.columns(6)
countries = ['NSE_India', 'NYSE', 'SSE', 'JPX', 'Euronext', 'LSE']
for i, c in enumerate(countries):
    change = (latest[c] - prev[c]) / prev[c] * 100
    m_cols[i].metric(c.replace('_', ' '), f"{latest[c]:,.0f}", f"{change:.2f}%")

# --- 6. GRAPH: NORMALIZED & LOG SCALE ---
st.subheader("📈 Global Growth Comparison (Normalized to 100 & Log Scale)")
fig = go.Figure()
for col in df_norm.columns:
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], name=col))

fig.update_layout(
    yaxis_type="log",
    template="plotly_white",
    hovermode="x unified",
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig, use_container_width=True)

# --- 7. SECTION II: ODYSSEY SIMULATOR ---
st.divider()
st.header("🕹️ Section III: Odyssey Simulator")
with st.sidebar:
    st.header("⚙️ Risk Controls")
    s_vix = st.slider("VIX (Fear Index)", 5.0, 100.0, float(latest['VIX']))
    s_yield = st.slider("Yield Spread (%)", -3.0, 5.0, float(curr_yield_spread))
    s_coupling = st.slider("Market Coupling", 0.0, 1.0, 0.45)
    st.write("---")
    s_gold_cu = st.slider("Gold/Copper Ratio", 300.0, 800.0, float(curr_gold_cu))
    s_usd = st.slider("USD Index", 90.0, 120.0, float(latest['USD_Index']))

sim_p = calc_risk(s_vix, s_yield, s_coupling)
s1, s2, s3 = st.columns(3)
s1.metric("Simulated Today 🦢", f"{sim_p:.2f}%")
s2.metric("Simulated 6 Months 🦆", f"{min(sim_p*1.3, 99.0):.2f}%")
s3.metric("Simulated 12 Months 🖤", f"{min(sim_p*1.8, 99.0):.2f}%")

# --- 8. SECTION III: HISTORICAL MIRROR ---
st.divider()
st.header("🪞 Section III: Historical Mirror")
history = {
    "Lehman (2008)": [80.0, -0.5, 0.95],
    "COVID (2020)": [66.0, 1.1, 0.88],
    "Dot-com (2000)": [35.0, -0.2, 0.65],
    "Asian Crisis (1997)": [45.0, 1.5, 0.70]
}

def calc_sim(v1, v2):
    def n(v): return np.array([v[0]/100, (v[1]+3)/10, v[2]])
    vec1, vec2 = n(v1), n(v2)
    return cosine_similarity([vec1], [vec2])[0][0]

h_cols = st.columns(4)
curr_v = [s_vix, s_yield, s_coupling]
for i, (name, vec) in enumerate(history.items()):
    score = calc_sim(curr_v, vec) * 100
    display_score = min(max((score - 70) * 3.3, 5.0), 99.5) if score > 70 else score/2
    with h_cols[i]:
        st.write(f"**{name}**")
        st.subheader(f"{display_score:.1f}%")
        st.progress(display_score/100)

# --- 9. SWAN'S WHISPER ---
st.divider()
if sim_p > 50:
    st.error(f"🖤 **AI Assessment:** ตลาดเริ่มส่อแววเปราะบาง สัญญาณจำลองมีความคล้ายคลึงกับวิกฤตในอดีต")
else:
    st.success(f"🦢 **AI Assessment:** ตลาดยังอยู่ในสภาวะ Antifragile หงส์ขาวยังคงว่ายน้ำอย่างสงบสุข")
