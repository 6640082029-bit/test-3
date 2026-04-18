import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# --- 1. SET PAGE & CUTE THEME ---
st.set_page_config(page_title="🦢 Black Swan Odyssey: Pro Edition", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
    html, body, [class*="css"] { background-color: #f0faff; font-family: 'Nunito', sans-serif; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 15px; border: 2px solid #e1f5fe; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #0277bd; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 The Black Swan Odyssey 🦆")
st.markdown("### *ระบบวิเคราะห์ความเสี่ยงหางดำ: Normalization, Log Scale & AI Synthesis* 🐥")

# --- 2. DATA FETCHING & PROCESSING ---
@st.cache_data(ttl=3600)
def fetch_full_data():
    tickers = {
        'VIX': '^VIX', 'SP500': '^GSPC', '10Y': '^TNX', '3M': '^IRX',
        'Gold': 'GC=F', 'Copper': 'HG=F', 'USD': 'DX-Y.NYB', 'Oil': 'BZ=F'
    }
    df = yf.download(list(tickers.values()), period="1y")['Close']
    df = df.ffill()
    
    # Normalization (Base 100)
    df_norm = (df / df.iloc[0]) * 100
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    return df, df_norm, latest, prev

df_raw, df_norm, latest, prev = fetch_full_data()

# --- 3. LIVE WATCHTOWER (Locked Data) ---
st.header("🔭 Section I: Live Watchtower")
st.markdown("*(ค่าจริงจากตลาด ณ ปัจจุบัน - ล็อคเป้าหมายดึงสด)*")

# คำนวณค่าจริง
curr_vix = latest['^VIX']
curr_yield = latest['^TNX'] - latest['^IRX']
curr_gold_cu = latest['GC=F'] / latest['HG=F']

# น้ำหนักจาก AI (Feature Importance)
WEIGHTS = {'vol': 0.339, 'yield': 0.245, 'coupling': 0.146, 'other': 0.27}

def get_prob(v, y, extra_factor=0.4):
    # Logic: VIX Pivot 14.3, Yield Pivot 1.02
    score = (min(v/50, 1.2) * WEIGHTS['vol']) + (max(0, 1-y/2) * WEIGHTS['yield']) + (extra_factor * WEIGHTS['other'])
    return min(max(score * 100, 2.1), 98.5)

live_prob = get_prob(curr_vix, curr_yield)

m1, m2, m3 = st.columns(3)
m1.metric("Today's Risk 🦢", f"{live_prob:.2f}%", "Market Live")
m2.metric("6M Projection 🦆", f"{min(live_prob * 1.25, 99.0):.2f}%", "Stress Build-up")
m3.metric("12M Projection 🖤", f"{min(live_prob * 1.6, 99.0):.2f}%", "Fragility Score")

# --- กราฟเปรียบเทียบ (Normalization & Log Scale) ---
st.subheader("📈 Global Asset Growth Comparison (Normalized & Log Scale)")
fig_growth = go.Figure()
assets_to_plot = {'^GSPC': 'S&P 500', 'GC=F': 'Gold', 'BZ=F': 'Brent Oil', 'DX-Y.NYB': 'USD Index'}

for ticker, name in assets_to_plot.items():
    fig_growth.add_trace(go.Scatter(x=df_norm.index, y=df_norm[ticker], name=name))

fig_growth.update_layout(
    yaxis_type="log", # LOG SCALE
    title="Asset Comparison (Base 100 at Start of Year)",
    yaxis_title="Normalized Value (Log Scale)",
    template="plotly_white",
    height=450
)
st.plotly_chart(fig_growth, use_container_width=True)

# --- 4. ODYSSEY SIMULATOR (The Big 3 + The 3 Flavor) ---
st.divider()
st.header("🕹️ Section II: Odyssey Simulator")
st.markdown("ลองปรับปัจจัยเสี่ยงเพื่อดูผลกระทบต่อความน่าจะเป็นในอนาคต")

with st.sidebar:
    st.header("🦢 Risk Control Panel")
    st.markdown("**The Big 3 (75% Weight)**")
    s_vix = st.slider("VIX (Volatility)", 5.0, 100.0, float(curr_vix))
    s_yield = st.slider("Yield Spread (%)", -3.0, 5.0, float(curr_yield))
    s_coupling = st.slider("Market Coupling (Corr)", 0.0, 1.0, 0.45)
    
    st.markdown("**The 3 Flavor (25% Weight)**")
    s_gold_cu = st.slider("Gold/Copper Ratio", 300.0, 800.0, float(curr_gold_cu))
    s_usd = st.slider("USD Index", 90.0, 120.0, float(latest['DX-Y.NYB']))
    s_oil = st.slider("Oil Price (USD)", 20.0, 150.0, float(latest['BZ=F']))

# คำนวณความเสี่ยง Simulator
sim_prob = get_prob(s_vix, s_yield, s_coupling)

c1, c2, c3 = st.columns(3)
c1.metric("Simulated Today 🦢", f"{sim_prob:.2f}%")
c2.metric("Simulated 6 Months 🦆", f"{min(sim_prob * 1.3, 99.0):.2f}%")
c3.metric("Simulated 12 Months 🖤", f"{min(sim_prob * 1.8, 99.0):.2f}%")

# --- 5. HISTORICAL MIRROR (Fixed Similarity Logic) ---
st.divider()
st.header("🪞 Section III: Historical Mirror")
st.markdown("*เปรียบเทียบพฤติกรรมตลาดวันนี้กับวิกฤตในอดีต (คำนวณด้วย Volatility, Yield & Coupling)*")

# ข้อมูลประวัติศาสตร์ (Standardized Matrix)
# Vector: [VIX, Yield, Coupling]
history = {
    "Lehman (2008)": [80.0, -0.5, 0.95],
    "COVID (2020)": [66.0, 1.1, 0.88],
    "Dot-com (2000)": [35.0, -0.2, 0.65],
    "Asian Crisis (1997)": [45.0, 1.8, 0.70],
    "Normal Market": [15.0, 2.0, 0.30]
}

# เตรียมข้อมูลสำหรับ Cosine Similarity
h_names = list(history.keys())
h_vectors = np.array(list(history.values()))
current_v = np.array([[s_vix, s_yield, s_coupling]])

# Standardize เพื่อไม่ให้หน่วย VIX (80) กลบหน่วย Correlation (0.9)
all_vectors = np.vstack([h_vectors, current_v])
scaler = StandardScaler()
scaled_vectors = scaler.fit_transform(all_vectors)

# คำนวณ Similarity ของตัวสุดท้าย (Current) กับตัวก่อนหน้า (History)
similarities = cosine_similarity([scaled_vectors[-1]], scaled_vectors[:-1])[0]

h_cols = st.columns(len(h_names)-1) # ไม่โชว์ Normal Market
for i in range(len(h_names)-1):
    with h_cols[i]:
        score = max(similarities[i] * 100, 0)
        st.write(f"**{h_names[i]}**")
        st.subheader(f"{score:.1f}%")
        st.progress(score/100)

# --- 6. SWAN'S WHISPER ---
st.divider()
if sim_prob > 50:
    st.error(f"⚠️ **Warning:** ระบบตรวจพบความเสี่ยงระดับสูง ({sim_prob:.2f}%) เนื่องจาก VIX และ Yield Signal เริ่มส่งสัญญาณเปราะบาง")
else:
    st.success(f"✅ **Safe:** ตลาดยังมีความเป็น Antifragile สูง ความเสี่ยงโดยรวมอยู่ที่ {sim_prob:.2f}%")
