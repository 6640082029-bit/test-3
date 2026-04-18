import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# --- 1. SET PAGE & CUTE THEME ---
st.set_page_config(page_title="🦢 Black Swan Odyssey: Cute Edition", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0faff; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 15px; border: 2px solid #e1f5fe; }
    h1, h2, h3 { color: #0277bd; font-family: 'Comic Sans MS', cursive; }
    .stButton>button { border-radius: 20px; background-color: #0288d1; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 The Black Swan Odyssey 🦆")
st.markdown("### *ระบบเฝ้าระวังหงส์ดำสุดคิวท์ โดย AI และสถิติระดับโลก* 🐥")

# --- 2. DATA FETCHING (LIVE DATA) ---
@st.cache_data(ttl=3600)
def fetch_live_data():
    tickers = {
        'VIX': '^VIX',
        'SP500': '^GSPC',
        '10Y': '^TNX',
        '3M': '^IRX',
        'Gold': 'GC=F',
        'Copper': 'HG=F',
        'USD': 'DX-Y.NYB'
    }
    data = yf.download(list(tickers.values()), period="5d")['Close']
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    # Calculate Features
    vix = latest['^VIX']
    sp500 = latest['^GSPC']
    yield_spread = latest['^TNX'] - latest['^IRX']
    gold_cu = latest['GC=F'] / latest['HG=F']
    usd = latest['DX-Y.NYB']
    
    return {
        'vix': vix, 'vix_chg': (vix - prev['^VIX'])/prev['^VIX']*100,
        'sp500': sp500, 'sp500_chg': (sp500 - prev['^GSPC'])/prev['^GSPC']*100,
        'yield': yield_spread,
        'gold_cu': gold_cu,
        'usd': usd
    }

live = fetch_live_data()

# --- 3. CORE LOGIC & WEIGHTS ---
# Weights based on your AI Feature Importance
WEIGHTS = {
    'vol': 0.339,
    'yield': 0.245,
    'coupling': 0.146,
    'kurtosis': 0.141,
    'safe_haven': 0.128
}

def calculate_risk(vol, yield_sp, coupling, extra=0):
    # Normalize inputs for a simplified model
    vol_score = min(vol / 40, 1.0) * WEIGHTS['vol']
    # Yield spread: Risk increases as it approaches 0 or becomes negative
    yield_score = max(0, (1.0 - yield_sp/2)) * WEIGHTS['yield']
    coupling_score = coupling * WEIGHTS['coupling']
    
    total_prob = (vol_score + yield_score + coupling_score + extra) * 100
    return min(max(total_prob, 1.5), 98.0) # Limit between 1.5% and 98%

# --- 4. SIDEBAR - SIMULATOR CONTROLS ---
st.sidebar.header("🕹️ Black Swan Simulator")
st.sidebar.markdown("ลองปรับค่าสมมติเพื่อดูผลกระทบ 🦆")

sim_vix = st.sidebar.slider("VIX (Volatility)", 5.0, 100.0, float(live['vix']))
sim_yield = st.sidebar.slider("Yield Spread (10Y-3M %)", -2.0, 5.0, float(live['yield']))
sim_coupling = st.sidebar.slider("Market Coupling (Corr)", 0.0, 1.0, 0.4)
sim_gold_cu = st.sidebar.slider("Gold/Copper Ratio", 300.0, 800.0, float(live['gold_cu']))

# --- 5. LIVE WATCHTOWER (Locked to Real Data) ---
st.header("🔭 Section I: Live Watchtower")
st.markdown("*(ข้อมูลปัจจุบันจากตลาดจริง ไม่เปลี่ยนแปลงตาม Simulator)*")

col1, col2, col3 = st.columns(3)
# Calculate Live Probabilities
live_prob_now = calculate_risk(live['vix'], live['yield'], 0.4)

with col1:
    st.metric("Probability Today 🦢", f"{live_prob_now:.2f}%", "Real-time")
with col2:
    st.metric("6 Months Outlook 🦆", f"{live_prob_now * 1.2:.2f}%", "Stress Build-up")
with col3:
    st.metric("12 Months Outlook 🖤", f"{live_prob_now * 1.5:.2f}%", "Fragility Score")

st.subheader("📊 Live Market Snapshot")
m1, m2, m3, m4 = st.columns(4)
m1.metric("S&P 500", f"{live['sp500']:,.2f}", f"{live['sp500_chg']:.2f}%")
m2.metric("VIX Index", f"{live['vix']:.2f}", f"{live['vix_chg']:.2f}%")
m3.metric("Yield Spread", f"{live['yield']:.2f}%")
m4.metric("Gold/Cu Ratio", f"{live['gold_cu']:.2f}")

# --- 6. SIMULATOR OUTPUT ---
st.divider()
st.header("🎮 Section II: Odyssey Simulator")
st.markdown("*(ผลลัพธ์จากการปรับค่าในแถบด้านซ้าย)*")

sim_prob = calculate_risk(sim_vix, sim_yield, sim_coupling)

sc1, sc2 = st.columns([2, 1])
with sc1:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = sim_prob,
        title = {'text': "Simulated Risk Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0288d1"},
            'steps': [
                {'range': [0, 20], 'color': "#c8e6c9"},
                {'range': [20, 50], 'color': "#fff9c4"},
                {'range': [50, 100], 'color': "#ffcdd2"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_view_with_height=True)

with sc2:
    st.markdown("### 🤖 AI Weight Weights")
    st.write(f"1. Volatility (VIX): **{WEIGHTS['vol']*100:.1f}%**")
    st.write(f"2. Yield Signal: **{WEIGHTS['yield']*100:.1f}%**")
    st.write(f"3. Market Coupling: **{WEIGHTS['coupling']*100:.1f}%**")
    st.info("💡 น้ำหนักเหล่านี้อ้างอิงจาก Feature Importance ของ Random Forest Model")

# --- 7. HISTORICAL MIRROR ---
st.divider()
st.header("🪞 Section III: Historical Mirror")
st.markdown("*เปรียบเทียบ Pattern วันนี้กับวิกฤตในอดีต (Based on Volatility & Correlation)*")

# Mock Historical Data for Similarity Check
hist_events = {
    "Lehman Moment (2008)": [80.0, -0.5, 0.9],
    "COVID Crash (2020)": [65.0, 1.2, 0.85],
    "Dot-com Bust (2000)": [35.0, 0.2, 0.6],
    "Asian Crisis (1997)": [40.0, 1.5, 0.75]
}

current_vector = np.array([[sim_vix, sim_yield, sim_coupling]])

h_cols = st.columns(4)
for i, (name, vector) in enumerate(hist_events.items()):
    sim_score = cosine_similarity(current_vector, [vector])[0][0] * 100
    with h_cols[i]:
        st.markdown(f"**{name}**")
        st.subheader(f"{sim_score:.1f}%")
        st.caption("Similarity Score")

# --- 8. SWAN'S WHISPER (AI NARRATIVE) ---
st.divider()
st.header("💬 Section IV: Swan's Whisper")

if sim_prob < 20:
    narrative = "🦢 **หงส์ขาวนิ่งสงบ:** ตลาดยังอยู่ในภาวะปกติ ความเสี่ยงหางอ้วน (Fat-tail) ยังไม่ปรากฏชัดเจน"
elif sim_prob < 50:
    narrative = "🐥 **ลูกหงส์เริ่มตื่นตัว:** เริ่มเห็นความเครียดสะสมในบางปัจจัย โดยเฉพาะ Yield Spread หรือความผันผวนที่ขยับขึ้น"
else:
    narrative = "🖤 **หงส์ดำสยายปีก:** สัญญาณอันตราย! VIX ทะลุ Pivot 14.3 และ Yield Curve ส่งสัญญาณเปราะบาง ระวังแรงกระแทก Non-linear"

st.success(narrative)
st.markdown(f"""
- **VIX Analysis:** ปัจจุบันอยู่ที่ {sim_vix:.2f} (Pivot Point ของ AI คือ 14.3)
- **Yield Insight:** Spread อยู่ที่ {sim_yield:.2f}% (สัญญาณ Recession มักเกิดเมื่อเข้าใกล้ 0 หรือติดลบ)
- **Taleb's Logic:** วิกฤต Black Swan มักเกิดเมื่อคนประมาท และความสัมพันธ์ของตลาด (Coupling) พุ่งสูงขึ้นจนกระจายความเสี่ยงไม่ได้
""")
