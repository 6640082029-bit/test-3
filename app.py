import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SET PAGE & THEME (Anuphan Font + Blue Background) ---
st.set_page_config(page_title="Swan Odyssey: 1975 Era", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #E3F2FD !important;
        font-family: 'Anuphan', sans-serif;
    }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 15px; border: 1px solid #BBDEFB; }
    h1, h2, h3 { color: #01579B; font-family: 'Anuphan', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 The Black Swan Odyssey 🌍")
st.markdown("### *Long-term Historical Horizon (Since 1975) & Real-time Assets* 🐥")

# --- 2. DATA FETCHING (Start from 1975) ---
@st.cache_data(ttl=3600)
def fetch_historical_data():
    tickers = {
        'NSE_India': '^NSEI', 'NYSE': '^NYA', 'SSE': '000001.SS', 
        'JPX': '^N225', 'Euronext': '^N100', 'LSE': '^FTSE',
        'VIX': '^VIX', 'Gold': 'GC=F', 'Crude_Oil': 'BZ=F',
        'Copper': 'HG=F', 'USD_Index': 'DX-Y.NYB',
        '10Y_Bond': '^TNX', '2Y_Bond': '^IRX'
    }
    
    # ดึงข้อมูลโดยกำหนดวันเริ่มต้นเป็นปี 1975
    df = yf.download(list(tickers.values()), start="1975-01-01")['Close']
    
    if df.empty:
        return None, None, None, None

    # จัดการข้อมูล: เติมค่าว่างเพื่อให้คำนวณ Normalization ได้เสถียร
    df = df.ffill().bfill()
    
    # เปลี่ยนชื่อ Column
    inv_map = {v: k for k, v in tickers.items()}
    df = df.rename(columns=inv_map)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- 3. NORMALIZATION LOGIC (Base 100) ---
    price_cols = ['NSE_India', 'NYSE', 'SSE', 'JPX', 'Euronext', 'LSE', 'Gold', 'Crude_Oil', 'USD_Index']
    valid_cols = [c for c in price_cols if c in df.columns]
    
    # ฟังก์ชันปรับฐาน: ให้แต่ละตัวเริ่มที่ 100 ณ วันแรกที่มีข้อมูลของตัวเอง
    df_norm = df[valid_cols].copy()
    for col in df_norm.columns:
        first_valid_value = df_norm[col].dropna().iloc[0]
        df_norm[col] = (df_norm[col] / first_valid_value) * 100
    
    return df, df_norm, latest, prev

df_raw, df_norm, latest, prev = fetch_historical_data()

if df_raw is not None:
    # --- 4. SECTION I: LIVE WATCHTOWER (พร้อมหน่วย) ---
    st.header("🔭 Section I: Live Watchtower")
    
    # ตลาดหุ้น 6 ประเทศ
    st.subheader("🌎 Global Equity Indices")
    m_cols = st.columns(6)
    market_units = {
        'NSE_India': 'INR (Nifty 50)', 'NYSE': 'USD (Composite)', 'SSE': 'CNY (Composite)', 
        'JPX': 'JPY (Nikkei 225)', 'Euronext': 'EUR (Enext 100)', 'LSE': 'GBP (FTSE 100)'
    }
    for i, (key, label) in enumerate(market_units.items()):
        val = latest.get(key, 0)
        p_val = prev.get(key, 0)
        change = ((val - p_val) / p_val * 100) if p_val != 0 else 0
        m_cols[i].metric(label, f"{val:,.2f}", f"{change:.2f}%")

    # สินทรัพย์อื่นๆ
    st.subheader("📊 Commodity & Indicators")
    a_cols = st.columns(4)
    a_cols[0].metric("Gold (XAU/USD)", f"${latest['Gold']:,.2f}", "USD / t oz")
    a_cols[1].metric("Brent Oil", f"${latest['Crude_Oil']:.2f}", "USD / bbl")
    a_cols[2].metric("USD Index", f"{latest['USD_Index']:.2f}", "Points")
    a_cols[3].metric("VIX Index", f"{latest['VIX']:.2f}", f"{latest['VIX']-prev['VIX']:.2f}")

    # --- 5. GRAPH: SINCE 1975 LOG SCALE ---
    st.subheader(f"📈 Long-term Growth Comparison (Since 1975 - Normalized)")
    st.caption("หมายเหตุ: ดัชนีบางตัวอาจเริ่มแสดงผลช้ากว่าปี 1975 ตามวันที่มีข้อมูลครั้งแรกในระบบ")

    fig = go.Figure()
    for col in df_norm.columns:
        fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], name=col, line=dict(width=1.2)))

    fig.update_layout(
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=650,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        legend=dict(orientation="h", y=1.1),
        yaxis=dict(title="Growth Index (Base 100)"),
        xaxis=dict(title="Year")
    )
    st.plotly_chart(fig, use_container_width=True)
#SECTION 2 Predict
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# --- 1. CONFIG (ต้องชิดขอบซ้ายสุด ห้ามมีช่องว่างข้างหน้า) ---
st.set_page_config(page_title="Global Systemic Risk Index", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
        font-family: 'Anuphan', sans-serif;
    }
    h1 { color: #0F172A; text-align: center; font-weight: 600; padding-top: 20px; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 Global Systemic Risk Index")
st.markdown("<center style='color: #64748B;'>ดัชนีชี้วัดความเปราะบางของระบบการเงินโลก (Today's Fragility Score)</center>", unsafe_allow_html=True)

# --- 2. BACKEND ENGINE (ซ่อนการคำนวณ 4 มิติไว้ที่นี่) ---
@st.cache_data(ttl=3600)
def get_systemic_risk_data():
    tickers = {
        "NSE_India": "^NSEI", "NYSE": "^NYA", "SSE": "000001.SS",
        "JPX": "^N225", "Euronext": "^N100", "LSE": "^FTSE",
        "Gold": "GC=F", "Copper": "HG=F", "SP500": "^GSPC",
        "10Y_Yield": "^TNX", "2Y_Yield": "^IRX"
    }
    
    # ดึงข้อมูล
    df = yf.download(list(tickers.values()), start="2000-01-01")['Close']
    df = df.ffill().bfill().rename(columns={v: k for k, v in tickers.items()})
    
    # คำนวณ Returns
    markets = ['NYSE', 'SP500', 'Euronext', 'LSE', 'JPX', 'SSE', 'NSE_India']
    returns = df[markets].pct_change().dropna()

    # มิติที่ 1: Kurtosis (Fat-Tail)
    kurt = returns.rolling(252).kurt().mean(axis=1).iloc[-1]
    
    # มิติที่ 2: Volatility (Average)
    vol = (returns.rolling(252).std().mean(axis=1) * np.sqrt(252)).iloc[-1]
    
    # มิติที่ 3: Market Coupling (60-day Cross-Correlation)
    last_60_days = returns.tail(60)
    corr_matrix = last_60_days.corr()
    coupling = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()
    
    # มิติที่ 4: Macro Signals
    yield_spread = (df['10Y_Yield'] - df['2Y_Yield']).iloc[-1]
    gold_copper = (df['Gold'] / df['Copper']).iloc[-1]
    
    # ตรรกะ Score (0-100)
    score_k = np.clip((kurt / 12), 0, 1) * 30
    score_v = np.clip((vol / 0.45), 0, 1) * 20
    score_c = np.clip((coupling / 0.85), 0, 1) * 30
    score_m = (20 if yield_spread < 0 else 0) + (np.clip(gold_copper / 700, 0, 1) * 10)
    
    final_risk = min(max(score_k + score_v + score_c + score_m, 2.5), 98.5)
    
    # คำนวณค่าการเปลี่ยนแปลง (Trend) สำหรับ Predictor
    trend_22d = (returns.tail(22).mean(axis=1).mean()) * 1000 # ขยายสัญญาณ
    risk_3m = min(max(final_risk + trend_22d, 1.5), 99.0)
    risk_6m = min(max(final_risk + (trend_22d * 2), 1.5), 99.0)
    
    return final_risk, risk_3m, risk_6m

# ดึงผลลัพธ์
try:
    risk_today, r3m, r6m = get_systemic_risk_data()
except:
    st.error("ไม่สามารถเชื่อมต่อข้อมูลตลาดโลกได้")
    st.stop()

# --- 3. VISUAL GAUGE (เข็มไมล์) ---
st.markdown("<br>", unsafe_allow_html=True)

# ตรรกะการเลือกสถานะภาษาอังกฤษ (English Short Label)
if risk_today >= 70:
    status_label = "CRITICAL"
    risk_color = "#EF4444" # แดง
elif risk_today >= 35:
    status_label = "ELEVATED"
    risk_color = "#F59E0B" # ส้ม/เหลืองเข้ม
else:
    status_label = "NORMAL"
    risk_color = "#10B981" # เขียว

# สร้าง Gauge Chart
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = risk_today,
    domain = {'x': [0, 1], 'y': [0, 1]},
    # ตั้งค่าตัวเลขดัชนี (ไม่มี %)
    number = {'font': {'size': 90, 'color': '#1E293B'}, 'valueformat': '.1f'},
    gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
        'bar': {'color': "#1E293B"},
        'bgcolor': "white",
        'steps': [
            {'range': [0, 35], 'color': '#BBF7D0'},
            {'range': [35, 70], 'color': '#FEF08A'},
            {'range': [70, 100], 'color': '#FECACA'}],
        'threshold': {
            'line': {'color': "#EF4444", 'width': 5},
            'thickness': 0.75,
            'value': 90}}
))

# ปรับตำแหน่ง Label ให้อยู่เหนือตัวเลข
fig.add_annotation(
    x=0.5, y=0.4, # ตำแหน่งเหนือตัวเลขดัชนี
    text=status_label,
    showarrow=False,
    font=dict(size=22, color=risk_color, family="Anuphan"),
    bgcolor="white",
    bordercolor=risk_color,
    borderwidth=1,
    borderpad=5
)

fig.update_layout(
    height=450, 
    margin=dict(l=30, r=30, t=20, b=20),
    paper_bgcolor='rgba(0,0,0,0)',
    font={'family': "Anuphan"}
)

st.plotly_chart(fig, use_container_width=True)

# --- 4. DYNAMIC INTERPRETATION DISPLAY ---
st.markdown(f"""
    <div style="text-align: center; padding: 25px; border-radius: 15px; 
                background-color: white; border-top: 5px solid {risk_color}; 
                box-shadow: 0px 4px 10px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <h2 style="color: {risk_color}; margin: 0; font-weight: 600;">{risk_label}</h2>
        <p style="color: #64748B; font-size: 1.1rem; margin-top: 8px;">{risk_desc}</p>
    </div>
    """, unsafe_allow_html=True)

# --- 4. PREDICTIONS ---
st.divider()
c1, c2 = st.columns(2)
c1.metric("3-Month Forward Predictor", f"{r3m:.2f}%", f"{r3m-risk_today:+.2f}%")
c2.metric("6-Month Forward Predictor", f"{r6m:.2f}%", f"{r6m-risk_today:+.2f}%")

# --- 5. INTERPRETATION ---
if risk_today > 70:
    st.error("🚨 **CRITICAL FRAGILITY:** ระบบเปราะบางสูง โอกาสเกิด Black Swan มีมาก")
elif risk_today > 35:
    st.warning("⚠️ **ELEVATED RISK:** ความเสี่ยงเริ่มสะสม ควรระมัดระวัง")
else:
    st.success("Swan Status: **ANTIFRAGILE**")
# --- 5. BLACK SWAN PROBABILITY ENGINE (Backend) ---

def calculate_black_swan_prob(frag_row):
    # 1. คำนวณ Systemic Stress (Insight Mining)
    # ให้น้ำหนักตามที่ AI ของคุณเรียนรู้มา (Feature Importance)
    stress_score = (
        (frag_row['Volatility'] * 0.3389) +
        (abs(frag_row['Yield_Signal']) * 0.2450) + # Yield Curve Stress
        (frag_row['Coupling'] * 0.1463) +
        (frag_row['Kurtosis'] / 15 * 0.1411) +    # Normalize Kurtosis
        (frag_row['Safe_Haven'] / 800 * 0.1284)
    )
    
    # 2. คำนวณ Probability โดยเทียบกับค่าวิกฤต (Taleb's Evidence: 0.0549)
    # ใช้ Sigmoid-like scaling เพื่อเปลี่ยน Stress เป็น %
    critical_threshold = 0.0549
    probability = 1 / (1 + np.exp(-100 * (stress_score - critical_threshold)))
    
    return min(max(probability * 100, 0.0), 100.0), stress_score

# คำนวณวันนี้
prob_today, stress_today = calculate_black_swan_prob(latest)

# คำนวณล่วงหน้า (ใช้ Acceleration ของความเครียดสะสม)
stress_velocity = (df_fragility['Coupling'].diff().tail(22).mean() * 5) # ตัวเร่งจาก Coupling
prob_3m, _ = calculate_black_swan_prob(latest + (stress_velocity * 3))
prob_6m, _ = calculate_black_swan_prob(latest + (stress_velocity * 6))

# --- 6. DISPLAY: PROBABILITY DASHBOARD ---
st.divider()
st.subheader("🔮 Black Swan Probability Forecast")
st.markdown("<center><i>Probability of a Black Swan event based on Systemic Stress Pattern Recognition</i></center>", unsafe_allow_html=True)

p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    st.metric("Probability (Today)", f"{prob_today:.2f}%")
    st.caption("Based on current pattern")

with p_col2:
    st.metric("3-Month Forecast", f"{prob_3m:.2f}%", f"{prob_3m - prob_today:+.2f}%")
    st.caption("Projected Stress Trend")

with p_col3:
    st.metric("6-Month Forecast", f"{prob_6m:.2f}%", f"{prob_6m - prob_today:+.2f}%")
    st.caption("Long-term Fragility Path")

# --- 7. STRATEGIC ALERT (จากบทเรียน Taleb) ---
st.markdown("<br>", unsafe_allow_html=True)
if stress_today > 0.0549:
    st.error(f"⚠️ **SYSTEMIC STRESS ALERT:** ระดับความเครียดปัจจุบันอยู่ที่ {stress_today:.4f} ซึ่งสูงกว่าเกณฑ์วิกฤต (0.0549) ระวังการเกิด Black Swan ภายใน 30-90 วัน!")
else:
    st.info(f"🛡️ **SYSTEMIC STRESS:** {stress_today:.4f} / Threshold: 0.0549 (ระบบยังอยู่ในสภาวะปกติ)")

st.caption("Machine Learning Logic: Random Forest Weighted Analysis (Volatility, Yield, Coupling, Kurtosis, Safe Haven)")
