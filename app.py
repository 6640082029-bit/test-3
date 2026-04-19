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

# --- 1. CONFIG (ต้องอยู่บรรทัดแรกสุดของคำสั่ง Streamlit) ---
st.set_page_config(page_title="Black Swan Prediction Dashboard", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
        font-family: 'Anuphan', sans-serif;
    }
    .main-title { color: #0F172A; text-align: center; font-weight: 600; font-size: 2.5rem; padding-bottom: 5px; }
    .sub-title { color: #64748B; text-align: center; margin-bottom: 30px; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-box { text-align: center; padding: 20px; border-radius: 12px; background-color: white; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND ENGINES ---

@st.cache_data(ttl=3600)
def get_market_data():
    tickers = {
        "NSE_India": "^NSEI", "NYSE": "^NYA", "SSE": "000001.SS",
        "JPX": "^N225", "Euronext": "^N100", "LSE": "^FTSE",
        "Gold": "GC=F", "Copper": "HG=F", "SP500": "^GSPC",
        "10Y_Yield": "^TNX", "2Y_Yield": "^IRX"
    }
    df = yf.download(list(tickers.values()), start="2000-01-01")['Close']
    df = df.ffill().bfill().rename(columns={v: k for k, v in tickers.items()})
    return df

def calculate_systemic_risk(df):
    markets = ['NYSE', 'SP500', 'Euronext', 'LSE', 'JPX', 'SSE', 'NSE_India']
    returns = df[markets].pct_change().dropna()
    
    # 4 Dimensions
    kurt = returns.rolling(252).kurt().mean(axis=1).iloc[-1]
    vol = (returns.rolling(252).std().mean(axis=1) * np.sqrt(252)).iloc[-1]
    
    last_60_days = returns.tail(60)
    corr_matrix = last_60_days.corr()
    coupling = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()
    
    yield_spread = (df['10Y_Yield'] - df['2Y_Yield']).iloc[-1]
    gold_copper = (df['Gold'] / df['Copper']).iloc[-1]
    
    # Scoring Logic
    score_k = np.clip((kurt / 12), 0, 1) * 30
    score_v = np.clip((vol / 0.45), 0, 1) * 20
    score_c = np.clip((coupling / 0.85), 0, 1) * 30
    score_m = (20 if yield_spread < 0 else 0) + (np.clip(gold_copper / 700, 0, 1) * 10)
    
    final_score = min(max(score_k + score_v + score_c + score_m, 2.5), 98.5)
    
    # Systemic Stress (For Probability Engine)
    stress = (vol * 0.3389 + abs(yield_spread) * 0.2450 + coupling * 0.1463 + (kurt/15) * 0.1411)
    return final_score, stress

def estimate_black_swan_mc(current_stress, horizon_days=30, simulations=50000):
    baseline_daily_prob = 1 / 5000 
    threshold = 0.0549
    risk_factor = np.power(current_stress / threshold, 1.15) if current_stress > threshold else current_stress / threshold
    adj_prob = baseline_daily_prob * risk_factor
    draws = np.random.random((simulations, horizon_days))
    return (np.any(draws < adj_prob, axis=1).sum() / simulations) * 100

# --- 3. EXECUTION ---
try:
    data = get_market_data()
    risk_index, stress_today = calculate_systemic_risk(data)
    
    # Forecast Scenario
    monthly_trend = 0.015
    p_today = estimate_black_swan_mc(stress_today)
    p_3m = estimate_black_swan_mc(stress_today + (monthly_trend * 3 * 0.85))
    p_6m = estimate_black_swan_mc(stress_today + (monthly_trend * 6 * 0.70))
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- 4. DISPLAY UI ---
st.markdown('<div class="main-title">🦢 Prediction of Black Swan Event</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ระบบวิเคราะห์ความเปราะบางและพยากรณ์โอกาสเกิดวิกฤตการณ์ทางการเงินโลก</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: Gauge Index ---
with col_left:
    st.subheader("📊 Global Systemic Risk Index")
    
    if risk_index >= 70:
        status, color, desc = "CRITICAL", "#EF4444", "ระบบมีความเปราะบางสูงสุด เสี่ยงต่อการพังทลายรุนแรง"
    elif risk_index >= 35:
        status, color, desc = "ELEVATED", "#F59E0B", "ระบบมีความเครียดสะสม ควรเฝ้าระวังปัจจัยแทรกซ้อน"
    else:
        status, color, desc = "NORMAL", "#10B981", "สภาวะตลาดโลกอยู่ในเกณฑ์ปกติ มีความยืดหยุ่นสูง"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = risk_index,
        number = {'font': {'size': 80, 'color': '#1E293B'}, 'valueformat': '.1f'},
        gauge = {
            'axis': {'range': [0, 100]}, 'bar': {'color': "#1E293B"},
            'steps': [{'range': [0, 35], 'color': '#BBF7D0'}, {'range': [35, 70], 'color': '#FEF08A'}, {'range': [70, 100], 'color': '#FECACA'}],
            'threshold': {'line': {'color': "#EF4444", 'width': 4}, 'thickness': 0.8, 'value': 90}}
    ))
    fig.update_layout(height=400, margin=dict(t=20, b=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
        <div class="status-box" style="border-top: 5px solid {color};">
            <h2 style="color: {color}; margin: 0;">{status}</h2>
            <p style="color: #64748B; margin-top: 5px;">{desc}</p>
        </div>
    """, unsafe_allow_html=True)

# --- RIGHT COLUMN: Probability Forecast ---
with col_right:
    st.subheader("🔮 Probability Forecast")
    st.write("")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Today", f"{p_today:.2f}%")
    m2.metric("3M Forward", f"{p_3m:.2f}%", f"{p_3m-p_today:+.2f}%")
    m3.metric("6M Forward", f"{p_6m:.2f}%", f"{p_6m-p_today:+.2f}%")
    
    # Path Chart
    df_path = pd.DataFrame({
        "Timeline": ["Today", "3M Forecast", "6M Forecast"],
        "Prob": [p_today, p_3m, p_6m]
    })
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_path["Timeline"], y=df_path["Prob"],
        mode='lines+markers+text', text=[f"{p_today:.1f}%", f"{p_3m:.1f}%", f"{p_6m:.1f}%"],
        textposition="top center", line=dict(color=color, width=4),
        fill='tozeroy', fillcolor=f'rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1)'
    ))
    fig_line.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           yaxis=dict(showgrid=True, gridcolor='#E2E8F0'))
    st.plotly_chart(fig_line, use_container_width=True)
    
    with st.expander("ℹ️ Methodology Note"):
        st.caption("คำนวณผ่าน Monte Carlo Simulation 50,000 รอบ โดยอิงจาก Daily Baseline Risk 1 ใน 5,000 วัน และปรับค่าความเปราะบางด้วยกฎเลขยกกำลัง (Power Law)")

st.divider()
st.caption("Data sources: Yahoo Finance | Methodology: Antifragile Systemic Risk Model")
