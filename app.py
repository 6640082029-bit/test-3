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

# --- 1. CONFIG (ต้องอยู่บนสุด) ---
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
    .status-box { text-align: center; padding: 25px; border-radius: 12px; background-color: white; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); }
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
    # ดึงข้อมูลย้อนหลังเพื่อให้ครอบคลุมการคำนวณ Rolling 252 วัน
    df = yf.download(list(tickers.values()), start="2022-01-01")['Close']
    df = df.ffill().bfill().rename(columns={v: k for k, v in tickers.items()})
    return df

def calculate_metrics(df):
    markets = ['NYSE', 'SP500', 'Euronext', 'LSE', 'JPX', 'SSE', 'NSE_India']
    returns = df[markets].pct_change().dropna()
    
    # 1. Kurtosis (Fat-Tail Dimension)
    kurt = returns.rolling(252).kurt().mean(axis=1).iloc[-1]
    
    # 2. Volatility (Panic Dimension)
    vol = (returns.rolling(252).std().mean(axis=1) * np.sqrt(252)).iloc[-1]
    
    # 3. Market Coupling (Fragility Dimension)
    last_60_days = returns.tail(60)
    corr_matrix = last_60_days.corr()
    coupling = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()
    
    # 4. Yield Spread (Recession Signal)
    yield_spread = (df['10Y_Yield'] - df['2Y_Yield']).iloc[-1]
    
    # --- Logic: Systemic Stress Factor (ตัวเดียวกับใน Colab) ---
    stress = (vol * 0.3389 + abs(yield_spread/100) * 0.2450 + coupling * 0.1463 + (kurt/15) * 0.1411)
    
    # ปรับจูน Scale ของ Index (0-100) ให้ไวขึ้นตามค่า Stress
    final_index = np.clip(stress / 0.5 * 100, 2.5, 98.5)
    
    return final_index, stress

def estimate_black_swan_mc(current_stress, horizon_days=30, simulations=50000):
    baseline_daily_prob = 1 / 5000 
    threshold = 0.0549
    # Power Law Tuning
    if current_stress > threshold:
        risk_factor = np.power(current_stress / threshold, 1.15)
    else:
        risk_factor = current_stress / threshold
        
    adj_daily_prob = baseline_daily_prob * risk_factor
    draws = np.random.random((simulations, horizon_days))
    success_count = np.any(draws < adj_daily_prob, axis=1).sum()
    return (success_count / simulations) * 100

# --- 3. EXECUTION ---
try:
    data = get_market_data()
    risk_index, stress_today = calculate_metrics(data)
    
    # พยากรณ์ล่วงหน้าด้วย Mean Reversion
    monthly_trend = 0.015
    p_today = estimate_black_swan_mc(stress_today)
    p_3m = estimate_black_swan_mc(stress_today + (monthly_trend * 3 * 0.85))
    p_6m = estimate_black_swan_mc(stress_today + (monthly_trend * 6 * 0.70))
    
except Exception as e:
    st.error(f"⚠️ การดึงข้อมูลล้มเหลว: {e}")
    st.stop()

# --- 4. UI DISPLAY ---
st.markdown('<div class="main-title">🦢 Prediction of Black Swan Event</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">วิเคราะห์ความเปราะบางเชิงระบบและพยากรณ์โอกาสเกิดวิกฤตการณ์ทางการเงิน</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

# --- ฝั่งซ้าย: GLOBAL SYSTEMIC RISK GAUGE ---
with col_left:
    st.subheader("📊 Global Systemic Risk Index")
    
    if risk_index >= 70:
        status, color, desc = "CRITICAL", "#EF4444", "ระบบมีความเปราะบางสูงมาก เสี่ยงต่อการเกิดภาวะ Black Swan"
    elif risk_index >= 35:
        status, color, desc = "ELEVATED", "#F59E0B", "ระบบมีความเครียดสะสมเหนือระดับปกติ ควรเพิ่มความระมัดระวัง"
    else:
        status, color, desc = "NORMAL", "#10B981", "สภาวะตลาดโลกมีความยืดหยุ่นสูง ความเสี่ยงเชิงระบบอยู่ในเกณฑ์ต่ำ"

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = risk_index,
        number = {'font': {'size': 80, 'color': '#1E293B'}, 'valueformat': '.1f'},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1E293B"},
            'steps': [
                {'range': [0, 35], 'color': '#BBF7D0'},
                {'range': [35, 70], 'color': '#FEF08A'},
                {'range': [70, 100], 'color': '#FECACA'}
            ],
            'threshold': {'line': {'color': "#EF4444", 'width': 5}, 'thickness': 0.8, 'value': 90}
        }
    ))
    fig_gauge.update_layout(height=400, margin=dict(t=20, b=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown(f"""
        <div class="status-box" style="border-top: 5px solid {color};">
            <h2 style="color: {color}; margin: 0; font-weight: 600;">{status}</h2>
            <p style="color: #64748B; margin-top: 8px; font-size: 1.1rem;">{desc}</p>
        </div>
    """, unsafe_allow_html=True)

# --- ฝั่งขวา: PROBABILITY FORECAST ---
with col_right:
    st.subheader("🔮 Probability Forecast")
    st.markdown(f"**Current Systemic Stress:** `{stress_today:.4f}`")
    
    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Today", f"{p_today:.2f}%")
    m2.metric("3M Forward", f"{p_3m:.2f}%", f"{p_3m-p_today:+.2f}%")
    m3.metric("6M Forward", f"{p_6m:.2f}%", f"{p_6m-p_today:+.2f}%")
    
    # Path Chart
    df_path = pd.DataFrame({
        "Timeline": ["Today", "3M Forward", "6M Forward"],
        "Prob": [p_today, p_3m, p_6m]
    })
    
    fig_path = go.Figure()
    fig_path.add_trace(go.Scatter(
        x=df_path["Timeline"], y=df_path["Prob"],
        mode='lines+markers+text',
        text=[f"{p_today:.1f}%", f"{p_3m:.1f}%", f"{p_6m:.1f}%"],
        textposition="top center",
        line=dict(color=color, width=4),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1)'
    ))
    fig_path.update_layout(
        height=320, margin=dict(t=30, b=0, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='#E2E8F0', title="Probability (%)", range=[0, max(p_6m+5, 15)])
    )
    st.plotly_chart(fig_path, use_container_width=True)
    
    with st.expander("ℹ️ Methodology Insight"):
        st.caption("""
        - **Daily Baseline:** อิงจากสถิติเหตุการณ์หายากระดับโลก (1 ใน 5,000 วันทำการ)
        - **Monte Carlo Engine:** จำลองเหตุการณ์ตลาดอนาคต 50,000 รูปแบบในแต่ละจุดเวลา
        - **Power Law Scaling:** ปรับระดับความเสี่ยงตามความเครียดเชิงระบบ (Systemic Stress) แบบ Non-linear
        - **Mean Reversion:** การพยากรณ์ระยะยาวรวมสมมติฐานการปรับตัวของกลไกตลาดและนโยบายภาครัฐ
        """)

st.divider()
st.caption("Data source: Yahoo Finance | Framework: Antifragile Quantitative Risk Model")


##Section 3 : Simulation
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Black Swan Predictor & Simulator", layout="wide")

def apply_dynamic_style(prob, is_sim=False):
    # กำหนดสีและเอฟเฟกต์ตามระดับความเสี่ยง
    if prob < 5:
        bg_color, status_color, status_text = "#ECFDF5", "#10B981", "NORMAL (Safe Haven)"
    elif prob < 15:
        bg_color, status_color, status_text = "#FFF7ED", "#F59E0B", "ELEVATED (Anxious)"
    else:
        bg_color, status_color, status_text = "#450A0A" if is_sim else "#FEF2F2", "#EF4444", "CRITICAL (Black Swan!)"
    
    shake_class = "shake" if prob >= 5 else ""
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Anuphan', sans-serif;
            transition: background-color 0.8s ease;
        }}
        .metric-card {{ background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .shake {{ animation: shake 0.5s infinite; }}
        @keyframes shake {{
            0% {{ transform: translate(1px, 1px) rotate(0deg); }}
            10% {{ transform: translate(-1px, -2px) rotate(-1deg); }}
            20% {{ transform: translate(-3px, 0px) rotate(1deg); }}
            100% {{ transform: translate(1px, -2px) rotate(-1deg); }}
        }}
        .duck-icon {{ font-size: 100px; text-align: center; display: block; }}
        </style>
        """, unsafe_allow_html=True)
    return bg_color, status_color, status_text, shake_class

# --- 2. BACKEND ENGINES ---
@st.cache_data(ttl=3600)
def get_realtime_data():
    tickers = {"VIX": "^VIX", "10Y": "^TNX", "2Y": "^IRX", "Gold": "GC=F", "Copper": "HG=F", "SP500": "^GSPC"}
    df = yf.download(list(tickers.values()), period="2y")['Close']
    df = df.ffill().bfill()
    
    # คำนวณค่าจริงวันนี้
    vol = df['^VIX'].iloc[-1] / 100 # Normalize VIX
    yield_spread = (df['^TNX'] - df['^IRX']).iloc[-1]
    returns = df['^GSPC'].pct_change().dropna()
    kurt = returns.rolling(252).kurt().iloc[-1]
    
    # สำหรับ Coupling (สมมติค่าจากความแกว่งตัวเฉลี่ยของภูมิภาค)
    coupling = 0.45 # ค่าเฉลี่ยตลาดปกติ
    gold_copper = (df['GC=F'] / df['HG=F']).iloc[-1]
    
    return vol, yield_spread, coupling, kurt, gold_copper

def estimate_black_swan_mc(stress, horizon_days=30, simulations=50000):
    baseline_daily_prob = 1 / 5000
    threshold = 0.0549
    risk_factor = np.power(stress / threshold, 1.15) if stress > threshold else stress / threshold
    draws = np.random.random((simulations, horizon_days))
    return (np.any(draws < (baseline_daily_prob * risk_factor), axis=1).sum() / simulations) * 100

def get_stress_score(v, y, c, k, g):
    return (v * 0.3389 + abs(y/100) * 0.2450 + c * 0.1463 + (k/15) * 0.1411)

# --- 3. EXECUTION: FETCH REAL DATA ---
v_real, y_real, c_real, k_real, g_real = get_realtime_data()
stress_real = get_stress_score(v_real, y_real, c_real, k_real, g_real)
p_real = estimate_black_swan_mc(stress_real)

# --- 4. SECTION 1: PREDICTION OF BLACK SWAN EVENT (READ-ONLY) ---
st.markdown("<h1 style='text-align: center;'>🦢 Prediction of Black Swan Event</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>ข้อมูลจริงจากตลาดโลก (Real-time Market Data Analysis)</p>", unsafe_allow_html=True)

col_r1, col_r2 = st.columns([1, 2])
with col_r1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = p_real,
        title = {'text': "Current Risk (%)"},
        gauge = {'axis': {'range': [0, 25]}, 'bar': {'color': "#1E293B"},
                 'steps': [{'range': [0, 5], 'color': "#BBF7D0"}, {'range': [5, 15], 'color': "#FEF08A"}, {'range': [15, 25], 'color': "#FECACA"}]}
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=50, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_r2:
    st.write("### Market Status Today")
    st.write(f"- **Systemic Stress Factor:** `{stress_real:.4f}`")
    mr1, mr2, mr3 = st.columns(3)
    mr1.metric("Today Risk", f"{p_real:.2f}%")
    mr2.metric("3M Forecast", f"{estimate_black_swan_mc(stress_real + 0.012):.2f}%")
    mr3.metric("6M Forecast", f"{estimate_black_swan_mc(stress_real + 0.025):.2f}%")
    st.info("ส่วนนี้ใช้ข้อมูลจาก Yahoo Finance แบบ Real-time ไม่สามารถปรับแต่งได้")

st.divider()

# --- 5. SECTION 2: SIMULATION PROBABILITY (SANDBOX) ---
st.markdown("<h1 style='text-align: center;'>🎮 Simulation Probability</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>ปรับแต่งปัจจัยเพื่อจำลองเหตุการณ์ Black Swan ในแบบของคุณ</p>", unsafe_allow_html=True)

# ปรับค่าเริ่มต้น (Initial Value) ให้เท่ากับค่า Real-time
with st.container():
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        s_vol = st.slider("Market Panic (Volatility)", 0.05, 1.0, float(v_real))
        s_kurt = st.slider("Fat-Tail Frequency (Kurtosis)", 0.0, 30.0, float(k_real))
    with col_s2:
        s_yield = st.slider("Recession Inversion (Yield Spread)", -3.0, 3.0, float(y_real))
        s_gold = st.slider("Safe Haven Ratio (Gold/Copper)", 200.0, 1000.0, float(g_real))
    with col_s3:
        s_coupling = st.slider("Global Coupling", 0.0, 1.0, float(c_real))
        # ตัวแปรเซอร์ไพรส์
        st.write("**The Butterfly Effect**")
        butterfly = st.checkbox("🦋 Trigger Unforeseen Event", help="สุ่มเหตุการณ์ไม่คาดคิดที่อาจขยายความเสี่ยงทันที")
        chaos_mult = np.random.uniform(1.3, 3.0) if butterfly else 1.0

# Calculation for Simulation
stress_sim = get_stress_score(s_vol, s_yield, s_coupling, s_kurt, s_gold) * chaos_mult
p_sim_today = estimate_black_swan_mc(stress_sim)
p_sim_3m = estimate_black_swan_mc(stress_sim + 0.015)
p_sim_6m = estimate_black_swan_mc(stress_sim + 0.030)

# Apply Visual Logic
bg_sim, color_sim, status_sim, shake_sim = apply_dynamic_style(p_sim_today, is_sim=True)

# Display Simulation Result
st.markdown(f"<div style='background-color:{bg_sim}; padding:30px; border-radius:20px; border: 2px solid {color_sim};'>", unsafe_allow_html=True)
sc1, sc2 = st.columns([1, 2])

with sc1:
    st.markdown(f"<div class='{shake_sim}'>", unsafe_allow_html=True)
    if p_sim_today < 5:
        st.markdown("<span class='duck-icon'>🦆</span>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center; color:{color_sim};'>Happy Duck</h3>", unsafe_allow_html=True)
    elif p_sim_today < 15:
        st.markdown("<span class='duck-icon'>🐥</span>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center; color:{color_sim};'>Anxious Duck</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='duck-icon'>🦢</span>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center; color:white;'>THE BLACK SWAN RAGE!</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with sc2:
    if butterfly: st.warning(f"🦋 Butterfly Effect Active: Chaos Multiplier x{chaos_mult:.2f}")
    st.markdown(f"<h2 style='color:{color_sim if p_sim_today < 15 else 'white'};'>Simulation Risk: {p_sim_today:.2f}%</h2>", unsafe_allow_html=True)
    
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Sim Today", f"{p_sim_today:.2f}%", delta=f"{p_sim_today-p_real:+.2f}%")
    sm2.metric("Sim 3M", f"{p_sim_3m:.2f}%")
    sm3.metric("Sim 6M", f"{p_sim_6m:.2f}%")
    
    # Chart
    fig_path = go.Figure(go.Scatter(x=["Today", "3M", "6M"], y=[p_sim_today, p_sim_3m, p_sim_6m], 
                                   fill='tozeroy', line=dict(color=color_sim, width=4)))
    fig_path.update_layout(height=250, margin=dict(t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_path, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
