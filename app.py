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

# --- 1. CONFIG & THEME ---
st.set_page_config(page_title="Swan Predictor", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #E3F2FD !important; font-family: 'Anuphan', sans-serif; }
    h1, h2, h3 { color: #01579B; font-family: 'Anuphan', sans-serif; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 Global Systemic Risk Index 🖤")
st.markdown("<center><i>ระบบวิเคราะห์ความเปราะบางของโครงสร้างการเงินโลกล่วงหน้า</i></center>", unsafe_allow_html=True)

# --- 2. BACKEND: DATA FETCHING & ANALYSIS ---
@st.cache_data(ttl=3600)
def fetch_and_analyze_fragility():
    tickers = {
        "NSE_India": "^NSEI", "NYSE": "^NYA", "SSE": "000001.SS",
        "JPX": "^N225", "Euronext": "^N100", "LSE": "^FTSE",
        "VIX": "^VIX", "Gold": "GC=F", "Crude_Oil": "BZ=F",
        "Copper": "HG=F", "USD_Index": "DX-Y.NYB", "SP500": "^GSPC",
        "10Y_Yield": "^TNX", "2Y_Yield": "^IRX"
    }
    
    # 2.1 ดึงข้อมูล
    df = yf.download(list(tickers.values()), start="1975-01-01")['Close']
    df = df.ffill().bfill().rename(columns={v: k for k, v in tickers.items()})
    
    # 2.2 คำนวณพื้นฐาน
    df['Gold_Copper_Ratio'] = df['Gold'] / df['Copper']
    df['Yield_Curve_Spread'] = df['10Y_Yield'] - df['2Y_Yield']

    # 2.3 คำนวณความเปราะบาง (Fragility)
    frag_df = pd.DataFrame(index=df.index)
    markets = ['NYSE', 'SP500', 'Euronext', 'LSE', 'JPX', 'SSE', 'NSE_India']
    returns = df[markets].pct_change().dropna()

    # Pillar 1 & 2: Kurtosis & Volatility
    frag_df['Kurtosis'] = returns.rolling(252).kurt().mean(axis=1)
    frag_df['Volatility'] = returns.rolling(252).std().mean(axis=1) * np.sqrt(252)
    
    # Pillar 3: Coupling (ความสัมพันธ์ข้ามตลาด) - แก้ไขวิธีคำนวณที่นี่
    def get_avg_corr(window_data):
        corr_matrix = window_data.corr()
        # ดึงเฉพาะค่า Correlation ระหว่างคู่ (ไม่เอาค่า 1.0 แนวทแยง)
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        return corr_matrix.where(mask).stack().mean()

    # ใช้ List Comprehension เพื่อความเร็วและเลี่ยง Error positional argument
    coupling_list = []
    window = 60
    for i in range(len(returns)):
        if i < window:
            coupling_list.append(np.nan)
        else:
            subset = returns.iloc[i-window:i]
            coupling_list.append(get_avg_corr(subset))
    
    frag_df['Coupling'] = coupling_list
    
    # Pillar 4: Macro Signals
    # เชื่อมข้อมูล Yield และ Gold กลับเข้าไปตาม Index ของ frag_df
    frag_df = frag_df.join(df[['Yield_Curve_Spread', 'Gold_Copper_Ratio']])
    frag_df = frag_df.rename(columns={'Gold_Copper_Ratio': 'Safe_Haven', 'Yield_Curve_Spread': 'Yield_Signal'})

    return frag_df.dropna()

# รันระบบหลังบ้าน
try:
    df_fragility = fetch_and_analyze_fragility()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
    st.stop()

# --- 3. RISK CALCULATION ENGINE ---
def calculate_final_score(row):
    # ปรับจูนน้ำหนักตามหลักความเปราะบาง
    k_score = np.clip((row['Kurtosis'] / 10), 0, 1) * 35 
    v_score = np.clip((row['Volatility'] / 0.4), 0, 1) * 25
    c_score = np.clip((row['Coupling'] / 0.8), 0, 1) * 25
    # Macro Signal (Yield Curve Inversion ให้ 15 คะแนนทันทีถ้าติดลบ)
    m_score = (15 if row['Yield_Signal'] < 0 else 0) + (row['Safe_Haven'] / 800 * 10)
    
    return min(max(k_score + v_score + c_score + m_score, 1.5), 99.5)

latest = df_fragility.iloc[-1]
risk_today = calculate_final_score(latest)

# พยากรณ์ล่วงหน้าด้วยความเร่งของดัชนี (1 เดือนที่ผ่านมา)
diff = (df_fragility.iloc[-1] - df_fragility.iloc[-22])
risk_3m = min(max(risk_today + (calculate_final_score(diff) if risk_today > 10 else 0) * 0.3, 1.5), 99.0)
# สำหรับพยากรณ์ 3-6 เดือน เราใช้ Trend แบบเรียบง่ายเพื่อให้เห็นทิศทาง
risk_3m = min(max(risk_today + (risk_today * 0.05 if latest['Coupling'] > 0.6 else -2), 1.5), 99.0)
risk_6m = min(max(risk_3m + (risk_3m * 0.08 if latest['Yield_Signal'] < 0 else -1), 1.5), 99.0)

# --- 4. DISPLAY: GAUGE CHART ---
st.markdown("<br>", unsafe_allow_html=True)
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = risk_today,
    title = {'text': "Current Systemic Risk Level", 'font': {'size': 24}},
    gauge = {
        'axis': {'range': [0, 100]},
        'bar': {'color': "#01579B"},
        'steps': [
            {'range': [0, 40], 'color': '#A5D6A7'},
            {'range': [40, 70], 'color': '#FFF59D'},
            {'range': [70, 100], 'color': '#EF9A9A'}],
        'threshold': {'line': {'color': "red", 'width': 4}, 'value': 90}}
))
fig_gauge.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font={'family': "Anuphan"})
st.plotly_chart(fig_gauge, use_container_width=True)

# --- 5. PREDICTION HORIZON ---
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.metric("3-Month Forward Outlook 🦆", f"{risk_3m:.2f}%", f"{risk_3m-risk_today:+.2f}%")
with c2:
    st.metric("6-Month Forward Outlook 🖤", f"{risk_6m:.2f}%", f"{risk_6m-risk_today:+.2f}%")

# --- 6. SUMMARY ---
st.markdown("<br>", unsafe_allow_html=True)
if risk_today > 70:
    st.error("🚨 **SYSTEMIC ALERT:** ระบบอยู่ในสภาวะเปราะบางสูงสุด (Fragile) โอกาสเกิดเหตุการณ์รุนแรงมีสูง")
elif risk_today > 40:
    st.warning("⚠️ **RISK WARNING:** ระบบเริ่มมีความเปราะบางสะสม ความสัมพันธ์ระหว่างสินทรัพย์เริ่มสูงขึ้น")
else:
    st.success("🦢 **ANTIFRAGILE STATUS:** ระบบมีความยืดหยุ่นสูง ความเสี่ยงเชิงระบบอยู่ในเกณฑ์ต่ำ")

st.info("Backend Analysis: ประมวลผลผ่านดัชนี Kurtosis (Fat-tail), Volatility, Market Coupling และ Macro Yield Spreads เรียบร้อยแล้ว")
