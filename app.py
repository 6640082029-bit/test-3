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

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# --- 1. CONFIG & CUTE THEME ---
st.set_page_config(page_title="Swan Predictor: Systemic Risk", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #E3F2FD !important; font-family: 'Anuphan', sans-serif; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #BBDEFB; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #01579B; font-family: 'Anuphan', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 Black Swan Predictor 🖤")
st.markdown("### *Systemic Fragility Intelligence: Looking into the Unknown*")

# --- 2. DATA FETCHING (Since 1975) ---
@st.cache_data(ttl=3600)
def fetch_and_analyze_fragility():
    tickers = {
        "NSE_India": "^NSEI", "NYSE": "^NYA", "SSE": "000001.SS",
        "JPX": "^N225", "Euronext": "^N100", "LSE": "^FTSE",
        "VIX": "^VIX", "Gold": "GC=F", "Crude_Oil": "BZ=F",
        "Copper": "HG=F", "USD_Index": "DX-Y.NYB", "SP500": "^GSPC"
    }
    
    # 2.1 Fetch from Yahoo Finance
    df_yf = yf.download(list(tickers.values()), start="1975-01-01")['Close']
    df_yf = df_yf.ffill().bfill().rename(columns={v: k for k, v in tickers.items()})
    df_yf['Gold_Copper_Ratio'] = df_yf['Gold'] / df_yf['Copper']

    # 2.2 Fetch from FRED (Macro Data)
    try:
        df_macro = web.DataReader(['T10Y2Y', 'FEDFUNDS'], 'fred', "1975-01-01")
        df_macro.columns = ['Yield_Curve_Spread', 'FED_Rate']
    except:
        df_macro = pd.DataFrame(index=df_yf.index) # Fallback

    df = pd.concat([df_yf, df_macro], axis=1).ffill()
    
    # 2.3 Calculate 4 Pillars of Fragility
    fragility_df = pd.DataFrame(index=df.index)
    markets = ['NYSE', 'SP500', 'Euronext', 'LSE', 'JPX', 'SSE', 'NSE_India']
    returns = df[markets].pct_change()

    # Pillar 1: Global Kurtosis (Fat-Tail)
    fragility_df['Kurtosis'] = returns.rolling(252).kurt().mean(axis=1)
    # Pillar 2: Global Volatility
    fragility_df['Volatility'] = returns.rolling(252).std().mean(axis=1) * np.sqrt(252)
    # Pillar 3: Network Coupling (60-day corr)
    fragility_df['Coupling'] = returns.rolling(60).corr().groupby(level=0).apply(
        lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean()
    )
    # Pillar 4: Macro Signals
    fragility_df['Yield_Signal'] = df['Yield_Curve_Spread']
    fragility_df['Safe_Haven'] = df['Gold_Copper_Ratio']

    return df, fragility_df.dropna()

df_input, df_fragility = fetch_and_analyze_fragility()

# --- 3. BLACK SWAN PREDICTION LOGIC ---
def predict_risk(current_data):
    # ตรรกะการให้คะแนนความเสี่ยง (0-100)
    # 1. Kurtosis > 3 คือเริ่มอันตราย
    k_score = np.clip((current_data['Kurtosis'] / 10), 0, 1) * 35 
    # 2. Volatility > 20% (0.2)
    v_score = np.clip((current_data['Volatility'] / 0.4), 0, 1) * 25
    # 3. Coupling > 0.7 คือไม่มีที่หนี
    c_score = np.clip((current_data['Coupling'] / 0.8), 0, 1) * 25
    # 4. Macro (Yield Inversion & Safe Haven)
    m_score = (15 if current_data['Yield_Signal'] < 0 else 0) + (current_data['Safe_Haven'] / 800 * 10)
    
    total = k_score + v_score + c_score + m_score
    return min(max(total, 1.5), 99.5)

latest_fragility = df_fragility.iloc[-1]
risk_today = predict_risk(latest_fragility)

# พยากรณ์ล่วงหน้า (ใช้ความเร่งหรือ Acceleration ของ Fragility)
velocity = (df_fragility.iloc[-1] - df_fragility.iloc[-22]).mean() # 1 month trend
risk_3m = min(max(risk_today + (velocity * 3), 1.5), 99.0)
risk_6m = min(max(risk_today + (velocity * 6), 1.5), 99.0)

# --- 4. DISPLAY SECTION ---
st.header("🔮 Black Swan Risk Horizon")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Today's Fragility 🦢", f"{risk_today:.2f}%", delta="Current Status", delta_color="inverse")
with c2:
    st.metric("3-Month Predictor 🦆", f"{risk_3m:.2f}%", delta=f"{risk_3m-risk_today:.1f}% vs Today")
with c3:
    st.metric("6-Month Predictor 🖤", f"{risk_6m:.2f}%", delta=f"{risk_6m-risk_today:.1f}% vs Today")

# --- 5. THE 4 PILLARS VISUALIZATION ---
st.divider()
st.subheader("🕵️ The 4 Pillars of Systemic Fragility")
p1, p2, p3, p4 = st.columns(4)

p1.metric("Kurtosis (Fat-Tail)", f"{latest_fragility['Kurtosis']:.2f}")
p2.metric("Volatility (Avg)", f"{latest_fragility['Volatility']*100:.1f}%")
p3.metric("Coupling (Corr)", f"{latest_fragility['Coupling']:.2f}")
p4.metric("Yield Spread", f"{latest_fragility['Yield_Signal']:.2f}")

# --- 6. HISTORICAL FRAGILITY CHART ---
st.subheader("📈 Historical Systemic Fragility Index")
fig = go.Figure()
# คำนวณ Index รวมเพื่อพล็อตกราฟ
df_fragility['Fragility_Index'] = df_fragility.apply(predict_risk, axis=1)

fig.add_trace(go.Scatter(x=df_fragility.index, y=df_fragility['Fragility_Index'], 
                         name="Systemic Risk Score", line=dict(color='#01579B', width=2)))

# เพิ่ม Highlight ช่วงวิกฤตสำคัญ
crises = {
    '1987-10-19': 'Black Monday',
    '2008-09-15': 'Lehman Crisis',
    '2020-03-15': 'COVID-19'
}
for date, name in crises.items():
    if pd.to_datetime(date) in df_fragility.index:
        fig.add_annotation(x=date, y=df_fragility.loc[date, 'Fragility_Index'], text=name, showarrow=True)

fig.update_layout(template="plotly_white", height=500, yaxis_title="Risk Score (0-100)", 
                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

# --- 7. AI INTERPRETATION ---
st.divider()
if risk_today > 60:
    st.error("🚨 **Systemic Alert:** ความเปราะบางของระบบสูงเกินเกณฑ์ ตลาดมีความสัมพันธ์กันสูง (Coupling) และเกิด Fat-Tail Risk ชัดเจน")
elif risk_today > 40:
    st.warning("⚠️ **Fragility Warning:** ระบบเริ่มเสียสมดุล Diversification อาจทำงานได้ไม่เต็มที่")
else:
    st.success("🦢 **Antifragile Status:** ระบบยังมีความยืดหยุ่นสูง ความเสี่ยงเชิงระบบอยู่ในระดับต่ำ")
