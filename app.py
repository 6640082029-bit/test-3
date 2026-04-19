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
##ทดลอง
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# --- 1. CONFIG ---
st.set_page_config(page_title="Global Systemic Risk Dashboard", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #F8FAFC !important; font-family: 'Anuphan', sans-serif; }
    h1 { color: #0F172A; text-align: center; font-weight: 600; padding-top: 20px; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 Global Systemic Risk Index")
st.markdown("<center style='color: #64748B;'>Systemic Fragility & Black Swan Probability Engine</center>", unsafe_allow_html=True)

# --- 2. BACKEND: DATA & LOGIC ---
@st.cache_data(ttl=3600)
def fetch_systemic_data():
    tickers = {
        "NSE_India": "^NSEI", "NYSE": "^NYA", "SSE": "000001.SS",
        "JPX": "^N225", "Euronext": "^N100", "LSE": "^FTSE",
        "Gold": "GC=F", "Copper": "HG=F", "SP500": "^GSPC",
        "10Y_Yield": "^TNX", "2Y_Yield": "^IRX"
    }
    df = yf.download(list(tickers.values()), start="2000-01-01")['Close']
    df = df.ffill().bfill().rename(columns={v: k for k, v in tickers.items()})
    
    # 4 มิติเบื้องหลัง
    markets = ['NYSE', 'SP500', 'Euronext', 'LSE', 'JPX', 'SSE', 'NSE_India']
    returns = df[markets].pct_change().dropna()

    frag_df = pd.DataFrame(index=returns.index)
    frag_df['Kurtosis'] = returns.rolling(252).kurt().mean(axis=1)
    frag_df['Volatility'] = returns.rolling(252).std().mean(axis=1) * np.sqrt(252)
    
    # Coupling Calculation
    def get_avg_corr(window_data):
        corr_matrix = window_data.corr()
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        return corr_matrix.where(mask).stack().mean()

    coupling_vals = [np.nan]*60
    for i in range(60, len(returns)):
        coupling_vals.append(get_avg_corr(returns.iloc[i-60:i]))
    
    frag_df['Coupling'] = coupling_vals
    frag_df['Yield_Signal'] = (df['10Y_Yield'] - df['2Y_Yield']).reindex(frag_df.index)
    frag_df['Safe_Haven'] = (df['Gold'] / df['Copper']).reindex(frag_df.index)
    
    return frag_df.dropna()

try:
    df_fragility = fetch_systemic_data()
    latest = df_fragility.iloc[-1]
except Exception as e:
    st.error(f"Data Fetch Error: {e}")
    st.stop()

# --- 3. RISK & PROBABILITY CALCULATION ---
# มิติ Index (0-100)
def calc_index(row):
    score = (np.clip(row['Kurtosis']/12, 0, 1)*30 + np.clip(row['Volatility']/0.45, 0, 1)*20 + 
             np.clip(row['Coupling']/0.85, 0, 1)*30 + (20 if row['Yield_Signal'] < 0 else 0))
    return min(max(score, 2.5), 98.5)

risk_today = calc_index(latest)

# มิติ Probability (ML Pattern Recognition)
def calc_probability(row):
    # น้ำหนัก Feature Importance ที่คุณเรียนรู้มา
    stress_score = (row['Volatility'] * 0.3389 + abs(row['Yield_Signal']) * 0.2450 + 
                    row['Coupling'] * 0.1463 + (row['Kurtosis']/15) * 0.1411 + (row['Safe_Haven']/800) * 0.1284)
    # เทียบกับขีดจำกัดความเครียด 0.0549
    prob = 1 / (1 + np.exp(-100 * (stress_score - 0.0549)))
    return prob * 100, stress_score

prob_today, stress_today = calc_probability(latest)

# พยากรณ์ล่วงหน้า (Acceleration)
trend = (df_fragility.iloc[-1] - df_fragility.iloc[-22])
prob_3m, _ = calc_probability(latest + (trend * 1.5))
prob_6m, _ = calc_probability(latest + (trend * 3.0))

# --- 4. VISUAL GAUGE & STATUS ---
if risk_today >= 70:
    status_label, risk_color = "CRITICAL", "#EF4444"
elif risk_today >= 35:
    status_label, risk_color = "ELEVATED", "#F59E0B"
else:
    status_label, risk_color = "NORMAL", "#10B981"

st.markdown("<br>", unsafe_allow_html=True)
fig = go.Figure(go.Indicator(
    mode = "gauge+number", value = risk_today,
    number = {'font': {'size': 80, 'color': '#1E293B'}, 'valueformat': '.1f'},
    gauge = {
        'axis': {'range': [0, 100]}, 'bar': {'color': "#1E293B"},
        'steps': [{'range': [0, 35], 'color': '#BBF7D0'}, {'range': [35, 70], 'color': '#FEF08A'}, {'range': [70, 100], 'color': '#FECACA'}],
        'threshold': {'line': {'color': "red", 'width': 5}, 'value': 90}}
))

fig.add_annotation(x=0.5, y=0.4, text=status_label, showarrow=False, font=dict(size=24, color=risk_color, weight="bold"))
fig.update_layout(height=400, margin=dict(t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

# --- 5. PROBABILITY FORECAST ---
st.divider()
st.subheader("🔮 Black Swan Probability Forecast")
c1, c2, c3 = st.columns(3)
c1.metric("Today's Probability", f"{prob_today:.2f}%")
c2.metric("3-Month Forecast", f"{prob_3m:.2f}%", f"{prob_3m-prob_today:+.2f}%")
c3.metric("6-Month Forecast", f"{prob_6m:.2f}%", f"{prob_6m-prob_today:+.2f}%")

if stress_today > 0.0549:
    st.error(f"🚨 **STRESS ALERT:** Systemic Stress ({stress_today:.4f}) exceeded threshold (0.0549)!")
else:
    st.success(f"🛡️ **SYSTEMIC STABILITY:** Stress Level at {stress_today:.4f} (Under 0.0549)")
