import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# --- 1. SET PAGE & CUTE THEME ---
st.set_page_config(page_title="🦢 Black Swan Odyssey: Global Watch", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #E3F2FD !important;
        font-family: 'Anuphan', sans-serif;
    }
    h1, h2, h3, .stMetric { font-family: 'Anuphan', sans-serif; color: #01579B; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 15px; border: 1px solid #BBDEFB; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 The Black Swan Odyssey 🦆")
st.markdown("### *Global Market Watch: 6 Nations & Economic Indicators* 🌍")

# --- 2. DATA FETCHING (6 Countries + Macro) ---
@st.cache_data(ttl=3600)
def fetch_global_market_data():
    # กำหนด Tickers ตามที่คุณต้องการ
    tickers = {
        'NSE_India': '^NSEI', 
        'NYSE': 'NYA', 
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
        '2Y_Bond': '^IRX' # ใช้ 3M หรือ 2Y ตามความเหมาะสม (ในที่นี้ใช้ T-Bill 13w แทนได้ถ้าจะหา Spread สั้น)
    }
    
    # ดึงข้อมูลย้อนหลัง 1 ปีเพื่อทำ Normalization
    data = yf.download(list(tickers.values()), period="1y")['Close']
    data = data.ffill().dropna()
    
    # เปลี่ยนชื่อ Column ให้ตรงตาม Key ที่เราตั้งไว้
    inv_tickers = {v: k for k, v in tickers.items()}
    data = data.rename(columns=inv_tickers)
    
    # คำนวณค่าเสริม
    data['Gold_Copper_Ratio'] = data['Gold'] / data['Copper']
    data['Yield_Curve_Spread'] = data['10Y_Bond'] - data['2Y_Bond']
    
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    # ทำ Normalization (Base 100) สำหรับคอลัมน์ราคา
    price_cols = ['NSE_India', 'NYSE', 'SSE', 'JPX', 'Euronext', 'LSE', 'Gold', 'Crude_Oil', 'USD_Index']
    df_norm = (data[price_cols] / data[price_cols].iloc[0]) * 100
    
    return data, df_norm, latest, prev

try:
    df_raw, df_norm, latest, prev = fetch_global_market_data()
    st.sidebar.success("✅ ข้อมูลตลาดโลกอัปเดตแล้ว")
except Exception as e:
    st.error(f"ไม่สามารถดึงข้อมูลได้: {e}")
    st.stop()

# --- 3. SECTION I: LIVE WATCHTOWER ---
st.header("🔭 Section I: Live Watchtower")

# คำนวณความเสี่ยงเบื้องต้น (Logic เดิมจาก AI ของคุณ)
WEIGHTS = {'vol': 0.339, 'yield': 0.245, 'coupling': 0.146, 'others': 0.27}
l_prob = ((latest['VIX']/45) * WEIGHTS['vol'] + max(0, 1 - latest['Yield_Curve_Spread']/2) * WEIGHTS['yield'] + 0.15) * 100
l_prob = min(max(l_prob, 1.5), 98.0)

p1, p2, p3 = st.columns(3)
p1.metric("Today's Risk 🦢", f"{l_prob:.2f}%")
p2.metric("6 Months Outlook 🦆", f"{min(l_prob*1.2, 99.0):.2f}%")
p3.metric("12 Months Outlook 🖤", f"{min(l_prob*1.6, 99.0):.2f}%")

# แสดงค่าจาก 6 ประเทศหลัก
st.subheader("🌎 Global Indices & Snapshot")
cols = st.columns(6)
countries = ['NSE_India', 'NYSE', 'SSE', 'JPX', 'Euronext', 'LSE']
for i, country in enumerate(countries):
    change = (latest[country] - prev[country]) / prev[country] * 100
    cols[i].metric(country.replace('_', ' '), f"{latest[country]:,.0f}", f"{change:.2f}%")

# --- 4. กราฟเปรียบเทียบ (NORMALIZED & LOG SCALE) ---
st.subheader("📈 Global Market Comparison (Normalized to 100 & Log Scale)")
st.markdown("*เปรียบเทียบการเติบโตสัมพัทธ์ของสินทรัพย์และดัชนี 6 ประเทศ*")

fig = go.Figure()
for col in df_norm.columns:
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], name=col, mode='lines', line=dict(width=2)))

fig.update_layout(
    yaxis_type="log", # ใช้ Log Scale ตามที่คุณต้องการ
    xaxis_title="Timeline (Last 1 Year)",
    yaxis_title="Growth from Base 100 (Log Scale)",
    hovermode="x unified",
    template="plotly_white",
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

