import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SET PAGE & THEME ---
st.set_page_config(page_title="Swan Odyssey: Max History", layout="wide")

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
st.markdown("### *Max Historical Analysis & Real-time Global Assets* 🐥")

# --- 2. DATA FETCHING (Max Period) ---
@st.cache_data(ttl=3600)
def fetch_max_global_data():
    tickers = {
        'NSE_India': '^NSEI', 'NYSE': '^NYA', 'SSE': '000001.SS', 
        'JPX': '^N225', 'Euronext': '^N100', 'LSE': '^FTSE',
        'VIX': '^VIX', 'Gold': 'GC=F', 'Crude_Oil': 'BZ=F',
        'Copper': 'HG=F', 'USD_Index': 'DX-Y.NYB',
        '10Y_Bond': '^TNX', '2Y_Bond': '^IRX'
    }
    
    # ดึงข้อมูลแบบ MAX เพื่อดูตั้งแต่จุดเริ่มต้น
    df = yf.download(list(tickers.values()), period="max")['Close']
    
    if df.empty:
        return None, None, None, None

    # จัดการข้อมูล: ffill เพื่อเติมช่องว่าง, bfill เพื่อให้แถวแรกมีค่า
    df = df.ffill().bfill()
    
    # เปลี่ยนชื่อให้เรียกง่าย
    inv_map = {v: k for k, v in tickers.items()}
    df = df.rename(columns=inv_map)
    
    # ตัดข้อมูลให้เริ่ม ณ วันที่ทุกดัชนีมีข้อมูลครบ (เพื่อให้จุดเริ่ม Normalized เป็น 100 พร้อมกัน)
    df_clean = df.dropna()
    
    latest = df_clean.iloc[-1]
    prev = df_clean.iloc[-2]
    
    # Normalization สำหรับกราฟ (Base 100)
    price_cols = ['NSE_India', 'NYSE', 'SSE', 'JPX', 'Euronext', 'LSE', 'Gold', 'Crude_Oil', 'USD_Index']
    valid_cols = [c for c in price_cols if c in df_clean.columns]
    df_norm = (df_clean[valid_cols] / df_clean[valid_cols].iloc[0]) * 100
    
    return df_clean, df_norm, latest, prev

df_raw, df_norm, latest, prev = fetch_max_global_data()

if df_raw is not None:
    # --- 3. SECTION I: LIVE WATCHTOWER (พร้อมหน่วย) ---
    st.header("🔭 Section I: Live Watchtower")
    
    # ตลาดหุ้น 6 ประเทศ
    st.subheader("🌎 Global Equity Indices")
    m_cols = st.columns(6)
    market_info = {
        'NSE_India': 'INR (Nifty 50)', 'NYSE': 'USD (Composite)', 'SSE': 'CNY (Composite)', 
        'JPX': 'JPY (Nikkei 225)', 'Euronext': 'EUR (Enext 100)', 'LSE': 'GBP (FTSE 100)'
    }
    for i, (key, label) in enumerate(market_info.items()):
        change = (latest[key] - prev[key]) / prev[key] * 100
        m_cols[i].metric(label, f"{latest[key]:,.2f}", f"{change:.2f}%")

    # ทรัพย์สินอื่นๆ
    st.subheader("📊 Commodity & Risk Indicators")
    a_cols = st.columns(4)
    a_cols[0].metric("Gold (XAU/USD)", f"${latest['Gold']:,.2f}", "USD / t oz")
    a_cols[1].metric("Brent Crude Oil", f"${latest['Crude_Oil']:.2f}", "USD / bbl")
    a_cols[2].metric("USD Index (DXY)", f"{latest['USD_Index']:.2f}", "Points")
    a_cols[3].metric("VIX Index (Fear)", f"{latest['VIX']:.2f}", f"{latest['VIX']-prev['VIX']:.2f}")

    # --- 4. GRAPH: MAX HISTORY LOG SCALE ---
    st.subheader(f"📈 All-Time Growth Comparison (Since {df_norm.index[0].year} - Normalized)")
    st.info(f"กราฟนี้เริ่มคำนวณจากปี {df_norm.index[0].year} ซึ่งเป็นจุดที่ตลาดทั้ง 6 แห่งมีข้อมูลพร้อมกันวันแรก")

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
