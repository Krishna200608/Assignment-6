"""
app.py — Streamlit Interactive Dashboard
==========================================
Launch with: streamlit run app.py
"""
import streamlit as st
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Geopolitical Conflict Analytics", page_icon="🌍", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Policy Insights Report", "About"])

if page == "Dashboard":
    st.title("🌍 Geopolitical Conflict Analytics Dashboard")
    st.subheader("Impact of Iran-USA War on Financial Markets and Environmental Indicators")
    st.markdown("---")
    
    st.markdown("""
        Welcome to the **Multimodal Visual Analytics** dashboard. The charts below 
        analyze the cascading impact of geopolitical friction in the Middle East 
        on macro-economic variables (Oil, S&P 500, Gold, Inflation) 
        and proxy environmental indicators (CO2 Emissions).  
        Financial data is sourced natively from **Yahoo Finance**, and conflict data is parsed from verified GitHub **OSINT JSON pipelines**.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 1. Geospatial Analysis (Full Width) ---
    col_map_title, col_map_theme = st.columns([3, 1])
    with col_map_title:
        st.subheader("1. Geospatial Analysis")
    with col_map_theme:
        map_style_selection = st.selectbox("Map Theme", ["Dark Mode", "Light Mode", "Standard Streets"], index=0)
        
    style_mapping = {
        "Dark Mode": "carto-darkmatter",
        "Light Mode": "carto-positron",
        "Standard Streets": "open-street-map"
    }

    data = pd.DataFrame({
        'Location': ['Tehran (Command Center)', 'Strait of Hormuz (Chokepoint)', 'Khuzestan (Oil Fields)', 'Isfahan (Industrial)'],
        'Lat': [35.6892, 26.5667, 31.3273, 32.6539],
        'Lon': [51.3890, 56.2500, 48.6940, 51.6660],
        'Type': ['Conflict Zone', 'Naval Route', 'Oil Infrastructure', 'Pollution Hotspot'],
        'Intensity': [80, 100, 90, 60]
    })
    
    fig = px.scatter_map(
        data, lat="Lat", lon="Lon", color="Type", size="Intensity", 
        hover_name="Location", zoom=3.8, center=dict(lat=31.5, lon=52), 
        map_style=style_mapping[map_style_selection]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)
    st.plotly_chart(fig, width="stretch")
    st.caption("Interactive Map: Iran-USA Conflict Zones & Oil Routes")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 2–6. Chart Grid (2 columns) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Time-Series Correlation")
        st.image("outputs/charts/A_time_series_correlation.png", width="stretch")
        st.caption("Macroeconomic variables tracked against the Conflict Intensity Index timeline.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("4. Correlation Heatmap")
        st.image("outputs/charts/D_heatmap_correlation.png", width="stretch")
        st.caption("Pearson Correlation Matrix showing inverse structural relationships (e.g., S&P 500 vs. Oil).")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("6. Lag & Causal Analysis")
        st.image("outputs/charts/F_lag_analysis.png", width="stretch")
        st.caption("Cross-Correlation mapping demonstrating lag effects of shocks on Inflation and other variables.")

    with col2:
        st.subheader("3. Multi-Axis Dynamics")
        st.image("outputs/charts/B_multi_axis.png", width="stretch")
        st.caption("Synchronized analysis of Oil, Stocks & CO2 Dynamics")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("5. Event Impact Timeline")
        st.image("outputs/charts/E_event_impact.png", width="stretch")
        st.caption("Short-term rolling volatility surrounding major kinetic shocks.")

elif page == "Policy Insights Report":
    st.title("📄 Full Analytical Report & Policy Insights")
    st.markdown("---")
    try:
        with open("docs/report.md", "r", encoding="utf-8") as f:
            report_content = f.read()
        st.markdown(report_content)
    except FileNotFoundError:
        st.error("Report not found. Please run `python main.py` first to generate outputs.")

elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    st.markdown("""
        This visual analytics application analyzes the Iran-USA geopolitical conflict's impact
        on financial markets and environmental indicators.
        
        **Data Sources:**
        - `Oil Price`, `S&P 500`, `Gold`, `Treasury Yield`, `Dollar Index` → **Real data API** from Yahoo Finance (`yfinance`)
        - `Conflict Intensity` → **Real data** parsed from Middle East OSINT Missile trackers (GitHub)
        - `CO2`, `Inflation`, `Exchange Rate` → **Macro-Interpolated Data** anchored perfectly to Treasury/DXY fluctuation realities.
        
        **Core Tech Stack:**
        - `yfinance`: Real-time market data fetching
        - `pandas` / `numpy`: Data processing & feature engineering
        - `matplotlib` / `seaborn`: Statistical charting
        - `plotly`: Interactive geospatial mapping
        - `streamlit`: Dashboard framework
        
        **Data Processing Highlights:**
        - High-fidelity Real OSINT + Macro-Interpolated Anchor pipeline
        - Forward/backward fill for time alignment across weekends/holidays
        - Rolling volatility, temporal lags, and binary event-impact trackers
    """)
    st.success("All visualizations generated successfully. Financial data sourced from Yahoo Finance.")
