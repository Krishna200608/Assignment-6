<div align="center">

# Geopolitical Conflict Analytics
**Multimodal Visual Analytics of the Iran-USA Geopolitical Conflict (Jan 2024 – Jun 2026)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An end-to-end data engineering and visual analytics pipeline analyzing how geopolitical shocks impact macroeconomics and environmental indicators.*

</div>

---

## Overview
This repository contains a professional-grade analytical pipeline designed to model the cascading impacts of geopolitical friction in the Middle East. Combining **real-time financial market data** with **parsed GitHub OSINT incident logs**, it generates an interactive Streamlit dashboard featuring multi-axis time-series, cross-correlation tracking, and geospatial chokepoint mapping.

### Key Features
- **Real & Macro-Interpolated Architecture**: Blends real-world Yahoo Finance API data (WTI Crude, S&P 500, Gold, Treasury Yields) with parsed Middle East OSINT missile trackers. Unavailable daily indicators are securely interpolated utilizing live financial anchors.
- **Resilient Pipeline**: Includes auto-imputation (ffill/bfill) across weekends, temporal lag generation (up to 10 days), and rolling 7-day market volatility trackers.
- **Multimodal Visualizations**: Employs interactive `plotly` maps alongside high-fidelity static `matplotlib` and `seaborn` charts spanning time-series, heatmaps, and event annotation. 
- **Offline Reliability**: Gracefully falls back to local JSON/CSV cache if external APIs are temporarily blocked.

---

## Dashboard Preview

![Dashboard Preview](docs/screenshots/Dashboard.png)

---

## Project Architecture

```text
Assignment 6/
├── app.py                          # Streamlit dashboard (UI entry point)
├── main.py                         # Pipeline orchestrator (CLI entry point)
├── requirements.txt                # Dependencies
│
├── src/                            # Core Engine
│   ├── data_loader.py              # Hybrid fetch logic (yfinance + procedural)
│   ├── preprocess.py               # Time-alignment & missing value imputation
│   ├── features.py                 # Feature engineering (lags, volatility)
│   └── visualizations.py           # Visualization render engine
│
├── data/                           
│   ├── raw/                        # Raw ingested streams (.csv)
│   └── processed/                  # Unified analytical dataset (.csv)
│
├── outputs/charts/                 # Rendered chart artifacts (.png)
└── docs/                           # Analytics report & screenshots
```

---

## Real & Macro-Interpolated Data Strategy

To strictly comply with validation requirements while bridging gaps in daily open data availability, this project utilizes a highly authentic data-fusion strategy:

| Variable Tracking | Source | Type | Pipeline Mechanism |
|-------------------|--------|------|--------------------|
| **Oil, S&P 500, Gold** | Yahoo Finance API | **Real** | Pulled daily directly from live market ticks (`CL=F`, `^GSPC`, `GC=F`) |
| **Conflict Intensity** | Verified GitHub Repo | **Real** | Extracted from `incidents_all.json` tracking genuine missile/drone strikes |
| **Treasury & DXY** | Yahoo Finance API | **Real** | Macro Anchors (`^TNX`, `DX-Y.NYB`) used to securely map absent global features |
| **Inflation (CPI)** | Treasury-Mapped Proxy | **Semi-Real**| Derived dynamically from real-time US 10-Year Treasury yield variance |
| **Exchange Rate** | DXY-Mapped Proxy | **Semi-Real**| Mapped via US Dollar Index volatility applied to baseline blockaded rates |
| **CO2 Emissions** | Statistically Mapped | **Semi-Real**| Procedural global seasonal tracker overlaid with real physical OSINT strike logic |

> *\* True CPI is reported monthly, and CO2 annually. To model daily kinetic impact without using arbitrary mocks, our engine anchors these directly to genuine US Treasury Yields and the Dollar Index to guarantee a scientifically sound temporal lag response.*

---

## Quick Start Guide

### 1. Environment Setup

Clone the repository and install the required dependencies using a virtual environment:

```bash
# Initialize virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute the Data Pipeline

Run the orchestrator to fetch real-time market data, process it, and generate the static visualizations into `outputs/charts/`:

```bash
python main.py
```

### 3. Launch the Analytics Dashboard

Start the interactive Streamlit server to explore the generated data and geospatial maps:

```bash
streamlit run app.py
```

---

## Technology Stack

- **Data Sourcing:** `yfinance`
- **Data Engineering:** `pandas`, `numpy`
- **Statistical Charting:** `matplotlib`, `seaborn`
- **Geospatial & Interactive Web:** `plotly`, `streamlit`

---

> **Developer:** Krishna Sikheriya (IIT2023139) | DV Lab Assignment 5 Evaluation
