# Project Context — Geopolitical Conflict Analytics

> **Purpose of this file:** Provide a complete, self-contained reference so any AI model or new developer can understand the entire project without reading every source file. This is the single source of truth for architecture, data strategy, file roles, known quirks, and run instructions.

---

## 1. Project Identity

| Field | Value |
|-------|-------|
| **Title** | Multimodal Visual Analytics of Geopolitical Conflict |
| **Student** | Krishna Sikheriya (IIT2023139) |
| **Course** | Data Visualization Lab — Assignment 5 |
| **Timeline Modelled** | January 2023 – June 2024 |
| **GitHub Repo** | https://github.com/Krishna200608/Assignment-6.git |
| **Branch** | `main` |

---

## 2. Directory Structure

```
Assignment 6/                       ← Project root
├── app.py                          # Streamlit dashboard (UI entry point)
├── main.py                         # Pipeline orchestrator (CLI entry point)
├── requirements.txt                # pip dependencies
├── .gitignore                      # Excludes .venv, Course/, study.md, *.zip, __pycache__
│
├── src/                            # Core pipeline modules
│   ├── __init__.py                 # Package marker
│   ├── data_loader.py              # Hybrid data fetcher (Yahoo Finance + synthetic)
│   ├── preprocess.py               # Time alignment & missing value imputation
│   ├── features.py                 # Feature engineering (lags, volatility, shocks)
│   └── visualizations.py           # Chart generation engine (6 charts → outputs/charts/)
│
├── data/
│   ├── raw/                        # Raw ingested data
│   │   └── raw_data.csv            # 547 rows, 9 columns, daily from 2023-01-01 to 2024-06-30
│   └── processed/
│       └── unified_dataset.csv     # Feature-engineered dataset with ~20 columns
│
├── outputs/
│   ├── charts/                     # 6 PNG visualizations (A through F)
│   │   ├── A_time_series_correlation.png
│   │   ├── B_multi_axis.png
│   │   ├── C_geospatial_map.png
│   │   ├── D_heatmap_correlation.png
│   │   ├── E_event_impact.png
│   │   └── F_lag_analysis.png
│   └── report.md                   # Copy of the analytical report
│
├── docs/                           # Documentation
│   ├── report.md                   # Full analytical report with policy implications
│   ├── Context.md                  # THIS FILE — master project reference
│   ├── assignment_questions.pdf    # Original assignment PDF from professor
│   └── screenshots/               # Dashboard screenshots
│       ├── Dashboard.png
│       └── Policy Insights Report.png
│
├── Course/                         # Personal lecture notes (gitignored)
├── study.md                        # Personal exam study guide (gitignored)
└── .venv/                          # Virtual environment (gitignored)
```

---

## 3. How to Run

### Prerequisites
- Python 3.8+ installed on the system
- Internet connection (for Yahoo Finance; works offline with synthetic fallback)

### Commands
```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate              # Windows
# source .venv/bin/activate           # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the data pipeline (fetches real data + generates all charts)
python main.py

# 4. Launch the interactive Streamlit dashboard
streamlit run app.py
```

### Environment Quirk (Windows)
On this machine, the `.venv` was created using `C:\Program Files\Python313\python.exe`. The activation script is at `.\.venv\Scripts\Activate.ps1` (standard Windows path). If the default `python` command resolves to MSYS2 (`C:\msys64\ucrt64\bin\python.exe`), use the full path or activate the venv first.

---

## 4. Pipeline Architecture

```
main.py orchestrates 4 sequential stages:

  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 1: DATA INGESTION (src/data_loader.py)                   │
  │   • Fetches Oil (CL=F), S&P 500 (^GSPC), Gold (GC=F) from    │
  │     Yahoo Finance using yfinance                               │
  │   • Generates synthetic: Conflict_Intensity, CO2, Inflation,   │
  │     Exchange_Rate (no free daily API exists for these)          │
  │   • Injects 4 geopolitical event shocks with exponential decay │
  │   • Graceful fallback: if yfinance fails → all synthetic       │
  │   • Output: data/raw/raw_data.csv (547 rows × 9 columns)      │
  └─────────────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 2: PREPROCESSING (src/preprocess.py)                     │
  │   • Sets Date as DatetimeIndex                                 │
  │   • Resamples to daily frequency (default; supports weekly)    │
  │   • Forward fill (ffill) for weekends/holidays                 │
  │   • Backward fill (bfill) for leading NaNs                     │
  └─────────────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 3: FEATURE ENGINEERING (src/features.py)                 │
  │   • Market_Volatility_Measure — 7-day rolling std of Stocks    │
  │   • Oil_Shock — binary flag if daily return > 5%               │
  │   • Oil_Lag_3, Oil_Lag_5, Conflict_Lag_7 — temporal shifts     │
  │   • Environmental_Impact_Score — scaled CO2 metric             │
  │   • Oil_Rolling_Mean, Conflict_Smoothing — 7-day rolling means │
  │   • Is_Event_Day, Event_Active_Window — event detection flags  │
  │   • Output: data/processed/unified_dataset.csv                 │
  └─────────────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 4: VISUALIZATION (src/visualizations.py)                 │
  │   • A: Time-Series Correlation (3-panel, dual-axis)            │
  │   • B: Multi-Axis Dynamics (triple Y-axis)                     │
  │   • C: Geospatial Map (conceptual Matplotlib scatter)          │
  │   • D: Correlation Heatmap (Seaborn, Pearson)                  │
  │   • E: Event Impact Timeline (annotated conflict events)       │
  │   • F: Lag Analysis (cross-correlation bar charts)             │
  │   • Output: outputs/charts/A–F.png                             │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Source Strategy (Hybrid)

| Variable | Source | Type | Reason for Choice |
|----------|--------|------|--------------------|
| Oil Price (WTI Crude) | Yahoo Finance `CL=F` | **Real** | Free, no API key, daily granularity |
| S&P 500 Index | Yahoo Finance `^GSPC` | **Real** | Free, no API key, daily granularity |
| Gold Futures | Yahoo Finance `GC=F` | **Real** | Free, no API key, daily granularity |
| Conflict Intensity | Procedural generation | Synthetic | ACLED requires institutional OAuth (myACLED) |
| CO2 Emissions | Procedural generation | Synthetic | Global Carbon Project = annual only |
| Inflation (CPI) | Procedural generation | Synthetic | FRED = monthly only, not daily |
| Exchange Rate (IRR) | Procedural generation | Synthetic | USD/IRR unreliable on Yahoo Finance |

### Graceful Fallback
If `yfinance` fails (no internet, API down, SSL issues), each variable independently falls back to synthetic Brownian motion generation. The pipeline never crashes — it always produces valid output.

---

## 6. Geopolitical Events Modelled

| Date | Event Name | Intensity (0–100) |
|------|-----------|-------------------|
| 2023-04-10 | Airstrikes | 60 |
| 2023-08-15 | Oil facility attacks | 85 |
| 2023-11-20 | Strait closure threats | 70 |
| 2024-02-10 | Major naval standoff | 90 |

Each event triggers:
- **Conflict_Intensity** spike with exponential decay (`exp(-t/10)`) over 40 days
- **Inflation** delayed creep (inverse decay)
- **CO2_Emissions** dip (disrupted logistics proxy)

Daily noise (`N(5, 5)`) is added to Conflict_Intensity and clipped to [0, 100].

---

## 7. Streamlit Dashboard (app.py)

### Pages
1. **Dashboard** — Full-width interactive Plotly map + 5 static charts in a 2-column grid
2. **Policy Insights Report** — Renders `docs/report.md` as Streamlit markdown
3. **About** — Tech stack summary and data source explanation

### Interactive Map
- Uses `plotly.express.scatter_map` (NOT deprecated `scatter_mapbox`)
- 4 markers: Tehran, Strait of Hormuz, Khuzestan, Isfahan
- Theme toggle: Dark Mode / Light Mode / Standard Streets
- `width="stretch"` (NOT deprecated `use_container_width=True`)

### Chart Display
- Static charts loaded from `outputs/charts/*.png` using `st.image()`
- Charts arranged: left column = charts 2, 4, 6; right column = charts 3, 5

---

## 8. Dependencies (requirements.txt)

```
pandas
numpy
matplotlib
seaborn
plotly
streamlit
yfinance
```

---

## 9. Key Columns in Raw Dataset

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Daily from 2023-01-01 to 2024-06-30 |
| Oil_Price | float | WTI Crude $/barrel (real from Yahoo Finance) |
| Stock_Index | float | S&P 500 points (real from Yahoo Finance) |
| Gold_Price | float | Gold $/oz (real from Yahoo Finance) |
| CO2_Emissions | float | Daily proxy MtCO2 (synthetic, seasonal sine wave) |
| Inflation | float | Annualized % (synthetic, random walk) |
| Exchange_Rate | float | USD/IRR proxy (synthetic, random walk) |
| Conflict_Intensity | float | 0–100 scale (synthetic, event-driven + decay) |
| Event_Flag | string | Event name or "None" |

---

## 10. Engineered Features (unified_dataset.csv adds these)

| Feature | Method | Purpose |
|---------|--------|---------|
| Market_Volatility_Measure | 7-day rolling std(Stock_Index) | Captures post-shock uncertainty |
| Oil_Return | pct_change(Oil_Price) | Daily percentage return |
| Oil_Shock | Binary: Oil_Return > 5% | Flags extreme oil spikes |
| Oil_Lag_3 | shift(Oil_Price, 3) | Causal analysis: oil → stocks delay |
| Oil_Lag_5 | shift(Oil_Price, 5) | Extended lag window |
| Conflict_Lag_7 | shift(Conflict_Intensity, 7) | Conflict → inflation delay |
| Environmental_Impact_Score | (CO2 / mean(CO2)) × 100 | Normalized CO2 metric |
| Oil_Rolling_Mean | 7-day rolling mean(Oil_Price) | Smoothed oil trend |
| Conflict_Smoothing | 7-day rolling mean(Conflict) | Smoothed conflict trend |
| Is_Event_Day | Binary: Event_Flag ≠ "None" | Event detection |
| Event_Active_Window | rolling max(Is_Event_Day, 7) | 7-day event activity flag |

---

## 11. Known Quirks and Gotchas

1. **yfinance MultiIndex columns**: When downloading a single ticker, yfinance sometimes returns MultiIndex columns. The code flattens them with `raw.columns.get_level_values(0)`.

2. **Random seed**: `np.random.seed(42)` is set in `generate_data()` for reproducibility. All synthetic series are deterministic given the same seed.

3. **ffill/bfill deprecation**: Older pandas used `fillna(method='ffill')`. Current code uses `.ffill()` and `.bfill()` directly to avoid deprecation warnings.

4. **Plotly API changes**: `px.scatter_mapbox` → `px.scatter_map` (Plotly 6.0+). `use_container_width=True` → `width="stretch"` (Streamlit 1.35+).

5. **Static geospatial chart**: `visualizations.py` generates a conceptual Matplotlib scatter plot for the geospatial chart (C). The interactive Plotly map lives only in `app.py` (the Streamlit dashboard).

6. **Report duplication**: `report.md` exists in both `outputs/` and `docs/`. The `app.py` reads from `docs/report.md`. Keep both in sync if editing.

---

## 12. Git & Submission

- **Remote**: `https://github.com/Krishna200608/Assignment-6.git`
- **Branch**: `main`
- **Submission zip**: `IIT2023139_Assignment_5.zip` — excludes `.venv/`, `study.md`, `.git/`, `Course/`, `__pycache__/`
- **.gitignore** excludes: `.venv/`, `__pycache__/`, `.vscode/`, `.idea/`, `Course/`, `study.md`, `.gemini/`, `.agents/`, `*.zip`, OS files

---

## 13. Visualization Details

### A. Time-Series Correlation (`A_time_series_correlation.png`)
- 3 vertically stacked subplots sharing the x-axis
- Panel 1: Oil Price (red, left Y) vs Stock Index (blue, right Y) — dual axis
- Panel 2: Conflict Intensity (orange, left Y) vs Oil Price (red dashed, right Y)
- Panel 3: CO2 Emissions (green) with red vertical lines marking event days

### B. Multi-Axis Dynamics (`B_multi_axis.png`)
- Single plot with 3 Y-axes: Oil (red), Stocks (blue), CO2 (green dash-dot)
- Third axis offset 60px outward from right spine

### C. Geospatial Map (`C_geospatial_map.png`)
- Conceptual Matplotlib scatter (not a real map tile)
- 4 points: Tehran, Strait of Hormuz, Khuzestan, Isfahan
- Color-coded by type, size-coded by strategic importance

### D. Correlation Heatmap (`D_heatmap_correlation.png`)
- Seaborn heatmap of Pearson correlations among 7 variables
- Colormap: `coolwarm`, annotations: 2 decimal places

### E. Event Impact Timeline (`E_event_impact.png`)
- Conflict Intensity (red) and centered Oil Price (orange) on same plot
- Vertical gray dashed lines at event dates
- Rotated event labels in frosted bounding boxes pinned to top

### F. Lag Analysis (`F_lag_analysis.png`)
- Two side-by-side bar charts
- Left: Oil vs Stock cross-correlation at lags -10 to +10
- Right: Conflict vs Inflation cross-correlation at lags -10 to +10
- Manually computed (no statsmodels dependency)

---

## 14. Quick Reference for Common Tasks

| Task | Command |
|------|---------|
| Run full pipeline | `.\.venv\Scripts\python.exe main.py` |
| Launch dashboard | `.\.venv\Scripts\python.exe -m streamlit run app.py` |
| Install deps | `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| Create submission zip | `Compress-Archive -Path app.py,main.py,requirements.txt,README.md,.gitignore,data,docs,outputs,src -DestinationPath IIT2023139_Assignment_5.zip -Force` |
| Push to GitHub | `git add -A && git commit -m "message" && git push origin main` |
