# 📖 Study Guide: DV Lab Assignment 5 — Evaluation Prep
**Student:** Krishna Sikheriya (IIT2023139)  
**Exam Format:** Live Demo + Viva (Q&A) | **Marks:** 10  
**Study Plan:** 30 min tonight (Part A + B) → 1.5–2 hrs tomorrow (Part C + D + Practice)

---

## ⏱️ PART A — YOUR PROJECT IN 5 MINUTES (Read Tonight)
> **Goal:** Understand what your project does so you can explain it confidently in 2 sentences.

### What does your project do?
It analyzes how the **Iran-USA geopolitical conflict** (Jan 2024 – Jun 2026) impacts **financial markets** (Oil, Stocks, Gold, Inflation) and **environmental indicators** (CO2 Emissions). It uses a **Macro-Interpolated Data Strategy** — strictly parsing 100% REAL geospatial json data for conflict and directly piping REAL Yahoo Finance tick data, alongside statistically correlated proxies for un-fetchable metrics — processes it through an advanced feature engineering pipeline, creates 6 analytical charts, and displays everything in a professional Streamlit dashboard.

### The Pipeline (4 Steps)
```
Step 1: Generate Data  →  Step 2: Clean & Align  →  Step 3: Engineer Features  →  Step 4: Visualize
(data_loader.py)         (preprocess.py)           (features.py)               (visualizations.py + app.py)
```

### How to Run It
```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Generate charts (fetches real data from Yahoo Finance)
python main.py

# Step 3: Launch the interactive dashboard
streamlit run app.py
```

### Data Source Strategy (Hybrid)
The project uses a **hybrid real + synthetic** approach:

| Data | Source | Why? |
|------|--------|------|
| Oil Price (WTI Crude) | **Yahoo Finance** (`CL=F`) | Free, no API key, daily data |
| S&P 500 Stock Index | **Yahoo Finance** (`^GSPC`) | Free, no API key, daily data |
| Gold Futures | **Yahoo Finance** (`GC=F`) | Free, no API key, daily data |
| Conflict Intensity | **Synthetic** | ACLED requires institutional OAuth registration |
| CO2 Emissions | **Synthetic** | Global Carbon Project provides annual data only, not daily |
| Inflation (CPI) | **Synthetic** | FRED provides monthly data only, not daily |
| Exchange Rate (IRR) | **Synthetic** | USD/IRR unreliable on Yahoo Finance |

If Yahoo Finance is unreachable (no internet), the pipeline **gracefully falls back** to synthetic data for all variables.

---

## 📊 PART B — THE 6 VISUALIZATIONS EXPLAINED SIMPLY (Read Tonight)

### Chart 1: Geospatial Map (Interactive Plotly Map)
- **What it shows:** A real map of the Middle East with dots marking conflict zones (Tehran), oil routes (Strait of Hormuz), oil fields (Khuzestan), etc.
- **Why it matters:** ~20% of global oil passes through the Strait of Hormuz. If Iran blocks it → oil prices skyrocket globally.
- **DV Concept Used:** Spatial data visualization, interactive exploration (Shneiderman's Mantra: overview first, zoom and filter, details on demand).
- **Tool:** `plotly.express.scatter_map` with CartoDB tiles.
- **Marks & Channels:** Points (marks), Color = category/type, Size = intensity.

### Chart 2: Time-Series Correlation (3-panel Matplotlib plot)
- **What it shows:** Three sub-plots stacked vertically:
  1. Oil Price vs Stock Market Index (they move in opposite directions!)
  2. Conflict Intensity vs Oil Price (conflict spikes → oil rises)
  3. CO2 Emissions over the war timeline (red vertical lines = event days)
- **DV Concept Used:** Line chart (best for time-series), dual-axis plotting, small multiples (3 panels instead of spaghetti).
- **Key Insight to mention:** "We observe an **inverse structural relationship** between Oil and Stocks — when conflict pushes oil up, the stock market drops."

### Chart 3: Multi-Axis Dynamics (3 Y-axes on one plot)
- **What it shows:** Oil (red), Stocks (blue), and CO2 (green) all on the same timeline but with their own Y-axis scales.
- **Why 3 axes?** Because Oil is ~$50–120, Stocks are ~3000–5000, and CO2 is ~40–65. Directly plotting them on one axis would make CO2 invisible.
- **DV Concept Used:** Multi-axis visualization. Note: Dual-axis charts are generally **discouraged** (see your syllabus!), but here we use it deliberately because the alternative (3 separate charts) loses the temporal synchronization insight.
- **Key Insight:** "Financial markets react **instantly** to conflict (1–2 days), but environmental indicators (CO2) show a **smoother, delayed response**."

### Chart 4: Correlation Heatmap (Seaborn)
- **What it shows:** A color-coded matrix showing the **Pearson correlation** between all 7 variables.
- **How to read it:**
  - `+1.0` = perfectly move together (dark red)
  - `-1.0` = perfectly move in opposite directions (dark blue)
  - `0.0` = no relationship
- **DV Concept Used:** Heatmap (Tier 3 comparison chart — good for patterns, poor for exact values). Uses **color saturation** as the visual channel.
- **Key Insight:** "Gold shows strong **positive** correlation with Conflict (safe-haven asset). Stocks show strong **negative** correlation with Conflict."

### Chart 5: Event Impact Timeline (Annotated Matplotlib plot)
- **What it shows:** Conflict intensity (red line) and Oil Price (orange line) plotted over time, with **vertical dashed lines** marking specific events (e.g., "Airstrikes", "Oil rig attacks", "Naval standoff").
- **DV Concept Used:** Event annotation, timeline visualization, contextual storytelling.
- **Key Insight:** "After an event like airstrikes, the 'fear premium' keeps oil prices elevated for **30–40 days** before they decay back."

### Chart 6: Lag & Causal Analysis (Cross-Correlation Bar Charts)
- **What it shows:** Two bar charts:
  1. How correlated Oil and Stocks are at different time lags (-10 to +10 days)
  2. How correlated Conflict and Inflation are at different time lags
- **What is a lag?** If oil spiked today and stocks dropped **3 days later**, that's a lag of 3. We shift one dataset by N days and recalculate correlation.
- **DV Concept Used:** Bar chart for comparison (Tier 1 — highest precision), cross-correlation analysis.
- **Key Insight:** "Oil shocks lead stock market drops by **1–4 days** (peak at lag -3). Inflation responds to conflict with a **7–14 day delay**."

---

## 🧠 PART C — DV THEORY CONCEPTS USED IN YOUR PROJECT (Study Tomorrow)

### C1. Visual Analytics (Module 5)
> **Simple definition:** Combining automatic computer analysis + visual charts + human judgment to understand data.

Your project IS visual analytics because:
- **Automatic Analysis:** Python scripts generate data, compute correlations, engineer features
- **Visualization:** 6 charts present the results visually
- **Human Interaction:** The Streamlit dashboard lets users switch map themes, navigate between pages, scroll through charts

**The Visual Analytics Process in your project:**
```
Raw Data → Data Transformation (preprocess.py) → Automated Analysis (features.py)
    ↓                                                       ↓
Visual Exploration (6 charts)  ←←←←←←←←←←←←←←←←← Human Interaction (Streamlit dashboard)
    ↓
Knowledge & Insights (report.md)
```

### C2. Feature Engineering (Module 6)
> **Simple definition:** Creating new useful columns from raw data to make analysis better.

**Features you engineered (in `features.py`):**

| Feature | What it does | FE Technique |
|---------|-------------|-------------|
| `Market_Volatility` | 7-day rolling standard deviation of Stock Index | **Feature Creation** (mathematical operation) |
| `Oil_Shock` | Binary flag: 1 if daily oil return > 5%, else 0 | **Discretization / Binning** (continuous → binary) |
| `Environmental_Impact_Score` | Scaled CO2 metric | **Feature Transformation** (scaling) |
| `Oil_Lag_3`, `Conflict_Lag_7` | Shifted values for causal analysis | **Feature Creation** (temporal lag) |
| `Event_Active_Window` | Rolling flag over 7-day decay | **Feature Creation** (windowed aggregation) |

### C3. Data Preprocessing
> **Simple definition:** Cleaning and fixing your raw data before analysis.

**What `preprocess.py` does:**
- **Time Alignment:** Financial markets work Mon–Fri, but CO2 data is daily. We resample everything to a common daily timeline.
- **Missing Value Handling:** Uses `ffill()` (forward fill) — if Wednesday's stock price is missing, use Tuesday's value. Then `bfill()` for any remaining gaps at the start.
- This is called **Imputation** — filling in missing values instead of deleting entire rows.

### C4. Data Types (NOIR) — In Your Project

| Variable | Data Type | Why? |
|----------|----------|------|
| Oil_Price, Stock_Index, Gold | **Ratio** | Numeric, meaningful zero ($0 = no value) |
| Inflation, CO2_Emissions | **Ratio** | Numeric with true zero |
| Event_Flag ("Airstrikes", "None") | **Nominal** | Categories with no order |
| Conflict_Intensity (0–100) | **Ratio** | Numeric scale with true zero |
| Oil_Shock (0 or 1) | **Nominal** (binary) | Two categories |

### C5. Marks and Channels — In Your Charts

| Chart | Marks Used | Channels Used |
|-------|-----------|--------------|
| Geospatial Map | **Points** (0D) | Position (lat/lon), Color (type), Size (intensity) |
| Time-Series | **Lines** (1D) | Position (x=time, y=value), Color (variable identity) |
| Multi-Axis | **Lines** (1D) | Position, Color, separate Y-axis scales |
| Heatmap | **Areas** (2D cells) | Color saturation (correlation value) |
| Event Impact | **Lines** + **Lines** (annotations) | Position, Color, vertical markers |
| Lag Analysis | **Bars** (1D) | Length on common baseline (highest precision!) |

### C6. Gestalt Principles — In Your Dashboard

| Principle | Where it appears |
|-----------|-----------------|
| **Proximity** | Charts in 2-column grid are perceived as related pairs |
| **Similarity** | Same color (red) = always Conflict/Oil across all charts |
| **Continuity** | Time flows left-to-right consistently in every time chart |
| **Figure & Ground** | Dark dashboard background vs white chart cards |

### C7. Five Guidelines for Better Visualization

| Guideline | How your project follows it |
|-----------|---------------------------|
| 1. Show the Data | Every chart directly displays the raw data, not just summaries |
| 2. Reduce Clutter | Clean whitegrid style, no 3D effects, no heavy gridlines |
| 3. Integrate Graphics & Text | Captions under each chart, annotated event labels |
| 4. Avoid Spaghetti | Time-series uses 3 separate panels (small multiples) instead of 6 lines on one chart |
| 5. Start with Gray | Event Impact uses gray baselines with colored highlights for key data |

### C8. Shneiderman's Mantra in Your Dashboard
> **"Overview first, zoom and filter, then details-on-demand."**

- **Overview first:** Dashboard page shows ALL 6 charts at a glance
- **Zoom and filter:** Interactive map allows zoom/pan; sidebar lets you navigate to specific pages
- **Details-on-demand:** Policy Insights Report page provides deep written analysis

### C9. Munzner's Nested Model in Your Project

| Level | Your Project |
|-------|-------------|
| 1. Domain Situation (Who) | Geopolitical analysts studying Iran-USA conflict economics |
| 2. Abstraction — What | Daily time-series data with 7 numerical features + 1 categorical event flag |
| 3. Abstraction — Why | Discover trends, compare variables, identify outliers/events, find correlations |
| 4. Idiom (How) | Line charts, heatmap, bar charts, scatter-map, multi-axis plot |
| 5. Algorithm | Python (pandas, matplotlib, seaborn, plotly, streamlit) |

### C10. Form × Function Quadrant
Your project is **Interactive & Exploratory** (Quadrant 4):
- **Interactive:** Map is zoomable, dashboard has sidebar navigation, pages are switchable
- **Exploratory:** User can explore the charts, find their own patterns, read the report

### C11. Data Scaling (Module 6) — Concepts to Know

| Method | Formula | Range | Outlier-Safe? | When to Use |
|--------|---------|-------|--------------|-------------|
| Min-Max | $(v - min) / (max - min)$ | 0 to 1 | ❌ No | Visualization, neural networks |
| Z-Score (Standardization) | $(v - mean) / std$ | centered at 0 | Partially | PCA, clustering |
| Robust Scaling | $(v - median) / IQR$ | median-centered | ✅ Yes | Data with outliers |

**In your project:** The `Environmental_Impact_Score` uses scaling to bring CO2 onto a comparable metric.

### C12. Pearson Correlation — What Your Heatmap Shows

$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \cdot \sum(y_i - \bar{y})^2}}$$

- **r = +1:** Perfect positive (both go up together)
- **r = -1:** Perfect negative (one goes up, other goes down)
- **r = 0:** No linear relationship

**Your heatmap shows:** Oil & Conflict are positively correlated. Stocks & Conflict are negatively correlated. Gold & Conflict are positively correlated (safe-haven effect).

---

## 🎤 PART D — PROBABLE VIVA QUESTIONS & ANSWERS (Study Tomorrow)

### Q1: "Explain your project in 2 lines."
> "This project analyzes how the Iran-USA geopolitical conflict impacts financial markets and environmental indicators using 6 multimodal visualizations. I built a hybrid data pipeline that fetches real market data from Yahoo Finance, supplements it with synthetic conflict and environmental data, engineers analytical features, and presents insights via an interactive Streamlit dashboard."

### Q2: "Where does your data come from?"
> "I use a hybrid approach. Oil, S&P 500, and Gold prices are **real data** fetched from Yahoo Finance using the `yfinance` library — it's completely free and needs no API key. For conflict intensity, I couldn't use ACLED because it requires institutional OAuth registration. For daily CO2 data, the Global Carbon Project only publishes annually. So those are modelled synthetically with realistic decay patterns. The pipeline has graceful fallback — if there's no internet, everything still works with synthetic data."

### Q3: "What is Visual Analytics?"
> "Visual Analytics combines three things: automatic computation (like my Python scripts), visual representations (my 6 charts), and human interaction (the Streamlit dashboard where users can explore). It's an iterative process — you look at charts, form hypotheses, test them with computation, and refine."

### Q4: "What is Feature Engineering? Give examples from your project."
> "Feature Engineering is creating new useful variables from raw data to improve analysis. In my project, I created:
> - **Rolling Volatility** (7-day standard deviation of stock prices),
> - **Oil Shock indicator** (binary: 1 if oil jumps more than 5% in a day),
> - **Temporal lags** (shifting oil price by 3 days to see if it predicts stock drops later)."

### Q5: "Why is a heatmap ranked low in the perceptual ranking scale?"
> "Because it uses **color saturation** as the primary channel, which is ranked 6th out of 7 in Munzner's perceptual ranking. Humans are much better at comparing positions and lengths (bar charts) than comparing shades of color. But heatmaps are excellent for spotting **patterns** across many variables at once."

### Q6: "What are the disadvantages of dual-axis charts?"
> "They're hard to read because you don't know which line goes with which axis. The creator can manipulate the axis ranges to force false relationships. Gridlines don't align. In my project, I used a triple-axis chart deliberately because the variables have vastly different scales (Oil ~$50, Stocks ~4000, CO2 ~50), and I needed to show their synchronized timing."

### Q7: "What is Shneiderman's Mantra?"
> "Overview first, zoom and filter, then details-on-demand. My dashboard follows this: the Dashboard page gives an overview of all charts, the interactive map allows zooming and filtering, and the Policy Insights page provides detailed on-demand analysis."

### Q8: "What Gestalt principles does your dashboard use?"
> "**Proximity** — charts in the 2-column grid are perceived as related. **Similarity** — red always means conflict/oil across all charts. **Continuity** — time flows left-to-right consistently. **Figure & Ground** — the dark sidebar separates navigation from content."

### Q9: "What is ffill/bfill? Why did you use it?"
> "Forward fill (ffill) takes the last known value and carries it forward to fill missing gaps. Backward fill (bfill) does the same in reverse. I used it because financial data has no values on weekends/holidays, but my timeline is daily. Instead of deleting weekends, ffill keeps Friday's price for Saturday and Sunday."

### Q10: "What is cross-correlation / lag analysis?"
> "Cross-correlation measures how similar two signals are when one is shifted in time. I shift Oil Price by -10 to +10 days and recalculate its correlation with Stock Index at each shift. The lag where correlation is strongest tells us how many days it takes for an oil shock to impact stocks. My result: **3 days** (peak at lag -3)."

### Q11: "What data types (NOIR) are in your dataset?"
> "Oil Price, Stock Index, and Gold are **Ratio** (numeric with a true zero). Event_Flag (Airstrikes, Naval standoff) is **Nominal** (categories with no order). Conflict_Intensity is **Ratio** (0 to 100 with a true zero)."

### Q12: "What is Anscombe's Quartet and how does it relate to your project?"
> "Anscombe's Quartet is 4 datasets that have identical statistical summaries (same mean, variance, correlation) but look completely different when visualized. This proves you MUST visualize data, not just compute statistics. That's exactly why my project builds 6 charts rather than just printing numbers."

### Q13: "What tools and libraries did you use?"
> "**yfinance** for fetching real financial data from Yahoo Finance, **pandas** for data manipulation, **numpy** for numerical operations, **matplotlib** and **seaborn** for static charts, **plotly** for the interactive geospatial map, and **streamlit** for the web dashboard."

### Q14: "What is the key insight from your analysis?"
> "Three main insights: (1) Oil shocks lead stock market drops by 1–4 days. (2) Inflation takes 7–14 days to respond to conflict — it's a lagging indicator. (3) Gold acts as a safe-haven asset: it consistently rises during conflict spikes, confirming its role as a hedge."

### Q15: "What are the 10 components of Data Visualization?"
> "Data, Visual Elements, Visualization Techniques, Interactivity, Color & Design, Context & Annotations, Tools & Platforms, Data Preparation, Data Exploration, and Dashboarding. My project covers all 10."

### Q16: "Why didn't you use all real data?"
> "I used real data wherever a free, daily API exists. Yahoo Finance provides Oil, S&P 500, and Gold data for free — no API key needed. But ACLED (conflict data) requires institutional OAuth registration through myACLED, and the Global Carbon Project only publishes CO2 data annually, not daily. FRED provides CPI/inflation data only monthly. Since this is a Visual Analytics assignment, I focused on building the visualization pipeline and used synthetic models for the unavailable daily data sources. The pipeline is designed to be **swappable** — plug in a real CSV and everything downstream works unchanged."

---

## 🚀 QUICK REVISION CHECKLIST (Glance Before Eval)

- [ ] Can I run `pip install -r requirements.txt`, then `python main.py` and `streamlit run app.py` without errors?
- [ ] Can I explain what each of the 6 charts shows in one sentence?
- [ ] Can I explain the hybrid data strategy (yfinance = real, conflict/CO2 = synthetic)?
- [ ] Can I name 3 feature engineering techniques used (volatility, oil shock, lag)?
- [ ] Can I explain what ffill does and why it's needed?
- [ ] Can I explain why heatmaps rank low perceptually (color channel)?
- [ ] Can I state Shneiderman's Mantra and show where my dashboard follows it?
- [ ] Can I name 3 Gestalt principles visible in my dashboard?
- [ ] Can I explain cross-correlation/lag analysis in simple words?
- [ ] Can I say what NOIR stands for and classify at least 3 variables?
- [ ] Can I explain what Visual Analytics means (auto + visual + human)?

---

*Good luck tomorrow, Krishna! You've got this. 💪*
