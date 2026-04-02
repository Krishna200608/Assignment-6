# Multimodal Visual Analytics of Geopolitical Conflict
## Impact of Iran-USA War on Financial Markets and Environmental Indicators

**Author:** Krishna Sikheriya

**Roll Number:** IIT2023139

---

### 1. Objective
The objective of this project is to perform a multimodal visual analysis of a simulated timeline detailing the Iran-USA geopolitical conflict (January 2023 - June 2024). The analysis evaluates the compounding effects of conflict events—such as airstrikes and naval standoffs in the Strait of Hormuz—on key macroeconomic variables (Oil, S&P 500, Gold, Inflation, Exchange Rate) and environmental indicators (CO2 emissions proxies).

### 2. Data Design and Preprocessing
The project employs a **hybrid data strategy**, combining real-world financial market data with synthetic conflict and environmental indicators.
- **Real Market Data (Yahoo Finance):** Oil prices (WTI Crude Futures, `CL=F`), S&P 500 Index (`^GSPC`), and Gold Futures (`GC=F`) are fetched directly from Yahoo Finance using the `yfinance` Python library — free, no API key required.
- **Synthetic Data (Where Free Daily APIs Unavailable):** Conflict intensity events are modelled synthetically because ACLED requires institutional OAuth registration. Daily CO2 emissions are synthetic because the Global Carbon Project publishes annually. Inflation (CPI) is synthetic because FRED provides monthly data only.
- **Graceful Fallback:** If Yahoo Finance is unreachable (e.g., no internet), the pipeline automatically falls back to procedurally generated synthetic data for all variables, ensuring the project always runs offline.
- **Data Integration Pipeline:** The `data_loader.py` script produces 547 daily records encompassing 7 fundamental indicators alongside an integrated `Conflict_Intensity` index and an `Event_Flag`.
- **Time Alignment Logic:** The `preprocess.py` module resamples and aligns the metrics. Forward filling (`ffill`) ensures continuity for financial data across weekends and holidays, establishing a clean daily timescale for exact lag-analysis and rolling derivations.

### 3. Feature Engineering
The pipeline engineers several high-level analytical variables (`features.py`):
1. **Market_Volatility_Measure:** 7-day rolling standard deviation of the Stock Index, capturing market uncertainty directly following kinetic events.
2. **Oil_Shock:** A binary indicator triggered when daily oil price returns exceed 5%.
3. **Environmental_Impact_Score:** A scaled metric representing CO2 proxy data.
4. **Temporal Lags:** Variables like `Oil_Lag_3` and `Conflict_Lag_7` to allow causal analysis (e.g., measuring how long inflation takes to respond to a naval blockade threat).
5. **Event Active Window:** A rolling binary flag ensuring event impacts are analyzed dynamically over a 7-day decay period.

### 4. Visualizations and Insights

#### A. Time-Series Correlation Analysis
- **Observation:** We observe an inverse structural relationship between the S&P 500 and Oil Price immediately following conflict spikes. CO2 emissions exhibit a slight dip during maximum conflict intensity (likely proxying disrupted shipping and economic slowdowns), but quickly revert to baseline trends.

#### B. Multi-Axis Visualization
- **Observation:** Aligning Oil (Y1), Stocks (Y2), and CO2 (Y3) on a unified time axis reveals that market variables react almost instantaneously to conflict news, whereas environmental variables (CO2) exhibit a smoother, delayed structural lag.

#### C. Geospatial Visualization
- **Observation:** The conceptual Plotly map highlights critical infrastructure chokepoints (e.g., Strait of Hormuz). The intensity node size denotes the vulnerability of the region, aligning with the narrative that 20%+ of global oil transit is concentrated here, explaining the massive risk premium injected into oil prices during naval threats.

#### D. Correlation Heatmap
- **Observation:** There is a strong positive correlation between `Conflict_Intensity` and `Gold_Price` (safe-haven asset tracking). Conversely, `Stock_Index` shows a strong negative partial correlation with the conflict measure.

#### E. Event Impact Visualization
- **Observation:** By overlaying binary event flags (e.g., "Airstrikes" on April 10, 2023) atop the baseline oil price trajectory, we see clear structural breaks. Prices do not immediately decay; the "fear premium" sustains elevated oil prices for 30–40 days post-event.

#### F. Causal and Lag Analysis (Cross-Correlation)
- **Observation:** The lag analysis chart demonstrates that an oil price shock typically leads the stock market decline by 1 to 4 days (peak negative correlation at lag -3). Conversely, inflation shows a much longer tail, peaking in correlation with conflict intensity 7 to 14 days after the initial event. 

### 5. Limitations
- **Hybrid Data:** Financial market data (Oil, Stocks, Gold) is real from Yahoo Finance, but conflict intensity, CO2 emissions, and inflation are synthetically modelled due to the lack of free, daily APIs. Real-world structural breaks may exhibit different tail behavior.
- **Omitted Variables:** Real-world analyses should ideally include secondary policy responses (e.g., Federal Reserve rate hikes acting as confounding variables).

### 6. Policy Implications
1. **Energy Independence:** The rapid 1-to-4 day transmission mechanism from Strait of Hormuz naval threats to domestic stock market volatility underscores the critical need for strategic petroleum reserves and diversified energy portfolios.
2. **Environmental Resilience:** Interestingly, conflicts that depress global shipping temporarily reduce localized marine CO2 emissions but can increase localized pollution from burning oil infrastructure. Policy should address the environmental cost of war zone cleanups.
3. **Macroeconomic Hedging:** Central banks must monitor immediate gold and oil futures as early warning indicators, rather than waiting for lagging metrics like inflation, to implement monetary stabilization faster during geopolitical shocks.
