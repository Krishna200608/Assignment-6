import pandas as pd
import numpy as np

def fetch_real_financial_data(start_date='2025-01-01', end_date='2026-06-30'):
    """
    Fetches REAL financial market data from Yahoo Finance (free, no API key).
    Tickers:
      - CL=F  → WTI Crude Oil Futures ($/barrel)
      - ^GSPC → S&P 500 Index
      - GC=F  → Gold Futures ($/oz)
    """
    import yfinance as yf

    tickers = {
        'Oil_Price': 'CL=F',
        'Stock_Index': '^GSPC',
        'Gold_Price': 'GC=F',
    }

    frames = {}
    for col_name, ticker in tickers.items():
        try:
            print(f"  Downloading {col_name} ({ticker}) from Yahoo Finance...")
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if raw.empty:
                raise ValueError(f"No data returned for {ticker}")
            # yfinance may return MultiIndex columns; flatten them
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            
            series = raw['Close'].rename(col_name)
            frames[col_name] = series
        except Exception as e:
            print(f"  ⚠ Failed to fetch {col_name}: {e}. Will use synthetic fallback.")
            frames[col_name] = None

    return frames


def generate_synthetic_fallback(col_name, dates, n):
    """Generates synthetic data for a single column as a fallback."""
    np.random.seed(42)
    fallbacks = {
        'Oil_Price': 75.0 + np.cumsum(np.random.normal(0, 0.5, n)),
        'Stock_Index': 4000.0 + np.cumsum(np.random.normal(0.5, 10, n)),
        'Gold_Price': 1800.0 + np.cumsum(np.random.normal(0.2, 2, n)),
        'Exchange_Rate': 42000.0 + np.cumsum(np.random.normal(10, 50, n)),
    }
    return pd.Series(fallbacks.get(col_name, np.zeros(n)), index=dates, name=col_name)


def generate_data(start_date='2025-01-01', end_date='2026-06-30'):
    """
    Hybrid data loader:
      - Financial data (Oil, Stocks, Gold) → REAL data from Yahoo Finance
      - Conflict Intensity, CO2 Emissions   → Synthetic (no free daily API exists)
      - Exchange Rate                       → Attempted real, fallback to synthetic

    Why hybrid?
      ACLED conflict data requires institutional OAuth registration.
      Daily CO2 data from NASA/Global Carbon Project is annual, not daily.
      Financial data from Yahoo Finance is completely free and needs no API key.
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)

    # --- Attempt to fetch REAL financial data ---
    print("  [Real Data] Fetching financial data from Yahoo Finance...")
    real_data = fetch_real_financial_data(start_date, end_date)

    # --- Build the unified DataFrame ---
    data = pd.DataFrame({'Date': dates})
    data = data.set_index('Date')

    # Oil Price: real or fallback
    if real_data.get('Oil_Price') is not None:
        data = data.join(real_data['Oil_Price'], how='left')
        data['Oil_Price'] = data['Oil_Price'].ffill().bfill()
        data['_oil_source'] = 'Yahoo Finance (CL=F)'
    else:
        data['Oil_Price'] = generate_synthetic_fallback('Oil_Price', dates, n)
        data['_oil_source'] = 'Synthetic'

    # Stock Index: real or fallback
    if real_data.get('Stock_Index') is not None:
        data = data.join(real_data['Stock_Index'], how='left')
        data['Stock_Index'] = data['Stock_Index'].ffill().bfill()
        data['_stock_source'] = 'Yahoo Finance (^GSPC)'
    else:
        data['Stock_Index'] = generate_synthetic_fallback('Stock_Index', dates, n)
        data['_stock_source'] = 'Synthetic'

    # Gold Price: real or fallback
    if real_data.get('Gold_Price') is not None:
        data = data.join(real_data['Gold_Price'], how='left')
        data['Gold_Price'] = data['Gold_Price'].ffill().bfill()
        data['_gold_source'] = 'Yahoo Finance (GC=F)'
    else:
        data['Gold_Price'] = generate_synthetic_fallback('Gold_Price', dates, n)
        data['_gold_source'] = 'Synthetic'

    # --- Synthetic data (no free daily API available) ---

    # CO2 Emissions: Synthetic seasonal proxy
    # Reason: Global Carbon Project provides ANNUAL data, not daily.
    # NASA MODIS satellite data requires complex image processing.
    data['CO2_Emissions'] = 35.0 + np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.1, n)

    # Inflation: Synthetic gradual creep
    # Reason: CPI data from FRED is monthly, not daily.
    data['Inflation'] = 3.0 + np.cumsum(np.random.normal(0, 0.01, n))

    # Exchange Rate: Synthetic (USD/IRR is unreliable on Yahoo Finance)
    data['Exchange_Rate'] = 42000.0 + np.cumsum(np.random.normal(10, 50, n))

    # Conflict Intensity: Synthetic
    # Reason: ACLED requires institutional OAuth registration (myACLED account).
    data['Conflict_Intensity'] = np.zeros(n)

    # --- Inject geopolitical event shocks ---
    events = {
        '2025-04-10': ('Airstrikes', 60),
        '2025-08-15': ('Oil facility attacks', 85),
        '2025-11-20': ('Strait closure threats', 70),
        '2026-02-10': ('Major naval standoff', 90),
    }

    data['Event_Flag'] = 'None'

    for event_date_str, (event_name, intensity_spike) in events.items():
        event_date = pd.to_datetime(event_date_str)
        if event_date in data.index:
            data.loc[event_date, 'Conflict_Intensity'] = intensity_spike
            data.loc[event_date, 'Event_Flag'] = event_name

            # Decay impact over 40 days post-event
            for i in range(1, 40):
                future_date = event_date + pd.Timedelta(days=i)
                if future_date in data.index:
                    decay = np.exp(-i / 10.0)
                    data.loc[future_date, 'Conflict_Intensity'] += intensity_spike * decay

                    # Inflation creeps up with delay
                    data.loc[future_date, 'Inflation'] += (intensity_spike / 1000.0) * (1 - decay)

                    # CO2 dips due to disrupted logistics
                    data.loc[future_date, 'CO2_Emissions'] -= (intensity_spike / 50.0) * decay

    # Add daily noise to conflict intensity
    data['Conflict_Intensity'] += np.random.normal(5, 5, n)
    data['Conflict_Intensity'] = data['Conflict_Intensity'].clip(0, 100)

    # Reset index so Date becomes a column again
    data = data.reset_index()

    # Log data source summary
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │       DATA SOURCE SUMMARY                   │")
    print("  ├─────────────────────────────────────────────┤")
    print(f"  │ Oil Price        : {data.get('_oil_source', pd.Series(['?'])).iloc[0]:<24}│")
    print(f"  │ Stock Index      : {data.get('_stock_source', pd.Series(['?'])).iloc[0]:<24}│")
    print(f"  │ Gold Price       : {data.get('_gold_source', pd.Series(['?'])).iloc[0]:<24}│")
    print("  │ Exchange Rate    : Synthetic (IRR unreliable)│")
    print("  │ Inflation        : Synthetic (CPI=monthly)  │")
    print("  │ CO2 Emissions    : Synthetic (annual only)  │")
    print("  │ Conflict Data    : Synthetic (ACLED=OAuth)  │")
    print("  └─────────────────────────────────────────────┘")

    # Drop internal source tracking columns
    data = data.drop(columns=[c for c in data.columns if c.startswith('_')], errors='ignore')

    return data


# Legacy wrapper for backward compatibility with main.py
def generate_synthetic_data(start_date='2025-01-01', end_date='2026-06-30'):
    """Backward-compatible wrapper. Now fetches real data where possible."""
    return generate_data(start_date, end_date)


if __name__ == "__main__":
    df = generate_data()
    df.to_csv('../data/mock_conflict_data.csv', index=False)
    print(f"\nDataset generated: {len(df)} rows, {len(df.columns)} columns")
    print(df.head())
