# FX Price Forecast & Volatility Prototype

This project is a prototype for **deep learning–based forecasting of Forex prices and volatility**, with an emphasis on **explainability** and a simple **demo interface**.  
It is being developed as part of a PhD research direction on **applied AI for financial decision-making**.


---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/ammarhzaidi/fx-predict-volatility.git
cd fx-predict-volatility


python -m venv .venv
.\.venv\Scripts\activate     # (Windows PowerShell)
# OR
source .venv/bin/activate    # (Linux/macOS)

pip install -r fx-proto/requirements.txt


python fx-proto/scripts/fetch_test.py

🧩 Prototype Goals

This prototype aims to deliver:

Configuration

Centralized YAML configs for settings, data, and graph scenarios.

Stop hard-coding tickers, dates, and paths.

Data Pipeline

Fetch OHLCV Forex data (e.g., EURUSD).

Clean & preprocess into train/validation/test sets.

Generate simple features (returns, rolling averages, volatility).

Forecasting Models

Baseline statistical models (ARIMA, GARCH).

Deep learning baseline: LSTM/GRU.

Save forecasts + metrics (RMSE, MAE, Directional Accuracy).

Explainability

SHAP values & attention visualization (later stage).

Highlight most influential features/time steps.

Graph Demo

Define a toy financial knowledge graph (nodes: EUR, USD, ECB, Fed, InterestRate, GDP).

Simulate propagation of shocks (e.g., negative EU GDP surprise).

Visualize influence flow with a simple Graph Attention–style mechanism.

Interfaces

CLI commands (forecast, graphdemo).

Streamlit mini-app (ui_app.py) for Select → Run → Results demo.

Testing

Unit tests for config loading, data shapes, and graph building.


python fx-proto/scripts/run_forecast.py --pair EURUSD


python fx-proto/scripts/run_graphdemo.py --scenario EU_negative_GDP_shock


streamlit run fx-proto/scripts/ui_app.py

📅 Roadmap

 Repo scaffold, configs, and data fetch.

 Preprocessing & feature generation.

 LSTM/GRU baseline training & inference.

 CLI integration (forecast/graphdemo).

 Graph demo with toy scenarios.

 Streamlit UI for demo.

 Explainability (SHAP, attention visualization).
 


# 📋 Markdown Section: *All what we are going to do in this prototype project*

```markdown
## All What We Are Going To Do in This Prototype Project

1. **Configuration**
   - Centralize settings in YAML (`settings.yaml`, `data.yaml`, `graph.yaml`).
   - Stop hard-coding tickers, dates, and paths.

2. **Data Handling**
   - Fetch OHLCV Forex data via `yfinance`.
   - Store raw CSVs in `data/raw/`.
   - Clean, scale, and split into train/test sets.
   - Engineer basic features (returns, moving averages, volatility).

3. **Forecasting Models**
   - Implement a simple LSTM/GRU model for short-horizon prediction.
   - Keep a placeholder for ARIMA/GARCH baselines.
   - Save outputs to `outputs/charts/` (plots) and `outputs/reports/` (CSV).

4. **Explainability**
   - Add SHAP-based feature importance.
   - Add simple attention visualisations.

5. **Graph Demo**
   - Define a toy financial knowledge graph with ~6 nodes (EUR, USD, ECB, Fed, InterestRate, GDP).
   - Encode edges with weights (influence).
   - Simulate shock scenarios (e.g., negative EU GDP).
   - Visualise node/edge influence propagation.

6. **Interfaces**
   - CLI (`cli.py`):
     - `forecast` command → runs fetch → preprocess → train → predict → save results.
     - `graphdemo` command → runs toy graph scenario → plots influence.
   - Streamlit app (`ui_app.py`) with dropdowns & a Run button.

7. **Testing**
   - Unit tests for config loading, data shapes, graph building.
   - Keep PyCharm tests separate from demo scripts.

8. **Polish**
   - Document everything in README.
   - Demo-ready plots & reports for the meeting.

👤 Author
Syed Ammar Hasan Zaidi
Visiting Lecturer (Postgraduate)
Department of Computer & Information Systems Engineering
NED University of Engineering & Technology, Karachi, Pakistan
