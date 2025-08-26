# 🛠️ Prototype Build Order (Tight & Practical)

## 1. Decide the Demo Slice (Scope)
- **Pair:** `EURUSD` (daily data)  
- **Horizon:** 5 trading days ahead  
- **Metrics:** RMSE, MAE, Directional Accuracy  
- **Graph demo:** 6 nodes → `[EUR, USD, ECB, Fed, Interest Rate, GDP]`  
- **Scenario:** EU GDP negative shock  

---

## 2. Fill Configs First
- `config/settings.yaml` → paths, pair, horizon, train/val/test dates, seed  
- `config/data.yaml` → source = `yfinance`, symbol mapping, frequency, columns  
- `config/graph.yaml` → node list, edges (types + weights), scenarios  

---

## 3. Wire the Foundation
- `src/fxproto/utils/paths.py` → resolve project/data/output paths  
- `src/fxproto/utils/logging.py` → simple logger  
- `src/fxproto/config/loader.py` → load `.env` + YAMLs → typed settings dict/object  

---

## 4. Data Path (fetch → preprocess)
- `src/fxproto/data/fetch.py` → download/save EURUSD OHLCV  
- `src/fxproto/data/preprocess.py` → clean, split (train/val/test), scale **fit on train only**  
- `src/fxproto/data/features.py` → minimal features (returns, rolling mean/volatility)  

---

## 5. A Single Baseline Model
- `src/fxproto/models/lstm.py` → tiny LSTM/GRU baseline  
- `src/fxproto/models/infer.py` → load checkpoint, predict on test slice, return dataframe  

*(Skip GARCH/ARIMA + SHAP until later.)*

---

## 6. CLI Entrypoint (Experience Layer)
- `src/fxproto/app/cli.py` → commands:
  - `forecast`: fetch → preprocess → train → infer → save results  
  - `graphdemo`: load `graph.yaml` → build graph → scenario → plot  

---

## 7. Graph Demo (Toy but Visual)
- `src/fxproto/graphdemo/graph_build.py` → build from YAML  
- `src/fxproto/graphdemo/visualize.py` → draw nodes/edges + influence heatmap  

*(Add `message_passing.py` later for propagation logic.)*

---

## 8. GUI (Select → Run → Results)
- `scripts/ui_app.py` → simple **Streamlit app** for prototype demo  
  - Dropdowns for pair, horizon, date range  
  - Button to run `forecast` or `graphdemo` via `cli.py`  
  - Display charts from `outputs/charts/` and metrics from `outputs/reports/`  
- *(Later: migrate to `src/fxproto/app/server.py` for a FastAPI-based dashboard.)*

---

## 9. Sanity Tests
- `tests/test_config.py` → configs load, required keys present  
- `tests/test_shapes.py` → dataset windows have correct shapes  
- *(Add `test_graph_demo.py` after graph propagation is in place.)*  

---

## 10. Polish for Demo
- Save 2–3 plots to `outputs/charts/`:  
  - price + forecast  
  - residuals  
  - graph heatmap  
- Export one `outputs/reports/*.csv` with predictions + actuals + DA flag  

---

⚖️ **Why this order?**  
- You get a runnable CLI fast.  
- Config-first avoids rewrites.  
- GUI adds a polished “Select → Run → Results” experience without polluting core logic.  
- Tests catch shape/leakage bugs early.  
- Gives you baseline + graph demo + visuals in one week → perfect for your meeting.
