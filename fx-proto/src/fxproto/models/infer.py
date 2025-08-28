from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from fxproto.models.lstm import TinyLSTM

def predict_series(model: TinyLSTM, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        X_t = torch.tensor(X).to(device)
        preds = model(X_t).cpu().numpy()
    return preds

def add_predictions_to_frame(df: pd.DataFrame, idx_from: int, preds: np.ndarray, target_col: str, horizon: int):
    # Align predictions with original df timeline (start at index where first y exists)
    pred_index = df.index[idx_from + horizon : idx_from + horizon + len(preds)]
    out = pd.DataFrame({"pred": preds}, index=pred_index)
    out["actual"] = df.loc[pred_index, target_col]
    return out
