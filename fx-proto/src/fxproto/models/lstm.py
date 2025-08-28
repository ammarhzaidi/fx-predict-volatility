from __future__ import annotations
import torch
import torch.nn as nn

class TinyLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]         # last time step
        return self.head(out).squeeze(-1)

def train_model(model, X, y, epochs=15, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(1, epochs+1):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    return model
