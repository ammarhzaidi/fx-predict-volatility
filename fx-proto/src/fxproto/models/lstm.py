# fx-proto/src/fxproto/models/lstm.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """
    Simple Graph Attention Layer for processing node features.
    """

    def __init__(self, in_features: int, out_features: int, n_nodes: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_nodes = n_nodes

        # Linear transformation for node features
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Attention mechanism
        self.attention = nn.Linear(2 * out_features, 1, bias=False)

        # Learnable adjacency matrix
        self.adjacency_weights = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)

    def forward(self, node_features: torch.Tensor, adjacency_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of graph attention.

        Args:
            node_features: (batch_size, n_nodes, in_features)
            adjacency_mask: (n_nodes, n_nodes) - binary mask for valid edges

        Returns:
            updated_features: (batch_size, n_nodes, out_features)
        """
        batch_size = node_features.size(0)

        # Transform node features
        h = self.W(node_features)  # (batch, n_nodes, out_features)

        # Compute attention coefficients
        attention_scores = self._compute_attention(h)  # (batch, n_nodes, n_nodes)

        # Apply adjacency mask if provided
        if adjacency_mask is not None:
            attention_scores = attention_scores * adjacency_mask.unsqueeze(0)

        # Apply learnable adjacency weights
        attention_scores = attention_scores * torch.sigmoid(self.adjacency_weights).unsqueeze(0)

        # Normalize attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, n_nodes, n_nodes)

        # Aggregate neighbor features
        updated_features = torch.bmm(attention_weights, h)  # (batch, n_nodes, out_features)

        return updated_features, attention_weights

    def _compute_attention(self, h: torch.Tensor) -> torch.Tensor:
        """Compute pairwise attention scores between all nodes."""
        batch_size, n_nodes, features = h.shape

        # Create all pairs of nodes for attention computation
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # (batch, n_nodes, n_nodes, features)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # (batch, n_nodes, n_nodes, features)

        # Concatenate node pairs
        pairs = torch.cat([h_i, h_j], dim=-1)  # (batch, n_nodes, n_nodes, 2*features)

        # Compute attention scores
        attention_scores = self.attention(pairs).squeeze(-1)  # (batch, n_nodes, n_nodes)

        return attention_scores


class GraphEnhancedLSTM(nn.Module):
    """
    LSTM enhanced with Graph Attention for financial forecasting.
    Combines time series patterns with graph-based node relationships.
    """

    def __init__(self, n_price_features: int, n_graph_nodes: int,
                 lstm_hidden: int = 64, graph_hidden: int = 32,
                 n_layers: int = 2):
        super().__init__()
        self.n_price_features = n_price_features
        self.n_graph_nodes = n_graph_nodes
        self.lstm_hidden = lstm_hidden
        self.graph_hidden = graph_hidden

        # LSTM for price time series
        self.price_lstm = nn.LSTM(
            input_size=n_price_features,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0
        )

        # Graph attention for node features
        self.graph_attention = GraphAttentionLayer(1, graph_hidden, n_graph_nodes)

        # Feature fusion
        self.fusion_layer = nn.Linear(lstm_hidden + n_graph_nodes * graph_hidden, 64)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, price_sequence: torch.Tensor,
                graph_features: torch.Tensor,
                adjacency_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining LSTM and graph attention.

        Args:
            price_sequence: (batch, sequence_length, n_price_features)
            graph_features: (batch, n_graph_nodes, 1) - current node values
            adjacency_mask: (n_graph_nodes, n_graph_nodes) - graph structure

        Returns:
            predictions: (batch,) - price predictions
            attention_weights: (batch, n_graph_nodes, n_graph_nodes)
        """
        # Process price sequence with LSTM
        lstm_out, _ = self.price_lstm(price_sequence)
        price_representation = lstm_out[:, -1, :]  # Last timestep: (batch, lstm_hidden)

        # Process graph features with attention
        graph_representation, attention_weights = self.graph_attention(
            graph_features, adjacency_mask
        )
        # Flatten graph representation: (batch, n_nodes * graph_hidden)
        graph_flat = graph_representation.view(graph_representation.size(0), -1)

        # Combine representations
        combined = torch.cat([price_representation, graph_flat], dim=1)
        fused = F.relu(self.fusion_layer(combined))

        # Generate prediction
        predictions = self.output_head(fused).squeeze(-1)

        return predictions, attention_weights


class TinyLSTM(nn.Module):
    """Simple LSTM baseline (keeping for backward compatibility)."""

    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        return self.head(out).squeeze(-1)


def create_adjacency_mask(node_list: List[str], edges: List[Tuple[str, str]]) -> torch.Tensor:
    """Create binary adjacency mask from edge list."""
    n_nodes = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    mask = torch.zeros(n_nodes, n_nodes)
    for source, target in edges:
        if source in node_to_idx and target in node_to_idx:
            i, j = node_to_idx[source], node_to_idx[target]
            mask[i, j] = 1.0

    return mask


def train_model(model, X, y, epochs=20, lr=1e-3, device=None, X_graph=None):
    """
    Enhanced training function supporting both simple and graph-enhanced models.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert inputs to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    if X_graph is not None:
        X_graph_tensor = torch.tensor(X_graph, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, verbose=False
    )
    loss_fn = nn.MSELoss()

    model.train()
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Forward pass
        if isinstance(model, GraphEnhancedLSTM) and X_graph is not None:
            predictions, _ = model(X_tensor, X_graph_tensor)
        else:
            predictions = model(X_tensor)

        # Calculate loss
        loss = loss_fn(predictions, y_tensor)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Learning rate scheduling
        scheduler.step(loss)

        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"Training completed. Best loss: {best_loss:.6f}")
    return model


def prepare_graph_data(node_features: Dict[str, np.ndarray],
                       window_indices: np.ndarray) -> np.ndarray:
    """
    Prepare graph node features aligned with LSTM windows.

    Args:
        node_features: Dictionary mapping node names to feature time series
        window_indices: Indices where LSTM windows end (for alignment)

    Returns:
        X_graph: (n_windows, n_nodes, 1) - node values for each window
    """
    node_names = list(node_features.keys())
    n_windows = len(window_indices)
    n_nodes = len(node_names)

    X_graph = np.zeros((n_windows, n_nodes, 1))

    for i, window_end_idx in enumerate(window_indices):
        for j, node_name in enumerate(node_names):
            features = node_features[node_name]
            if window_end_idx < len(features):
                X_graph[i, j, 0] = features[window_end_idx]

    return X_graph.astype(np.float32)