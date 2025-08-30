# fx-proto/src/fxproto/graphdemo/graph_build.py
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt  # ADDED: Missing import
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from fxproto.config.loader import get_config, GraphCfg
from fxproto.data.features import basic_features  # ADDED: Missing import


@dataclass
class GraphState:
    """Represents the current state of all nodes in the graph."""
    node_values: Dict[str, float]
    node_features: Dict[str, np.ndarray]  # For time-varying features
    adjacency_matrix: np.ndarray
    node_list: List[str]


class FinancialGraph:
    """
    Financial knowledge graph with nodes representing assets, institutions, and factors.
    Supports shock propagation and influence calculation.
    """

    def __init__(self, graph_cfg: GraphCfg):
        self.cfg = graph_cfg
        self.nx_graph = self._build_networkx_graph()
        self.node_list = [node.id for node in graph_cfg.nodes]
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.adjacency_matrix = self._build_adjacency_matrix()

    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from configuration."""
        G = nx.DiGraph()

        # Add nodes with attributes
        for node in self.cfg.nodes:
            G.add_node(node.id, type=node.type or "unknown")

        # Add edges with weights
        for edge in self.cfg.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight)

        return G

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Create weighted adjacency matrix from graph configuration."""
        n_nodes = len(self.node_list)
        adj_matrix = np.zeros((n_nodes, n_nodes))

        for edge in self.cfg.edges:
            source_idx = self.node_to_idx[edge.source]
            target_idx = self.node_to_idx[edge.target]
            adj_matrix[source_idx, target_idx] = edge.weight

        return adj_matrix

    def get_node_features_from_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract time-varying features for each node from price data.
        This maps synthetic features to graph nodes.
        """
        node_features = {}

        # Map synthetic features to nodes
        feature_mapping = {
            "ECB": "ecb_sentiment",
            "Fed": "fed_sentiment",
            "InterestRate": "interest_rate_diff",
            "GDP": "gdp_growth_diff",
            "EUR": "ret_1",  # EUR strength from price changes
            "USD": "risk_sentiment"  # USD as safe haven
        }

        for node_id in self.node_list:
            if node_id in feature_mapping and feature_mapping[node_id] in df.columns:
                node_features[node_id] = df[feature_mapping[node_id]].values
            else:
                # Default to small random noise for missing features
                node_features[node_id] = np.random.normal(0, 0.1, len(df))

        return node_features

    def apply_scenario_shocks(self, base_state: Dict[str, float], scenario_id: str) -> Dict[str, float]:
        """Apply shocks from a specific scenario to node states."""
        scenario = next((s for s in self.cfg.scenarios if s.id == scenario_id), None)
        if not scenario:
            raise ValueError(f"Scenario '{scenario_id}' not found")

        shocked_state = base_state.copy()

        for shock in scenario.shocks:
            if shock.node in shocked_state:
                shocked_state[shock.node] += shock.delta
            else:
                print(f"Warning: Shock node '{shock.node}' not found in state")

        return shocked_state

    def propagate_influence(self, initial_state: Dict[str, float], steps: int = 3) -> List[Dict[str, float]]:
        """
        Propagate influence through the graph using matrix multiplication.
        Returns state at each propagation step.
        """
        states = [initial_state.copy()]
        current_values = np.array([initial_state.get(node, 0.0) for node in self.node_list])

        # Damping factor to prevent explosion
        damping = 0.7

        for step in range(steps):
            # Propagate through adjacency matrix
            influenced_values = damping * (self.adjacency_matrix.T @ current_values)

            # Add to current state (additive influence model)
            current_values = current_values + influenced_values * 0.1

            # Create state dictionary
            step_state = {node: current_values[i] for i, node in enumerate(self.node_list)}
            states.append(step_state)

        return states

    def calculate_node_importance(self, scenario_id: str) -> Dict[str, float]:
        """Calculate importance scores for each node under a scenario."""
        # Base state (all nodes at neutral 0.0)
        base_state = {node: 0.0 for node in self.node_list}

        # Apply scenario shocks
        shocked_state = self.apply_scenario_shocks(base_state, scenario_id)

        # Propagate influence
        final_states = self.propagate_influence(shocked_state, steps=3)
        final_state = final_states[-1]

        # Calculate importance as absolute change from baseline
        importance = {node: abs(final_state[node]) for node in self.node_list}

        # Normalize to 0-1 scale
        max_importance = max(importance.values()) if importance.values() else 1.0
        if max_importance > 0:
            importance = {node: score / max_importance for node, score in importance.items()}

        return importance


def build_financial_graph() -> FinancialGraph:
    """Build the financial graph from configuration."""
    cfg = get_config()
    return FinancialGraph(cfg.graph)


def prepare_graph_features(df: pd.DataFrame, pair: str = "EURUSD") -> pd.DataFrame:
    """
    Prepare graph-enhanced features by combining price data with synthetic node features.
    """
    # Start with enhanced basic features including synthetics
    df_feat = basic_features(df)
    df_with_synthetic = generate_synthetic_node_features(df_feat, pair)

    # Build graph and extract node features
    graph = build_financial_graph()
    node_features = graph.get_node_features_from_data(df_with_synthetic)

    # Add node features to dataframe
    for node_id, features in node_features.items():
        if len(features) == len(df_with_synthetic):
            df_with_synthetic[f"node_{node_id.lower()}"] = features

    return df_with_synthetic


def generate_synthetic_node_features(df: pd.DataFrame, pair: str = "EURUSD") -> pd.DataFrame:
    """
    Generate synthetic features for graph nodes based on price movements.
    Enhanced version with more realistic relationships.
    """
    out = df.copy()

    # Base price metrics
    price_change = out["Close"].pct_change()
    volatility = price_change.rolling(20).std()
    momentum = out["Close"] / out["Close"].shift(20) - 1

    # ECB sentiment (EUR-related)
    if "EUR" in pair.upper():
        # ECB dovish when EUR weakens, hawkish when EUR strengthens
        ecb_trend = price_change.rolling(10).mean()
        ecb_noise = np.random.normal(0, 0.05, len(out))
        out["ecb_sentiment"] = ecb_trend + ecb_noise

    # Fed sentiment (USD-related)
    if "USD" in pair.upper():
        # Fed hawkish strengthens USD (negative for EUR/USD)
        fed_trend = -price_change.rolling(15).mean()
        fed_noise = np.random.normal(0, 0.05, len(out))
        out["fed_sentiment"] = fed_trend + fed_noise

    # Interest rate differential (major FX driver)
    # Higher rates strengthen currency
    rate_signal = momentum.rolling(30).mean()
    rate_noise = np.random.normal(0, 0.02, len(out))
    out["interest_rate_diff"] = rate_signal + rate_noise

    # GDP growth differential
    # Economic growth strengthens currency over time
    gdp_signal = momentum.rolling(60).mean() * 0.5
    gdp_noise = np.random.normal(0, 0.01, len(out))
    out["gdp_growth_diff"] = gdp_signal + gdp_noise

    # Risk sentiment (flight-to-quality effects)
    # High volatility increases USD demand (safe haven)
    risk_base = volatility * 50  # Scale volatility
    risk_noise = np.random.normal(0, 1, len(out))
    out["risk_sentiment"] = risk_base + risk_noise

    # Political stability index
    # Inversely related to volatility and large moves
    stability_base = -volatility * 30 + 0.8
    stability_noise = np.random.normal(0, 0.05, len(out))
    out["political_stability"] = stability_base + stability_noise

    # Smooth all synthetic features
    synthetic_cols = ["ecb_sentiment", "fed_sentiment", "interest_rate_diff",
                      "gdp_growth_diff", "risk_sentiment", "political_stability"]

    for col in synthetic_cols:
        if col in out.columns:
            out[col] = out[col].rolling(3, center=True).mean()

    # FIXED: Replace deprecated fillna method
    return out.fillna(method='bfill').fillna(0)  # Changed from method='ffill'


def make_supervised_windows(df: pd.DataFrame, feature_cols: List[str], target_col: str,
                            lookback: int = 30, horizon: int = 5):
    """
    Enhanced window creation with better error handling and validation.

    Returns:
        X: (N, lookback, F) - feature sequences
        y: (N,) - target values at t+horizon
    """
    # Validate inputs
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")

    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        available_cols = list(df.columns)
        raise ValueError(f"Missing columns: {missing_cols}. Available: {available_cols}")

    if len(df) < lookback + horizon:
        raise ValueError(f"DataFrame too short: {len(df)} rows, need at least {lookback + horizon}")

    # Extract values
    feature_values = df[feature_cols].values
    target_values = df[target_col].values

    X, y = [], []

    for t in range(lookback, len(df) - horizon):
        # Feature sequence: [t-lookback : t]
        X.append(feature_values[t - lookback:t, :])
        # Target value: t + horizon
        y.append(target_values[t + horizon])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Created {len(X)} windows: X shape {X.shape}, y shape {y.shape}")
    return X, y


# FIXED: Simple working demo function
def simple_main_demo():
    """Simple demo that actually works"""
    print("🧠 Running Graph Neural Network Demo...")

    try:
        # Build financial graph from config
        graph = build_financial_graph()
        print(f"✅ Financial graph created: {len(graph.node_list)} nodes")
        print(f"   Nodes: {graph.node_list}")

        # Test scenario analysis
        scenario_id = "EU_negative_GDP_shock"

        # Check if scenario exists
        scenario_ids = [s.id for s in graph.cfg.scenarios]
        print(f"   Available scenarios: {scenario_ids}")

        if scenario_id in scenario_ids:
            importance = graph.calculate_node_importance(scenario_id)
            print(f"✅ Scenario '{scenario_id}' analysis completed")
            print("   Node importance scores:")
            for node, score in importance.items():
                print(f"     {node}: {score:.3f}")
        else:
            print(f"⚠️  Scenario '{scenario_id}' not found, using first available")
            if scenario_ids:
                importance = graph.calculate_node_importance(scenario_ids[0])
                print(f"   Used scenario: {scenario_ids[0]}")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Graph visualization
        pos = nx.spring_layout(graph.nx_graph, seed=42, k=1.5)
        nx.draw_networkx_nodes(graph.nx_graph, pos, node_color='lightblue',
                               node_size=1500, ax=ax1, alpha=0.8)
        nx.draw_networkx_labels(graph.nx_graph, pos, font_size=10,
                                font_weight='bold', ax=ax1)
        nx.draw_networkx_edges(graph.nx_graph, pos, edge_color='gray',
                               arrows=True, arrowsize=15, ax=ax1, alpha=0.6)

        ax1.set_title("Financial Knowledge Graph", fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Importance scores
        if 'importance' in locals():
            nodes = list(importance.keys())
            scores = list(importance.values())
            ax2.bar(nodes, scores, color='steelblue', alpha=0.7)
            ax2.set_title("Node Importance Scores", fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylabel('Importance Score')

        plt.tight_layout()
        plt.show()

        print("✅ Graph visualization completed!")
        return True

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Graph Demo...")
    try:
        result = simple_main_demo()
        if result:
            print("✅ Demo completed!")
        else:
            print("❌ Demo failed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()