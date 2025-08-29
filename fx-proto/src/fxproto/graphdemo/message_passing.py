# fx-proto/src/fxproto/graphdemo/message_passing.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from fxproto.graphdemo.graph_build import FinancialGraph


@dataclass
class AttentionWeights:
    """Store attention weights for interpretability."""
    source_to_target: Dict[str, Dict[str, float]]  # source -> target -> weight
    timestep: Optional[int] = None


class GraphMessagePassing:
    """
    Implements message passing on the financial graph with attention mechanisms.
    """

    def __init__(self, graph: FinancialGraph):
        self.graph = graph
        self.node_list = graph.node_list
        self.adj_matrix = graph.adjacency_matrix

    def compute_attention_weights(self, node_features: Dict[str, np.ndarray],
                                  timestep: int) -> AttentionWeights:
        """
        Compute attention weights between nodes based on current features.
        Uses a simple similarity-based attention mechanism.
        """
        n_nodes = len(self.node_list)
        attention_matrix = np.zeros((n_nodes, n_nodes))

        # Get feature values at this timestep
        features_t = {}
        for node in self.node_list:
            if node in node_features and timestep < len(node_features[node]):
                features_t[node] = node_features[node][timestep]
            else:
                features_t[node] = 0.0

        # Compute pairwise attention (similarity * edge weight)
        for i, source in enumerate(self.node_list):
            for j, target in enumerate(self.node_list):
                if i != j and self.adj_matrix[i, j] > 0:
                    # Simple attention: feature similarity * edge weight
                    similarity = self._compute_similarity(features_t[source], features_t[target])
                    edge_weight = self.adj_matrix[i, j]
                    attention_matrix[i, j] = similarity * edge_weight

        # Normalize attention weights (softmax per source node)
        attention_matrix = self._softmax_normalize_rows(attention_matrix)

        # Convert to dictionary format
        attention_dict = {}
        for i, source in enumerate(self.node_list):
            attention_dict[source] = {}
            for j, target in enumerate(self.node_list):
                if attention_matrix[i, j] > 0.01:  # Only store significant weights
                    attention_dict[source][target] = attention_matrix[i, j]

        return AttentionWeights(attention_dict, timestep)

    def _compute_similarity(self, val1: float, val2: float) -> float:
        """Compute similarity between two feature values."""
        # Simple exponential similarity based on difference
        diff = abs(val1 - val2)
        return np.exp(-diff)

    def _softmax_normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """Apply softmax normalization to each row."""
        result = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            row = matrix[i, :]
            if np.sum(row) > 0:
                exp_row = np.exp(row - np.max(row))  # Numerical stability
                result[i, :] = exp_row / np.sum(exp_row)
        return result

    def propagate_messages(self, initial_state: Dict[str, float],
                           node_features: Dict[str, np.ndarray],
                           timestep: int, steps: int = 3) -> Tuple[List[Dict[str, float]], List[AttentionWeights]]:
        """
        Propagate messages through graph with attention-based weighting.

        Returns:
            states: Node values at each propagation step
            attention_weights: Attention weights at each step
        """
        states = [initial_state.copy()]
        attention_history = []

        current_values = np.array([initial_state.get(node, 0.0) for node in self.node_list])

        for step in range(steps):
            # Compute attention weights for this step
            attention = self.compute_attention_weights(node_features, timestep)
            attention_history.append(attention)

            # Create attention-weighted adjacency matrix
            weighted_adj = self._create_attention_adjacency(attention)

            # Message passing: each node receives weighted messages from neighbors
            messages = weighted_adj.T @ current_values

            # Update node values with damping
            damping = 0.6
            update_rate = 0.2
            current_values = damping * current_values + update_rate * messages

            # Store state
            step_state = {node: current_values[i] for i, node in enumerate(self.node_list)}
            states.append(step_state)

        return states, attention_history

    def _create_attention_adjacency(self, attention: AttentionWeights) -> np.ndarray:
        """Convert attention weights to adjacency matrix format."""
        n_nodes = len(self.node_list)
        attention_adj = np.zeros((n_nodes, n_nodes))

        for source, targets in attention.source_to_target.items():
            if source in self.node_to_idx:
                source_idx = self.node_to_idx[source]
                for target, weight in targets.items():
                    if target in self.node_to_idx:
                        target_idx = self.node_to_idx[target]
                        attention_adj[source_idx, target_idx] = weight

        return attention_adj

    def analyze_influence_paths(self, scenario_id: str) -> Dict[str, any]:
        """
        Analyze how influence flows through the graph for a given scenario.
        Returns detailed metrics about the propagation.
        """
        scenario = next((s for s in self.graph.cfg.scenarios if s.id == scenario_id), None)
        if not scenario:
            return {}

        # Base state
        base_state = {node: 0.0 for node in self.node_list}

        # Apply shocks
        shocked_state = self.graph.apply_scenario_shocks(base_state, scenario_id)

        # Create dummy node features for analysis
        dummy_features = {node: np.random.normal(0, 0.1, 100) for node in self.node_list}

        # Propagate with attention
        states, attentions = self.propagate_messages(shocked_state, dummy_features, timestep=50)

        # Calculate metrics
        initial_shock_magnitude = sum(abs(v) for v in shocked_state.values())
        final_magnitude = sum(abs(v) for v in states[-1].values())

        # Find most influenced nodes
        final_impacts = {node: abs(states[-1][node]) for node in self.node_list}
        most_influenced = sorted(final_impacts.items(), key=lambda x: x[1], reverse=True)

        return {
            "scenario_id": scenario_id,
            "initial_shock": initial_shock_magnitude,
            "final_magnitude": final_magnitude,
            "amplification_factor": final_magnitude / max(initial_shock_magnitude, 1e-6),
            "most_influenced_nodes": most_influenced[:3],
            "propagation_states": states,
            "attention_weights": attentions,
        }


def simulate_scenario_impact(scenario_id: str, timesteps: int = 50) -> pd.DataFrame:
    """
    Simulate the impact of a scenario over multiple timesteps.
    Returns a DataFrame with node values over time.
    """
    graph = build_financial_graph()
    message_passer = GraphMessagePassing(graph)

    # Generate synthetic time-varying features for simulation
    node_features = {}
    for node in graph.node_list:
        # Each node gets different time-varying behavior
        trend = np.random.normal(0, 0.02, timesteps)
        noise = np.random.normal(0, 0.1, timesteps)
        node_features[node] = np.cumsum(trend) + noise

    # Run scenario at multiple timesteps
    results = []

    for t in range(10, timesteps - 10):  # Skip early/late timesteps
        # Base state for this timestep
        base_state = {node: node_features[node][t] for node in graph.node_list}

        # Apply scenario shock
        shocked_state = graph.apply_scenario_shocks(base_state, scenario_id)

        # Propagate influence
        final_states, _ = message_passer.propagate_messages(
            shocked_state, node_features, timestep=t, steps=2
        )

        # Store result
        result_row = {"timestep": t}
        for node in graph.node_list:
            result_row[f"{node}_baseline"] = base_state[node]
            result_row[f"{node}_shocked"] = final_states[-1][node]
            result_row[f"{node}_impact"] = final_states[-1][node] - base_state[node]

        results.append(result_row)

    return pd.DataFrame(results)