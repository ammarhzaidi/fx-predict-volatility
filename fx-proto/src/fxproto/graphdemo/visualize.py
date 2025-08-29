# fx-proto/src/fxproto/graphdemo/visualize.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns

from fxproto.graphdemo.graph_build import FinancialGraph, build_financial_graph
from fxproto.graphdemo.message_passing import GraphMessagePassing, simulate_scenario_impact


def plot_graph_network(graph: FinancialGraph, node_values: Optional[Dict[str, float]] = None,
                       attention_weights: Optional[Dict[str, Dict[str, float]]] = None,
                       save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
    """
    Plot the financial graph network with optional node values and attention weights.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Network structure with node values
    pos = nx.spring_layout(graph.nx_graph, seed=42, k=2, iterations=50)

    # Node colors based on values
    if node_values:
        values = [node_values.get(node, 0) for node in graph.node_list]
        max_abs_val = max(abs(v) for v in values) if values else 1
        node_colors = [v / max(max_abs_val, 1e-6) for v in values]
    else:
        node_colors = [0.5] * len(graph.node_list)

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        graph.nx_graph, pos, ax=ax1,
        node_color=node_colors,
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        node_size=1500,
        alpha=0.8
    )

    # Draw edges with weights
    edge_weights = [graph.nx_graph[u][v]['weight'] for u, v in graph.nx_graph.edges()]
    nx.draw_networkx_edges(
        graph.nx_graph, pos, ax=ax1,
        width=[w * 3 for w in edge_weights],
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20
    )

    # Node labels
    nx.draw_networkx_labels(graph.nx_graph, pos, ax=ax1, font_size=10, font_weight='bold')

    ax1.set_title("Financial Knowledge Graph")
    ax1.axis('off')

    # Add colorbar if we have node values
    if node_values:
        plt.colorbar(nodes, ax=ax1, label="Node Value")

    # Right plot: Attention heatmap
    if attention_weights:
        plot_attention_heatmap(graph, attention_weights, ax=ax2)
    else:
        # Show adjacency matrix as fallback
        plot_adjacency_heatmap(graph, ax=ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graph visualization saved to {save_path}")

    return fig


def plot_attention_heatmap(graph: FinancialGraph, attention_weights: Dict[str, Dict[str, float]], ax=None):
    """Plot attention weights as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create attention matrix
    n_nodes = len(graph.node_list)
    attention_matrix = np.zeros((n_nodes, n_nodes))

    for source, targets in attention_weights.items():
        if source in graph.node_to_idx:
            i = graph.node_to_idx[source]
            for target, weight in targets.items():
                if target in graph.node_to_idx:
                    j = graph.node_to_idx[target]
                    attention_matrix[i, j] = weight

    # Plot heatmap
    sns.heatmap(
        attention_matrix,
        xticklabels=graph.node_list,
        yticklabels=graph.node_list,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )

    ax.set_title("Attention Weights")
    ax.set_xlabel("Target Node")
    ax.set_ylabel("Source Node")


def plot_adjacency_heatmap(graph: FinancialGraph, ax=None):
    """Plot adjacency matrix as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        graph.adjacency_matrix,
        xticklabels=graph.node_list,
        yticklabels=graph.node_list,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        ax=ax,
        cbar_kws={'label': 'Edge Weight'}
    )

    ax.set_title("Graph Adjacency Matrix")
    ax.set_xlabel("Target Node")
    ax.set_ylabel("Source Node")


def plot_scenario_impact_over_time(scenario_id: str, save_path: Optional[str] = None):
    """
    Plot how scenario impact evolves over time for each node.
    """
    # Simulate scenario
    results_df = simulate_scenario_impact(scenario_id, timesteps=60)

    if results_df.empty:
        print(f"No results for scenario {scenario_id}")
        return None

    # Get node names from columns
    impact_cols = [col for col in results_df.columns if col.endswith('_impact')]
    node_names = [col.replace('_impact', '') for col in impact_cols]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot 1: Impact magnitude over time
    ax = axes[0]
    for col in impact_cols:
        ax.plot(results_df['timestep'], results_df[col], label=col.replace('_impact', ''), linewidth=2)
    ax.set_title("Node Impact Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Impact Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Total system impact
    ax = axes[1]
    total_impact = results_df[[col for col in impact_cols]].abs().sum(axis=1)
    ax.plot(results_df['timestep'], total_impact, 'r-', linewidth=3, label='Total System Impact')
    ax.set_title("Total System Impact")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total Impact")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Node comparison (final impact)
    ax = axes[2]
    final_impacts = results_df[impact_cols].iloc[-1].abs()
    final_impacts.index = node_names
    bars = ax.bar(range(len(final_impacts)), final_impacts.values)
    ax.set_xticks(range(len(final_impacts)))
    ax.set_xticklabels(node_names, rotation=45)
    ax.set_title("Final Impact by Node")
    ax.set_ylabel("Final Impact Magnitude")

    # Color bars by magnitude
    max_impact = final_impacts.max()
    for i, bar in enumerate(bars):
        color_intensity = final_impacts.iloc[i] / max_impact if max_impact > 0 else 0
        bar.set_color(plt.cm.Reds(color_intensity))

    # Plot 4: Impact distribution
    ax = axes[3]
    all_impacts = results_df[impact_cols].values.flatten()
    all_impacts = all_impacts[~np.isnan(all_impacts)]
    ax.hist(all_impacts, bins=30, alpha=0.7, edgecolor='black')
    ax.set_title("Impact Distribution")
    ax.set_xlabel("Impact Value")
    ax.set_ylabel("Frequency")
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax.legend()

    plt.suptitle(f"Scenario Analysis: {scenario_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Scenario impact plot saved to {save_path}")

    return fig


def plot_price_forecast_with_graph_context(df: pd.DataFrame, predictions: np.ndarray,
                                           graph_features: Dict[str, np.ndarray],
                                           save_path: Optional[str] = None):
    """
    Plot price forecast with graph node features for context.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Plot 1: Price and forecast
    ax = axes[0]
    ax.plot(df.index, df['Close'], label='Actual Price', alpha=0.8, linewidth=2)

    # Align predictions with dataframe index
    pred_start = len(df) - len(predictions)
    if pred_start >= 0:
        pred_index = df.index[pred_start:pred_start + len(predictions)]
        ax.plot(pred_index, predictions, 'r--', label='LSTM Forecast', linewidth=2)

    ax.set_title("EUR/USD Price Forecast")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Key graph node features
    ax = axes[1]
    important_nodes = ['ECB', 'Fed', 'InterestRate']
    colors = ['blue', 'green', 'orange']

    for i, node in enumerate(important_nodes):
        if node.lower() in [k.lower() for k in graph_features.keys()]:
            # Find matching key (case insensitive)
            actual_key = next(k for k in graph_features.keys() if k.lower() == node.lower())
            feature_values = graph_features[actual_key]

            if len(feature_values) == len(df):
                ax.plot(df.index, feature_values, label=f'{node} Sentiment',
                        color=colors[i], alpha=0.7)

    ax.set_title("Key Node Features")
    ax.set_ylabel("Feature Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Graph influence score
    ax = axes[2]
    # Create synthetic influence score from multiple node features
    influence_score = np.zeros(len(df))
    for node_features in graph_features.values():
        if len(node_features) == len(df):
            influence_score += np.abs(node_features) * 0.2

    ax.plot(df.index, influence_score, 'purple', label='Graph Influence Score', linewidth=2)
    ax.set_title("Overall Graph Influence")
    ax.set_xlabel("Date")
    ax.set_ylabel("Influence Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Forecast with graph context saved to {save_path}")

    return fig


def create_dashboard_plots(scenario_id: str, df: pd.DataFrame, predictions: Optional[np.ndarray] = None,
                           output_dir: str = "outputs/charts"):
    """
    Create a complete set of dashboard plots for the demo.
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build graph
    graph = build_financial_graph()

    # 1. Graph network visualization
    base_state = {node: 0.0 for node in graph.node_list}
    shocked_state = graph.apply_scenario_shocks(base_state, scenario_id)

    network_path = output_path / f"graph_network_{scenario_id}.png"
    plot_graph_network(graph, shocked_state, save_path=str(network_path))

    # 2. Scenario impact over time
    impact_path = output_path / f"scenario_impact_{scenario_id}.png"
    plot_scenario_impact_over_time(scenario_id, save_path=str(impact_path))

    # 3. Price forecast (if predictions provided)
    if predictions is not None and not df.empty:
        # Generate synthetic graph features for the plot
        graph_features = {}
        for node in graph.node_list:
            if node in ['ECB', 'Fed', 'InterestRate']:
                # Create plausible synthetic features aligned with price data
                price_change = df['Close'].pct_change().rolling(10).mean()
                noise = np.random.normal(0, 0.05, len(df))
                graph_features[node] = price_change.values + noise

        forecast_path = output_path / f"forecast_with_graph_context.png"
        plot_price_forecast_with_graph_context(
            df, predictions, graph_features, save_path=str(forecast_path)
        )

    print(f"Dashboard plots created in {output_dir}")
    return {
        "network_plot": network_path,
        "impact_plot": impact_path,
        "forecast_plot": forecast_path if predictions is not None else None
    }