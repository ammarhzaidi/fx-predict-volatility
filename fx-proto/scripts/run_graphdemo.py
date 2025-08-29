# fx-proto/scripts/run_graphdemo.py
"""
Graph demonstration script showcasing financial knowledge graph
with scenario simulation and visualization.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from fxproto.config.loader import get_config, resolved_outputs_dir
from fxproto.graphdemo.graph_build import build_financial_graph
from fxproto.graphdemo.message_passing import GraphMessagePassing, simulate_scenario_impact
from fxproto.graphdemo.visualize import (
    plot_graph_network, plot_scenario_impact_over_time, create_dashboard_plots
)


def main(scenario_id: str = "EU_negative_GDP_shock", steps: int = 3, save_plots: bool = True):
    """
    Main graph demo pipeline.
    """
    print("🕸️ FX Prototype - Financial Knowledge Graph Demo")
    print("=" * 50)

    try:
        # Setup
        cfg = get_config()
        out_dir = resolved_outputs_dir()
        (out_dir / "charts").mkdir(parents=True, exist_ok=True)
        (out_dir / "reports").mkdir(parents=True, exist_ok=True)

        print(f"📋 Configuration loaded")
        print(f"🎯 Scenario: {scenario_id}")
        print(f"📁 Output directory: {out_dir}")

        # 1. Build financial graph
        print(f"\n🏗️ Building financial knowledge graph...")
        graph = build_financial_graph()

        print(f"   • Nodes: {len(graph.node_list)} - {graph.node_list}")
        print(f"   • Edges: {len(graph.cfg.edges)}")

        # Display edge information
        for edge in graph.cfg.edges:
            print(f"     - {edge.source} → {edge.target} (weight: {edge.weight})")

        # Display scenario information
        scenario = next((s for s in graph.cfg.scenarios if s.id == scenario_id), None)
        if scenario:
            print(f"\n🎭 Scenario: {scenario.description}")
            print(f"   • Shocks:")
            for shock in scenario.shocks:
                print(f"     - {shock.node}: {shock.delta:+.2f}")
        else:
            available_scenarios = [s.id for s in graph.cfg.scenarios]
            print(f"❌ Scenario '{scenario_id}' not found!")
            print(f"Available scenarios: {available_scenarios}")
            return

        # 2. Initialize message passing
        print(f"\n🔄 Setting up message passing system...")
        message_passer = GraphMessagePassing(graph)

        # 3. Run scenario analysis
        print(f"\n⚡ Running scenario impact analysis...")

        # Base state (all nodes neutral)
        base_state = {node: 0.0 for node in graph.node_list}
        print(f"   • Base state: all nodes at 0.0")

        # Apply scenario shocks
        shocked_state = graph.apply_scenario_shocks(base_state, scenario_id)
        print(f"   • After shocks: {shocked_state}")

        # Analyze influence paths
        analysis = message_passer.analyze_influence_paths(scenario_id)

        if analysis:
            print(f"\n📊 Impact Analysis Results:")
            print(f"   • Initial shock magnitude: {analysis['initial_shock']:.4f}")
            print(f"   • Final system magnitude: {analysis['final_magnitude']:.4f}")
            print(f"   • Amplification factor: {analysis['amplification_factor']:.2f}x")

            print(f"   • Most influenced nodes:")
            for node, impact in analysis['most_influenced_nodes']:
                print(f"     - {node}: {impact:.4f}")

        # 4. Simulate scenario over time
        print(f"\n📈 Simulating scenario impact over time...")

        time_series_results = simulate_scenario_impact(scenario_id, timesteps=60)

        if not time_series_results.empty:
            print(f"   • Generated {len(time_series_results)} timestep simulations")

            # Save time series results
            results_path = out_dir / "reports" / f"scenario_{scenario_id}_timeseries.csv"
            time_series_results.to_csv(results_path, index=False)
            print(f"   • Time series saved to: {results_path}")

        # 5. Create visualizations
        if save_plots:
            print(f"\n📊 Creating visualizations...")

            # Network visualization with shock effects
            print("   • Creating network plot...")
            network_fig = plot_graph_network(
                graph,
                node_values=shocked_state,
                save_path=str(out_dir / "charts" / f"graph_network_{scenario_id}.png")
            )
            plt.close(network_fig)

            # Scenario impact over time
            print("   • Creating impact timeline...")
            impact_fig = plot_scenario_impact_over_time(
                scenario_id,
                save_path=str(out_dir / "charts" / f"scenario_impact_{scenario_id}.png")
            )
            if impact_fig:
                plt.close(impact_fig)

            # Create comprehensive dashboard
            print("   • Creating dashboard plots...")
            dashboard_plots = create_dashboard_plots(
                scenario_id=scenario_id,
                df=time_series_results if not time_series_results.empty else None,
                output_dir=str(out_dir / "charts")
            )

            print(f"   📁 Visualizations saved to: {out_dir / 'charts'}")

        # 6. Generate summary report
        print(f"\n📋 Generating summary report...")

        report_lines = [
            f"Financial Knowledge Graph Analysis Report",
            f"=" * 50,
            f"",
            f"Scenario: {scenario_id}",
            f"Description: {scenario.description if scenario else 'N/A'}",
            f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"Graph Structure:",
            f"  • Nodes: {len(graph.node_list)}",
            f"  • Edges: {len(graph.cfg.edges)}",
            f"  • Scenarios: {len(graph.cfg.scenarios)}",
            f"",
            f"Node Details:",
        ]

        for node in graph.cfg.nodes:
            report_lines.append(f"  • {node.id} ({node.type or 'unknown'})")

        report_lines.extend([
            f"",
            f"Edge Weights:",
        ])

        for edge in graph.cfg.edges:
            report_lines.append(f"  • {edge.source} → {edge.target}: {edge.weight}")

        if analysis:
            report_lines.extend([
                f"",
                f"Impact Analysis:",
                f"  • Initial shock magnitude: {analysis['initial_shock']:.4f}",
                f"  • Final magnitude: {analysis['final_magnitude']:.4f}",
                f"  • Amplification factor: {analysis['amplification_factor']:.2f}x",
                f"",
                f"Most Influenced Nodes:",
            ])

            for node, impact in analysis['most_influenced_nodes']:
                report_lines.append(f"  • {node}: {impact:.4f}")

        # Save report
        report_path = out_dir / "reports" / f"graph_analysis_{scenario_id}.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"   📄 Report saved to: {report_path}")

        print(f"\n✅ Graph demo completed successfully!")
        print(f"📁 All outputs saved to: {out_dir}")

        # Display summary
        print(f"\n📈 Quick Summary:")
        print(f"   • Scenario '{scenario_id}' analyzed")
        print(f"   • {len(graph.node_list)} nodes, {len(graph.cfg.edges)} edges")
        if analysis:
            print(f"   • {analysis['amplification_factor']:.1f}x impact amplification")
            most_affected = analysis['most_influenced_nodes'][0] if analysis['most_influenced_nodes'] else ("None", 0)
            print(f"   • Most affected node: {most_affected[0]} ({most_affected[1]:.3f})")

    except Exception as e:
        print(f"\n❌ Graph demo failed: {e}")
        import traceback
        traceback.print_exc()


def list_available_scenarios():
    """List all available scenarios from config."""
    try:
        cfg = get_config()
        scenarios = cfg.graph.scenarios

        print("📋 Available Scenarios:")
        print("-" * 30)

        for scenario in scenarios:
            print(f"🎭 {scenario.id}")
            if scenario.description:
                print(f"   Description: {scenario.description}")
            print(f"   Shocks: {len(scenario.shocks)}")
            for shock in scenario.shocks:
                print(f"     • {shock.node}: {shock.delta:+.2f}")
            print()

    except Exception as e:
        print(f"Failed to load scenarios: {e}")


def validate_graph_structure():
    """Validate the graph structure and connectivity."""
    try:
        print("🔍 Validating graph structure...")

        graph = build_financial_graph()

        # Check for isolated nodes
        isolated_nodes = []
        for node in graph.node_list:
            has_incoming = any(edge.target == node for edge in graph.cfg.edges)
            has_outgoing = any(edge.source == node for edge in graph.cfg.edges)

            if not has_incoming and not has_outgoing:
                isolated_nodes.append(node)

        if isolated_nodes:
            print(f"⚠️ Isolated nodes found: {isolated_nodes}")
        else:
            print("✅ All nodes are connected")

        # Check adjacency matrix properties
        adj_matrix = graph.adjacency_matrix
        print(f"📊 Adjacency matrix: {adj_matrix.shape}")
        print(f"   • Non-zero entries: {np.count_nonzero(adj_matrix)}")
        print(f"   • Total possible edges: {adj_matrix.size}")
        print(f"   • Density: {np.count_nonzero(adj_matrix) / adj_matrix.size:.3f}")

        # Check for self-loops
        self_loops = np.count_nonzero(np.diag(adj_matrix))
        print(f"   • Self-loops: {self_loops}")

        print("✅ Graph validation complete")

    except Exception as e:
        print(f"❌ Graph validation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run financial graph demonstration")
    parser.add_argument(
        "--scenario",
        default="EU_negative_GDP_shock",
        help="Scenario ID to simulate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of propagation steps"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate graph structure and exit"
    )

    args = parser.parse_args()

    if args.list_scenarios:
        list_available_scenarios()
    elif args.validate:
        validate_graph_structure()
    else:
        main(
            scenario_id=args.scenario,
            steps=args.steps,
            save_plots=not args.no_plots
        )