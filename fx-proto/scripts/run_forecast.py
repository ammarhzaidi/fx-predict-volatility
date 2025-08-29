# fx-proto/scripts/run_forecast.py
"""
Enhanced forecast script that integrates the complete pipeline:
Data → Features → Graph → Model → Predictions → Visualization
"""

import sys
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# Import our modules
from fxproto.config.loader import get_config, resolved_outputs_dir
from fxproto.data.preprocess import load_or_fetch, train_test_split_by_dates, scale_train_apply_test
from fxproto.data.features import basic_features, generate_synthetic_node_features, make_supervised_windows
from fxproto.models.lstm import TinyLSTM, GraphEnhancedLSTM, train_model, create_adjacency_mask, prepare_graph_data
from fxproto.models.infer import predict_series, add_predictions_to_frame
from fxproto.graphdemo.graph_build import build_financial_graph
from fxproto.graphdemo.visualize import create_dashboard_plots


def main(use_graph: bool = True, save_plots: bool = True):
    """
    Main forecast pipeline.
    """
    print("🚀 FX Prototype - Enhanced Forecasting Pipeline")
    print("=" * 50)

    try:
        # 1. Load configuration
        print("📋 Loading configuration...")
        cfg = get_config()
        pair = cfg.settings.pair
        interval = cfg.data.interval
        dates = cfg.settings.dates
        horizon = cfg.settings.horizon

        print(f"   • Pair: {pair}")
        print(f"   • Horizon: {horizon} steps")
        print(f"   • Data source: {cfg.data.source}")

        # Setup directories
        out_dir = resolved_outputs_dir()
        for subdir in ["charts", "reports", "models"]:
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)

        # 2. Load and enhance data
        print("\n📈 Loading and enhancing data...")
        df_raw = load_or_fetch(pair, dates.train_start, dates.test_end, interval)
        print(f"   • Loaded {len(df_raw)} raw records")

        # Apply basic features
        df_features = basic_features(df_raw)
        print(f"   • Applied basic features: {len(df_features)} records remain")

        # Add synthetic node features if using graph
        if use_graph:
            df_enhanced = generate_synthetic_node_features(df_features, pair)
            print(f"   • Added synthetic node features")
        else:
            df_enhanced = df_features

        # 3. Prepare features for modeling
        print("\n🔧 Preparing model features...")

        # Select price-based features
        price_features = ["ret_1", "ma_5", "vol_10"]
        available_price_features = [f for f in price_features if f in df_enhanced.columns]

        if use_graph:
            # Add synthetic node features
            synthetic_features = ["ecb_sentiment", "fed_sentiment", "interest_rate_diff"]
            available_synthetic = [f for f in synthetic_features if f in df_enhanced.columns]
            all_features = available_price_features + available_synthetic
        else:
            all_features = available_price_features

        print(f"   • Using features: {all_features}")
        target_col = "Close"

        # 4. Split and scale data
        print("\n✂️ Splitting and scaling data...")
        train, test = train_test_split_by_dates(df_enhanced)
        train_scaled, test_scaled, scaler = scale_train_apply_test(train, test, cols=all_features)

        print(f"   • Train samples: {len(train_scaled)}")
        print(f"   • Test samples: {len(test_scaled)}")

        # 5. Create supervised windows
        print("\n🪟 Creating supervised learning windows...")
        lookback = 30

        X_train, y_train = make_supervised_windows(
            train_scaled, all_features, target_col, lookback, horizon
        )
        X_test, y_test = make_supervised_windows(
            test_scaled, all_features, target_col, lookback, horizon
        )

        print(f"   • Train windows: X={X_train.shape}, y={y_train.shape}")
        print(f"   • Test windows: X={X_test.shape}, y={y_test.shape}")

        # 6. Prepare graph data if needed
        X_train_graph, X_test_graph, adjacency_mask = None, None, None

        if use_graph:
            print("\n🕸️ Preparing graph data...")

            # Build graph
            graph = build_financial_graph()
            print(f"   • Graph nodes: {graph.node_list}")

            # Create simple node features from synthetic data
            node_features = {}
            for node in graph.node_list:
                if node == "ECB" and "ecb_sentiment" in df_enhanced.columns:
                    node_features[node] = df_enhanced["ecb_sentiment"].values
                elif node == "Fed" and "fed_sentiment" in df_enhanced.columns:
                    node_features[node] = df_enhanced["fed_sentiment"].values
                elif node == "InterestRate" and "interest_rate_diff" in df_enhanced.columns:
                    node_features[node] = df_enhanced["interest_rate_diff"].values
                else:
                    # Default random features
                    node_features[node] = np.random.normal(0, 0.1, len(df_enhanced))

            # Create adjacency mask
            edges = [(e.source, e.target) for e in cfg.graph.edges]
            adjacency_mask = create_adjacency_mask(graph.node_list, edges)

            # Prepare aligned graph data
            train_indices = np.arange(lookback, len(train_scaled) - horizon)
            test_indices = np.arange(lookback, len(test_scaled) - horizon)

            # Adjust indices to account for train/test split
            train_offset = 0
            test_offset = len(train_scaled)

            X_train_graph = prepare_graph_data(node_features, train_indices)
            X_test_graph = prepare_graph_data(node_features, test_indices + test_offset)

            print(f"   • Train graph data: {X_train_graph.shape}")
            print(f"   • Test graph data: {X_test_graph.shape}")

        # 7. Create and train model
        print(f"\n🤖 Training {'Graph-Enhanced ' if use_graph else ''}LSTM model...")

        if use_graph and X_train_graph is not None:
            model = GraphEnhancedLSTM(
                n_price_features=len(available_price_features),
                n_graph_nodes=len(graph.node_list),
                lstm_hidden=64,
                graph_hidden=32
            )
            print("   • Using GraphEnhancedLSTM")

            # Train with graph data
            model = train_model(
                model, X_train[:, :, :len(available_price_features)], y_train,
                epochs=15, lr=1e-3, X_graph=X_train_graph
            )
        else:
            model = TinyLSTM(n_features=len(all_features), hidden=64)
            print("   • Using TinyLSTM")
            model = train_model(model, X_train, y_train, epochs=15, lr=1e-3)

        # 8. Generate predictions
        print("\n🔮 Generating predictions...")

        if use_graph and X_test_graph is not None:
            # Graph model predictions
            import torch
            model.eval()
            with torch.no_grad():
                X_test_price = torch.tensor(X_test[:, :, :len(available_price_features)], dtype=torch.float32)
                X_graph_tensor = torch.tensor(X_test_graph, dtype=torch.float32)
                predictions, attention_weights = model(X_test_price, X_graph_tensor, adjacency_mask)
                predictions = predictions.cpu().numpy()

                print(f"   • Generated {len(predictions)} predictions with attention weights")
        else:
            predictions = predict_series(model, X_test)
            print(f"   • Generated {len(predictions)} predictions")

        # 9. Evaluate and save results
        print("\n📊 Evaluating results...")

        # Align predictions with timeline
        start_idx = len(train) + lookback
        report = add_predictions_to_frame(df_enhanced, start_idx, predictions, target_col, horizon)

        # Calculate metrics on aligned data
        aligned_data = report.dropna()
        if len(aligned_data) > 0:
            mse = np.mean((aligned_data['actual'] - aligned_data['pred']) ** 2)
            mae = np.mean(np.abs(aligned_data['actual'] - aligned_data['pred']))

            # Directional accuracy
            actual_changes = aligned_data['actual'].diff().dropna()
            pred_changes = aligned_data['pred'].diff().dropna()

            if len(actual_changes) > 0:
                actual_dir = (actual_changes > 0).astype(int)
                pred_dir = (pred_changes > 0).astype(int)
                directional_accuracy = (actual_dir == pred_dir).mean()
            else:
                directional_accuracy = 0.0

            print(f"   • MSE: {mse:.6f}")
            print(f"   • MAE: {mae:.6f}")
            print(f"   • Directional Accuracy: {directional_accuracy:.3f}")
            print(f"   • Evaluated on {len(aligned_data)} aligned samples")
        else:
            print("   ⚠️ No aligned samples for evaluation")
            mse = mae = directional_accuracy = np.nan

        # Save results
        csv_path = out_dir / "reports" / f"{pair}_forecast_h{horizon}.csv"
        report.to_csv(csv_path, index=True)
        print(f"   • Saved report: {csv_path}")

        # Save model
        import torch
        model_path = out_dir / "models" / f"{pair}_model_h{horizon}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"   • Saved model: {model_path}")

        # 10. Create visualizations
        if save_plots:
            print("\n📈 Creating visualizations...")

            try:
                # Simple price and forecast plot
                plt.figure(figsize=(15, 8))

                # Plot 1: Price history and forecast
                plt.subplot(2, 1, 1)
                plt.plot(df_enhanced.index, df_enhanced['Close'],
                         label='Actual Price', alpha=0.7, linewidth=1)

                if len(report.dropna()) > 0:
                    plt.plot(report.index, report['pred'],
                             'r--', label=f'LSTM Forecast (h={horizon})', linewidth=2)

                plt.title(f"{pair} Price Forecast {'with Graph Enhancement' if use_graph else ''}")
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Plot 2: Prediction errors
                plt.subplot(2, 1, 2)
                if len(aligned_data) > 0:
                    errors = aligned_data['actual'] - aligned_data['pred']
                    plt.plot(aligned_data.index, errors, 'g-', alpha=0.6, label='Prediction Error')
                    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
                    plt.fill_between(aligned_data.index, errors, alpha=0.3)

                plt.title('Prediction Errors')
                plt.xlabel('Date')
                plt.ylabel('Error')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.tight_layout()

                chart_path = out_dir / "charts" / f"{pair}_forecast_h{horizon}.png"
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"   • Saved chart: {chart_path}")

                # Create dashboard plots if using graph
                if use_graph:
                    dashboard_plots = create_dashboard_plots(
                        scenario_id="EU_negative_GDP_shock",
                        df=df_enhanced,
                        predictions=predictions,
                        output_dir=str(out_dir / "charts")
                    )
                    print(f"   • Created dashboard plots in {out_dir / 'charts'}")

            except Exception as e:
                print(f"   ⚠️ Visualization failed: {e}")

        print("\n✅ Forecast pipeline completed successfully!")
        print(f"📁 Results saved to: {out_dir}")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FX forecast pipeline")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph enhancement")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    main(use_graph=not args.no_graph, save_plots=not args.no_plots)