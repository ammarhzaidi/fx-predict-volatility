# fx-proto/run_gui.py
"""
Simple launcher for the complete GUI
Just run: python run_gui.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit GUI"""

    print("🚀 FX Forecasting Prototype - Complete GUI")
    print("=" * 50)
    print("Starting web interface...")
    print("The app will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 50)

    # Get the script path
    project_root = Path(__file__).resolve().parent
    ui_script = project_root / "scripts" / "ui_app.py"

    if not ui_script.exists():
        print(f"❌ GUI script not found at {ui_script}")
        print("Please make sure ui_app.py is in the scripts/ folder.")
        return

    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(ui_script),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]

        subprocess.run(cmd, cwd=str(project_root))

    except KeyboardInterrupt:
        print("\n👋 GUI stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        print("\nTry running manually:")
        print(f"streamlit run {ui_script}")


if __name__ == "__main__":
    main()