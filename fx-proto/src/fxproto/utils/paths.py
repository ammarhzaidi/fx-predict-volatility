from pathlib import Path

# Root = three levels up from this file to get to fx-proto/ folder
# Current: fx-proto/src/fxproto/utils/paths.py
# Need: fx-proto/
project_root = Path(__file__).resolve().parents[3]

# Main directories - FIXED: Config is at project root level, not under src
config_dir = project_root / "config"  # fx-proto/config/
data_root = project_root / "data"     # fx-proto/data/
outputs_root = project_root / "src" / "fxproto" / "outputs"  # fx-proto/src/fxproto/outputs/

def ensure_dirs():
    """Make sure key directories exist (data/raw, data/processed, outputs)."""
    for d in [data_root / "raw", data_root / "processed", data_root / "external",
              outputs_root / "charts", outputs_root / "reports"]:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Project root:", project_root)
    print("Config dir:", config_dir)
    print("Data root:", data_root)
    print("Outputs root:", outputs_root)
    ensure_dirs()