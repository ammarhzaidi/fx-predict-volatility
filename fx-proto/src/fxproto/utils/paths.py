from pathlib import Path

# Root = two levels up from this file (your fx-proto/ folder)
project_root = Path(__file__).resolve().parents[2]

# Main directories
config_dir = project_root / "config"
data_root = project_root / "data"
outputs_root = project_root / "src" / "outputs"

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
