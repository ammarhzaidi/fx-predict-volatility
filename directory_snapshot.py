# scripts/print_tree.py
from pathlib import Path

def write_tree(root: Path, file, prefix: str = ""):
    entries = sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    for idx, entry in enumerate(entries):
        connector = "└── " if idx == len(entries) - 1 else "├── "
        file.write(prefix + connector + entry.name + ("\n" if entry.is_file() else "/\n"))
        if entry.is_dir():
            extension = "    " if idx == len(entries) - 1 else "│   "
            write_tree(entry, file, prefix + extension)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]  # go up to project root
    output_file = project_root / "project_tree.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(project_root.name + "/\n")
        write_tree(project_root, f)
    print(f"[OK] Project tree written to {output_file}")
