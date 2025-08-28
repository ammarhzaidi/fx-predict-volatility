from pathlib import Path
import shutil

def main(apply: bool = False):
    root = Path(__file__).resolve().parent
    canonical = root / "fx-proto" / "data"
    stray_candidates = [
        root / "fx-proto" / "scripts" / "fx-proto" / "data",
        root / "fx-proto" / "src" / "fxproto" / "data" / "fx-proto" / "data",
    ]

    moved, skipped = [], []

    print(f"[info] Canonical data dir: {canonical}")
    canonical.mkdir(parents=True, exist_ok=True)

    for stray in stray_candidates:
        if not stray.exists():
            print(f"[ok] Not found (good): {stray}")
            continue

        print(f"[warn] Found stray data folder: {stray}")
        for sub in ["raw", "processed", "external"]:
            sdir = stray / sub
            if not sdir.exists():
                continue
            tdir = canonical / sub
            tdir.mkdir(parents=True, exist_ok=True)

            for file in sdir.rglob("*"):
                if file.is_file():
                    dst = tdir / file.name
                    if dst.exists():
                        skipped.append((file, dst))
                        print(f"  [skip] {file} -> {dst} (already exists)")
                    else:
                        moved.append((file, dst))
                        print(f"  [move] {file} -> {dst}")
                        if apply:
                            shutil.move(str(file), str(dst))

        if apply:
            shutil.rmtree(stray)
            print(f"  [clean] Removed stray: {stray}")

    print("\n=== SUMMARY ===")
    print(f"Would move {len(moved)} files, skip {len(skipped)}.")
    if not apply:
        print("Dry run only. Run with apply=True to actually move.")

if __name__ == "__main__":
    main(apply=False)  # change to True once you're happy
