"""Regenerate every table and figure of the paper.

make_data.py is the single seeded source of all reported numbers: it writes results.json, keynums.json and
tables/*.tex. figures.py draws figures/*.pdf from results.json. Outputs land in paper/ (or --out).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "paper"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("make_data.py", "figures.py"):
        if (ROOT / "paper" / f).resolve() != (out / f).resolve():
            shutil.copy(ROOT / "paper" / f, out / f)
    subprocess.run([sys.executable, "make_data.py"], cwd=out, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "figures.py"], cwd=out, check=True)
    print("tables:", sorted(p.name for p in (out / "tables").glob("*.tex")))
    print("figures:", sorted(p.name for p in (out / "figures").glob("*.pdf")))


if __name__ == "__main__":
    main()
