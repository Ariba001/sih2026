"""Generate the synthetic burn-in dataset (wide schema) to data/burnin_synthetic.csv.

Usage: python data/generate_synthetic.py [--lots 20] [--parts N | --parts-min A --parts-max B | --small-lots]
                                         [--defect-rate 0.035] [--seed 7]
Default lot sizes: 50 % small space-style lots (30-80 parts) and 50 % large lots (250-500 parts).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lots", type=int, default=20)
    ap.add_argument("--parts", type=int, help="fixed parts per lot")
    ap.add_argument("--parts-min", type=int)
    ap.add_argument("--parts-max", type=int)
    ap.add_argument("--small-lots", action="store_true", help="all lots 30-80 parts")
    ap.add_argument("--defect-rate", type=float, default=0.035)
    ap.add_argument("--glitch-rate", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "burnin_synthetic.csv"))
    a = ap.parse_args()
    ppl = a.parts or ((a.parts_min, a.parts_max) if a.parts_min else ((30, 80) if a.small_lots else None))
    df = generate(a.lots, ppl, a.defect_rate, a.glitch_rate, a.seed)
    df.to_csv(a.out, index=False)
    comps = df.groupby("Component_ID")["Is_Defective"].max()
    sizes = df.groupby("Lot_ID")["Component_ID"].nunique()
    print(f"Wrote {len(df)} rows ({len(comps)} components, {len(sizes)} lots, sizes {sizes.min()}-{sizes.max()}) "
          f"to {a.out}")
    print(f"Defective components: {comps.sum()} ({100 * comps.mean():.2f}%); tester glitches on good parts: "
          f"{int(df['Injected_Glitch'].sum())}")
    print(df[df["Is_Defective"] == 1]["Defect_Type"].value_counts().to_string())


if __name__ == "__main__":
    main()
