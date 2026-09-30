"""Optional Module A demo on real pre/post burn-in wafer data (SparrowChang/Machine-learning-project).

The repository has NO license (docs/DATASETS.md §3.1): files are downloaded at runtime into
data/external/sparrowchang/ and are never committed. Cite the repo if you show results.

Finding: prebi.csv and postbi.csv share no (WaferID, X, Y) dies, so per-die drift cannot be built.
Demo instead (non-circular, wafer = lot):
  1. pre-BI lot-relative DPAT per wafer on log-free robust z of every BI test (k = 4.5 and 6)
  2. post-BI failure = any BI test outside test_limits.csv
  3. does a wafer's pre-BI DPAT outlier rate track its post-BI failure rate?  (Spearman, AEC-Q002-like)
  4. post-BI: how many within-limit dies are DPAT outliers (latent-defect candidates static limits miss)

Usage: python data/loaders/sparrowchang.py
"""
import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features import robust_sigma  # noqa: E402

BASE = "https://raw.githubusercontent.com/SparrowChang/Machine-learning-project/HEAD/Datasets/"
OUT_DIR = ROOT / "data" / "external" / "sparrowchang"


def download() -> bool:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in ["prebi.csv", "postbi.csv", "test_limits.csv"]:
        p = OUT_DIR / f
        if p.exists() and p.stat().st_size > 0:
            continue
        try:
            urllib.request.urlretrieve(BASE + f, p)
        except Exception as e:
            print(f"Download of {f} failed ({e}); fetch {BASE}{f} manually into {OUT_DIR}.")
            return False
    return True


def wafer_max_z(df: pd.DataFrame, tests: list) -> pd.Series:
    def rz(s):
        return (s - s.median()) / robust_sigma(s.dropna().values) if s.notna().sum() >= 10 else s * 0
    z = pd.concat({t: df.groupby("WaferID")[t].transform(rz).abs() for t in tests}, axis=1)
    return z.max(axis=1)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    if not download():
        print("Skipping SparrowChang demo.")
        return
    pre = pd.read_csv(OUT_DIR / "prebi.csv")
    post = pd.read_csv(OUT_DIR / "postbi.csv")
    lim = pd.read_csv(OUT_DIR / "test_limits.csv").rename(columns={"Unnamed: 0": "test"}).set_index("test")
    paired = len(pre.merge(post, on=["WaferID", "X", "Y"]))
    tests_post = [c for c in post.columns if c.startswith("BI_TEST_") and c in lim.index]
    tests_pre = [c for c in pre.columns if c.startswith("BI_TEST_")]
    v = post[tests_post]
    per_test = (v < lim.loc[tests_post, "Lower Limit"].values) | (v > lim.loc[tests_post, "Upper Limit"].values)
    usable = per_test.columns[per_test.mean() <= 0.2]   # tests failing > 20 % of dies: limits not applicable
    fail = per_test[usable].any(axis=1)
    post = post.assign(post_fail=fail)
    res = {"paired_dies": paired, "pre_dies": len(pre), "post_dies": len(post),
           "tests_used_for_labels": len(usable), "tests_excluded_high_fail": int(len(tests_post) - len(usable)),
           "post_fail_rate": float(fail.mean()), "wafers": int(post["WaferID"].nunique())}
    pre = pre.assign(maxz=wafer_max_z(pre, tests_pre))
    post = post.assign(maxz=wafer_max_z(post, tests_post))
    for k in (4.5, 6.0):
        w = pd.DataFrame({"pre_outlier_rate": pre.groupby("WaferID")["maxz"].apply(lambda s: (s >= k).mean()),
                          "post_fail_rate": post.groupby("WaferID")["post_fail"].mean()}).dropna()
        rho, p = spearmanr(w["pre_outlier_rate"], w["post_fail_rate"])
        within = (~post["post_fail"]) & (post["maxz"] >= k)
        res[f"k={k}"] = {"wafers_compared": len(w), "spearman_rho": float(rho), "p_value": float(p),
                         "post_within_limit_dpat_outliers": int(within.sum()),
                         "post_fails_also_dpat_outliers": float((post.loc[fail, "maxz"] >= k).mean())}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "real_data_sparrowchang.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
