"""Run the stock MP calibrator and additionally dump RAW per-group sigma.

The stock export (``ThresholdCalibrator.export``) reduces the in-memory
per-row error records to per-bucket means (``level_mean_error``); the raw
[n_rows, n_levels] arrays are discarded. For the Fig. 3 error-tolerance
figure we need the full per-group distribution, so this wrapper patches
``export`` to first ``np.savez`` the records, then defers to the stock
export unchanged. Measurement behavior is byte-identical to the stock
script; all CLI args pass through verbatim.

Usage (from repo root, GPU node):

  RAW_SIGMA_NPZ=/path/out.npz python -u benchmark/ppl/fig3_sigma/raw_sigma_dump.py \
      <stock calibrate_mp_thresholds.py args...>

npz schema:
  "{op}.t{t}.l{layer}.errors"  float32 [n, n_levels]  raw relative-L2 sigma,
                               columns in --mp_levels order (descending, HALVED)
  "{op}.t{t}.l{layer}.metrics" float32 [n]            per-call min-max normalized
                               dispatch metric (aligned row-for-row with errors)
  "{op}.t{t}.l{layer}.true_count" int64 scalar        un-subsampled row population
  "_levels_halved"             int64 [n_levels]       the HALVED stream lengths
"""
from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import benchmark.ppl.calibrate_mp_thresholds as C  # noqa: E402

_orig_export = C.ThresholdCalibrator.export


def _export_with_raw_dump(self):
    out_path = os.environ["RAW_SIGMA_NPZ"]
    arrs = {"_levels_halved": np.asarray(self.levels, dtype=np.int64)}
    for (op, t, l), rec in sorted(self.records.items()):
        key = f"{op}.t{t}.l{l}"
        arrs[key + ".errors"] = np.concatenate(rec["errors"], axis=0).astype(np.float32)
        arrs[key + ".metrics"] = np.concatenate(rec["metrics"]).astype(np.float32)
        arrs[key + ".true_count"] = np.asarray(
            self.true_counts.get((op, t, l), 0), dtype=np.int64
        )
    np.savez_compressed(out_path, **arrs)
    n_sites = sum(1 for k in arrs if k.endswith(".errors"))
    print(f"[raw_sigma_dump] wrote {out_path}: {n_sites} call sites, "
          f"levels(halved)={list(self.levels)}", flush=True)
    return _orig_export(self)


C.ThresholdCalibrator.export = _export_with_raw_dump

if __name__ == "__main__":
    if "RAW_SIGMA_NPZ" not in os.environ:
        raise SystemExit("RAW_SIGMA_NPZ env var is required (npz output path).")
    sys.argv[0] = "calibrate_mp_thresholds.py"
    C.main()
