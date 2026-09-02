#!/usr/bin/env python3
"""Independent structural and numerical checks for the transition supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd


SEEDS = {3, 42, 99, 123, 456}
POLICIES = {"legacy_sampled_eer", "full_dev_fmr_0.1pct", "full_dev_fmr_0.01pct"}
CURRENT = ["Original", "15", "11", "10", "9", "8", "7", "6", "5"]
COMBINED = ["Original", "95", "90", "80", "70", "50", "30", "15", "13", "11", "10", "9", "8", "7", "6", "5"]
GRIDS = {
    "canonical_primary_grid": CURRENT,
    "expanded_legacy_reuse_sensitivity_grid": COMBINED,
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def no_duplicates(frame: pd.DataFrame, keys: Sequence[str]) -> bool:
    return bool(not frame.duplicated(list(keys)).any())


def first_common(summary: pd.DataFrame, grid: str, policy: str) -> str:
    rows = summary[
        (summary["transition_grid"] == grid)
        & (summary["threshold_policy"] == policy)
        & summary["all_five_pointwise_cis_above_zero"].astype(bool)
    ]
    return "" if rows.empty else str(rows.iloc[0]["contrast"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    source = analysis_dir / "outputs" / "supplemental_transition"
    target = analysis_dir / "outputs" / "supplemental_validation" / "validation_report.json"

    paths = {
        "genuine_metrics": source / "genuine_metrics_subject_bootstrap.csv",
        "genuine_transitions": source / "adjacent_paired_fnmr_tar_subject_bootstrap.csv",
        "transition_summary": source / "cross_seed_transition_summary.csv",
        "fmr_metrics": source / "fmr_subject_bootstrap_topk_sensitivity.csv",
        "fmr_transitions": source / "adjacent_paired_fmr_subject_bootstrap.csv",
        "first_comparison": source / "first_version_point_estimate_comparison.csv",
        "reference_hard_pairs": source / "reference_mined_hard_top10_pairs.csv",
        "internal_validation": source / "validation_report.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing supplement outputs: {missing}")

    gm = pd.read_csv(paths["genuine_metrics"], dtype={"condition": str})
    gt = pd.read_csv(paths["genuine_transitions"], dtype={"lower_quality_condition": str, "prior_condition": str})
    ts = pd.read_csv(paths["transition_summary"], dtype={"lower_quality_condition": str, "prior_condition": str})
    fm = pd.read_csv(paths["fmr_metrics"], dtype={"condition": str})
    ft = pd.read_csv(paths["fmr_transitions"], dtype={"lower_quality_condition": str, "prior_condition": str})
    fc = pd.read_csv(paths["first_comparison"])
    rh = pd.read_csv(paths["reference_hard_pairs"], dtype={"probe_subject_id": str, "nonmate_reference_subject_id": str})
    internal = json.loads(paths["internal_validation"].read_text(encoding="utf-8"))

    expected_contrasts = {
        grid: [f"Q{condition}-{('Original' if prior == 'Original' else 'Q' + prior)}" for prior, condition in zip(conditions[:-1], conditions[1:])]
        for grid, conditions in GRIDS.items()
    }
    fmr_recomputed = fm["false_matches"].astype(float) / fm["valid_scores"].astype(float)
    first_max = float(fc["absolute_difference"].max())

    checks: Dict[str, bool] = {
        "genuine_metric_row_count_240": len(gm) == 5 * 3 * len(COMBINED),
        "genuine_transition_row_count_345": len(gt) == 5 * 3 * sum(len(values) - 1 for values in GRIDS.values()),
        "transition_summary_row_count_69": len(ts) == 3 * sum(len(values) - 1 for values in GRIDS.values()),
        "fmr_metric_row_count_1485": len(fm) == 5 * 3 * len(CURRENT) * 11,
        "fmr_transition_row_count_360": len(ft) == 5 * 3 * (len(CURRENT) - 1) * 3,
        "reference_hard_pair_row_count_68300": len(rh) == 68300,
        "genuine_metric_unique_keys": no_duplicates(gm, ["seed", "condition", "threshold_policy"]),
        "genuine_transition_unique_keys": no_duplicates(gt, ["seed", "transition_grid", "contrast", "threshold_policy"]),
        "fmr_metric_unique_keys": no_duplicates(fm, ["seed", "condition", "impostor_cohort", "threshold_policy"]),
        "fmr_transition_unique_keys": no_duplicates(ft, ["seed", "contrast", "impostor_cohort", "threshold_policy"]),
        "reference_hard_pair_unique_keys": no_duplicates(rh, ["seed", "probe_path", "hard_rank"]),
        "reference_hard_pairs_are_nonmates": bool((rh["probe_subject_id"].astype(str) != rh["nonmate_reference_subject_id"].astype(str)).all()),
        "reference_hard_ranks_are_one_to_ten": set(rh["hard_rank"].astype(int)) == set(range(1, 11)),
        "five_seeds_everywhere": all(set(frame["seed"].astype(int)) == SEEDS for frame in [gm, gt, fm, ft]),
        "three_policies_everywhere": all(set(frame["threshold_policy"].astype(str)) == POLICIES for frame in [gm, gt, ts, fm, ft]),
        "combined_condition_grid_complete": set(gm["condition"].astype(str)) == set(COMBINED),
        "primary_contrasts_complete": set(gt[gt["transition_grid"] == "canonical_primary_grid"]["contrast"].astype(str)) == set(expected_contrasts["canonical_primary_grid"]),
        "expanded_contrasts_complete": set(gt[gt["transition_grid"] == "expanded_legacy_reuse_sensitivity_grid"]["contrast"].astype(str)) == set(expected_contrasts["expanded_legacy_reuse_sensitivity_grid"]),
        "subject_cluster_for_all_intervals": all(frame["bootstrap_cluster"].eq("evaluation probe_subject_id").all() for frame in [gm, gt, fm, ft]),
        "five_thousand_resamples": all((frame["bootstrap_iterations"] == 5000).all() for frame in [gm, gt, fm, ft]),
        "finite_genuine_metrics": np.isfinite(gm[["FNMR", "TAR", "attempt_level_rejection_rate"]].to_numpy(dtype=float)).all(),
        "finite_genuine_deltas": np.isfinite(gt[["paired_FNMR_delta", "paired_TAR_delta"]].to_numpy(dtype=float)).all(),
        "finite_fmr_metrics": np.isfinite(fm["pair_weighted_FMR"].to_numpy(dtype=float)).all(),
        "finite_fmr_deltas": np.isfinite(ft["paired_FMR_delta"].to_numpy(dtype=float)).all(),
        "fmr_point_estimates_recompute": bool(np.max(np.abs(fmr_recomputed - fm["pair_weighted_FMR"].astype(float))) <= 1e-15),
        "tar_is_one_minus_fnmr": bool(np.max(np.abs(gm["TAR"].astype(float) - (1.0 - gm["FNMR"].astype(float)))) <= 1e-15),
        "paired_tar_is_negative_fnmr": bool(np.max(np.abs(gt["paired_TAR_delta"].astype(float) + gt["paired_FNMR_delta"].astype(float))) <= 1e-15),
        "first_version_points_match": first_max <= 1e-12,
        "wide_original_validated": bool(internal["checks"]["wide_original_reproduced"]),
        "q13_validated": bool(internal["checks"]["q13_available_and_shared_scores_reproduced"]),
        "legacy_primary_first_common_q7_q8": first_common(ts, "canonical_primary_grid", "legacy_sampled_eer") == "Q7-Q8",
        "fmr_0_1_primary_first_common_q7_q8": first_common(ts, "canonical_primary_grid", "full_dev_fmr_0.1pct") == "Q7-Q8",
        "fmr_0_01_primary_first_common_original_q15": first_common(ts, "canonical_primary_grid", "full_dev_fmr_0.01pct") == "Q15-Original",
        "exactly_three_positive_adjacent_fmr_cells": int(ft["FMR_ci_excludes_zero_positive"].astype(bool).sum()) == 3,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Independent supplement validation failed: {failed}")

    result = {
        "validation_status": "PASS",
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "checks": {name: bool(value) for name, value in checks.items()},
        "first_version_maximum_absolute_point_difference": first_max,
        "primary_grid_first_common_transition": {
            policy: first_common(ts, "canonical_primary_grid", policy)
            for policy in sorted(POLICIES)
        },
        "expanded_grid_first_common_transition": {
            policy: first_common(ts, "expanded_legacy_reuse_sensitivity_grid", policy)
            for policy in sorted(POLICIES)
        },
        "source_sha256": {name: digest(path) for name, path in paths.items()},
    }
    atomic_json(target, result)
    print(f"Independent validation passed: {len(checks)}/{len(checks)} checks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
