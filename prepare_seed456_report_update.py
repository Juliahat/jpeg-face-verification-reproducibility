#!/usr/bin/env python3
"""Prepare reproducible three-seed tables for the experiment report update."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


PROJECT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT / "experiment_report_seed42_seed123_v1"
SEED456_OUTPUTS = PROJECT / "formal_transition_seed456_gpu_v1" / "outputs"

EXISTING_METRICS = REPORT_DIR / "evaluation_only_verification_metrics.csv"
EXISTING_EVIDENCE = REPORT_DIR / "evaluation_only_adjacent_transition_evidence.csv"
SEED456_METRICS = SEED456_OUTPUTS / "seed456_transition_metrics_with_ci.csv"
SEED456_PAIRED = SEED456_OUTPUTS / "seed456_transition_paired_bootstrap.csv"
SEED456_AUDIT = SEED456_OUTPUTS / "seed456_transition_audit.json"

THREE_SEED_METRICS = REPORT_DIR / "evaluation_only_verification_metrics_three_seed.csv"
THREE_SEED_COMPARISON = REPORT_DIR / "evaluation_only_three_seed_comparison.csv"
THREE_SEED_EVIDENCE = (
    REPORT_DIR / "evaluation_only_adjacent_transition_evidence_three_seed.csv"
)
UPDATE_MANIFEST = REPORT_DIR / "seed456_report_update_manifest.json"

COMMON = ["Original", "15", "11", "10", "9", "8", "7", "6", "5"]
CONTRASTS = [
    "Q15-Original",
    "Q11-Q15",
    "Q10-Q11",
    "Q9-Q10",
    "Q8-Q9",
    "Q7-Q8",
    "Q6-Q7",
    "Q5-Q6",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
        handle.flush()
        os.fsync(handle.fileno())


def exclusive_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def normalize_seed456(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in source.to_dict(orient="records"):
        attempted = int(record["num_eval_genuine_attempted"]) + int(
            record["num_eval_impostor_attempted"]
        )
        valid = int(record["num_eval_genuine_valid"]) + int(
            record["num_eval_impostor_valid"]
        )
        rows.append(
            {
                "seed": 456,
                "condition": str(record["condition"]),
                "source_run": record["source_run"],
                "config_fingerprint": record["config_fingerprint"],
                "threshold": float(record["threshold"]),
                "threshold_source": record["threshold_source"],
                "evaluation_rows": attempted,
                "valid_similarity_rows": valid,
                "FMR": float(record["FMR"]),
                "FMR_ci_low": float(record["FMR_ci_low"]),
                "FMR_ci_high": float(record["FMR_ci_high"]),
                "FNMR": float(record["FNMR"]),
                "FNMR_ci_low": float(record["FNMR_ci_low"]),
                "FNMR_ci_high": float(record["FNMR_ci_high"]),
                "TAR": float(record["TAR"]),
                "TAR_ci_low": float(record["TAR_ci_low"]),
                "TAR_ci_high": float(record["TAR_ci_high"]),
                "TNR": float(record["TNR"]),
                "TNR_ci_low": float(record["TNR_ci_low"]),
                "TNR_ci_high": float(record["TNR_ci_high"]),
                "EER": float(record["EER"]),
                "EER_ci_low": float(record["EER_ci_low"]),
                "EER_ci_high": float(record["EER_ci_high"]),
                "mean_genuine_similarity": float(record["mean_genuine_similarity"]),
                "mean_genuine_similarity_ci_low": float(
                    record["mean_genuine_similarity_ci_low"]
                ),
                "mean_genuine_similarity_ci_high": float(
                    record["mean_genuine_similarity_ci_high"]
                ),
                "mean_impostor_similarity": float(record["mean_impostor_similarity"]),
                "mean_impostor_similarity_ci_low": float(
                    record["mean_impostor_similarity_ci_low"]
                ),
                "mean_impostor_similarity_ci_high": float(
                    record["mean_impostor_similarity_ci_high"]
                ),
                "average_probe_size_bytes": float(
                    record["average_probe_size_bytes"]
                ),
                "average_probe_size_bytes_ci_low": float(
                    record["average_probe_size_bytes_ci_low"]
                ),
                "average_probe_size_bytes_ci_high": float(
                    record["average_probe_size_bytes_ci_high"]
                ),
                "storage_saving": float(record["storage_saving_vs_original"]),
                "storage_saving_ci_low": float(
                    record["storage_saving_vs_original_ci_low"]
                ),
                "storage_saving_ci_high": float(
                    record["storage_saving_vs_original_ci_high"]
                ),
                "comparison_failure_count": int(
                    round(float(record["comparison_failure_count"]))
                ),
                "bootstrap_iterations": int(record["bootstrap_iterations"]),
                "bootstrap_unit": record["bootstrap_cluster"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    targets = [
        THREE_SEED_METRICS,
        THREE_SEED_COMPARISON,
        THREE_SEED_EVIDENCE,
        UPDATE_MANIFEST,
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite report-update artifacts: {}".format(existing))

    audit = json.loads(SEED456_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or not all(audit.get("checks", {}).values()):
        raise RuntimeError("Seed 456 audit is not fully passed")

    existing_metrics = pd.read_csv(
        EXISTING_METRICS, dtype={"condition": str}, low_memory=False
    )
    source456 = pd.read_csv(
        SEED456_METRICS, dtype={"condition": str}, low_memory=False
    )
    normalized456 = normalize_seed456(source456)
    if list(normalized456.columns) != list(existing_metrics.columns):
        raise RuntimeError("Normalized seed 456 schema does not match report metrics schema")
    combined = pd.concat([existing_metrics, normalized456], ignore_index=True)
    if combined.duplicated(["seed", "condition"]).any():
        raise RuntimeError("Duplicate seed-condition rows in combined report metrics")

    common = combined[
        combined["condition"].astype(str).isin(COMMON)
        & combined["seed"].astype(int).isin([42, 123, 456])
    ].copy()
    if len(common) != 27:
        raise RuntimeError("Expected 27 common-condition rows, found {}".format(len(common)))
    common_index = common.set_index(["seed", "condition"])
    comparison_rows = []
    for condition in COMMON:
        row: Dict[str, Any] = {"condition": condition}
        for seed in [42, 123, 456]:
            source = common_index.loc[(seed, condition)]
            for metric in ["FMR", "FNMR", "TAR", "EER"]:
                row["seed{}_{}".format(seed, metric)] = float(source[metric])
        for metric in ["FMR", "FNMR", "TAR", "EER"]:
            row["seed456_minus_seed42_{}_pp".format(metric)] = (
                row["seed456_{}".format(metric)] - row["seed42_{}".format(metric)]
            ) * 100.0
            row["seed456_minus_seed123_{}_pp".format(metric)] = (
                row["seed456_{}".format(metric)] - row["seed123_{}".format(metric)]
            ) * 100.0
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    existing_evidence = pd.read_csv(EXISTING_EVIDENCE, low_memory=False)
    paired456 = pd.read_csv(SEED456_PAIRED, low_memory=False)
    evidence456 = paired456[
        paired456["contrast"].isin(CONTRASTS)
        & paired456["metric"].isin(["FNMR", "EER"])
    ][["contrast", "metric", "difference", "ci_low", "ci_high"]].copy()
    evidence456.insert(0, "seed", 456)
    evidence456["ci_excludes_zero"] = (
        (evidence456["ci_low"].astype(float) > 0.0)
        | (evidence456["ci_high"].astype(float) < 0.0)
    )
    evidence = pd.concat([existing_evidence, evidence456], ignore_index=True)
    if (
        len(evidence) != 48
        or evidence.duplicated(["seed", "contrast", "metric"]).any()
    ):
        raise RuntimeError("Three-seed adjacent evidence is incomplete or duplicated")

    exclusive_csv(THREE_SEED_METRICS, combined)
    exclusive_csv(THREE_SEED_COMPARISON, comparison)
    exclusive_csv(THREE_SEED_EVIDENCE, evidence)

    sources = [
        EXISTING_METRICS,
        EXISTING_EVIDENCE,
        SEED456_METRICS,
        SEED456_PAIRED,
        SEED456_AUDIT,
        Path(__file__).resolve(),
    ]
    outputs = [THREE_SEED_METRICS, THREE_SEED_COMPARISON, THREE_SEED_EVIDENCE]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Seed 456 supplement for the existing seed42/seed123 report",
        "seed456_audit_status": audit["status"],
        "seed456_config_fingerprint": audit["config_fingerprint"],
        "seed456_threshold": audit["threshold"]["saved"],
        "source_files": {
            str(path.relative_to(PROJECT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sources
        },
        "generated_files": {
            path.name: {
                "rows": len(pd.read_csv(path)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        },
    }
    exclusive_json(UPDATE_MANIFEST, manifest)
    print(THREE_SEED_METRICS)
    print(THREE_SEED_COMPARISON)
    print(THREE_SEED_EVIDENCE)
    print(UPDATE_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
