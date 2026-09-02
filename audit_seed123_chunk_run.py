#!/usr/bin/env python3
"""Strict audit for one seed-123 immutable chunk-checkpoint run."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent
RUNNER_PATH = PROJECT / "feret_formal_experiment_10imp_chunk_checkpoint.py"
RESULT_KEYS = ["split", "pair_id", "quality"]
PAIR_STABLE_FIELDS = [
    "label",
    "label_name",
    "reference_subject_id",
    "probe_subject_id",
    "reference_path",
    "probe_path",
    "reference_category",
    "probe_category",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("chunk_runner_run_audit", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import immutable checkpoint runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def roc_points(labels: np.ndarray, scores: np.ndarray):
    thresholds = np.concatenate(([np.inf], np.unique(scores)[::-1]))
    positives = labels == 1
    negatives = labels == 0
    fpr = []
    tpr = []
    for threshold in thresholds:
        accepted = scores >= threshold
        fpr.append(float(accepted[negatives].mean()))
        tpr.append(float(accepted[positives].mean()))
    return np.asarray(fpr), np.asarray(tpr), thresholds


def eer_intersection(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    fnr = 1.0 - tpr
    difference = fpr - fnr
    exact = np.where(np.isclose(difference, 0.0, atol=1e-12))[0]
    if len(exact):
        index = int(exact[0])
        return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])
    crossings = np.where(np.sign(difference[:-1]) != np.sign(difference[1:]))[0]
    if len(crossings):
        left = int(crossings[0])
        right = left + 1
        denominator = abs(difference[left]) + abs(difference[right])
        weight = abs(difference[left]) / denominator if denominator else 0.0
        eer = float(fpr[left] + weight * (fpr[right] - fpr[left]))
        if np.isfinite(thresholds[left]) and np.isfinite(thresholds[right]):
            threshold = float(
                thresholds[left]
                + weight * (thresholds[right] - thresholds[left])
            )
        else:
            threshold = float(thresholds[right])
        return eer, threshold
    index = int(np.argmin(np.abs(difference)))
    return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])


def audit(
    run_dir: Path,
    conditions: Sequence[str],
    stdout_log: Path,
    exit_code_path: Path,
    legacy_hash: str,
) -> Dict[str, Any]:
    runner = load_runner()
    outputs = run_dir / "outputs"
    final_path = outputs / "results_final.csv"
    config_path = outputs / "experiment_config.json"
    threshold_path = outputs / "thresholds.json"
    metrics_path = outputs / "final_metrics_by_quality.csv"
    required_files = [
        final_path,
        config_path,
        threshold_path,
        metrics_path,
        outputs / "reference_probe_assignment.csv",
        outputs / "dev_subjects.csv",
        outputs / "eval_subjects.csv",
        outputs / "dev_pairs.csv",
        outputs / "eval_pairs.csv",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise RuntimeError("Missing required output files: {}".format(missing_files))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprint = str(config["config_fingerprint"])
    dtype = {
        "config_fingerprint": str,
        "pair_id": str,
        "quality": str,
        "reference_subject_id": str,
        "probe_subject_id": str,
    }
    final = pd.read_csv(final_path, dtype=dtype)
    assignment = pd.read_csv(
        outputs / "reference_probe_assignment.csv", dtype={"subject_id": str}
    )
    dev_subjects = pd.read_csv(
        outputs / "dev_subjects.csv", dtype={"subject_id": str}
    )
    eval_subjects = pd.read_csv(
        outputs / "eval_subjects.csv", dtype={"subject_id": str}
    )
    dev_pairs = pd.read_csv(
        outputs / "dev_pairs.csv",
        dtype={
            "pair_id": str,
            "probe_subject_id": str,
            "reference_subject_id": str,
        },
    )
    eval_pairs = pd.read_csv(
        outputs / "eval_pairs.csv",
        dtype={
            "pair_id": str,
            "probe_subject_id": str,
            "reference_subject_id": str,
        },
    )
    pairs = pd.concat([dev_pairs, eval_pairs], ignore_index=True)
    expected_pair_count = 19008
    expected_rows = expected_pair_count * len(conditions)

    checkpoint_results, checkpoint_skips, valid_commits = (
        runner._load_checkpoint_state(outputs, fingerprint, conditions)
    )
    checkpoint_sorted = checkpoint_results.sort_values(
        RESULT_KEYS, kind="stable"
    ).reset_index(drop=True)
    final_sorted = final.sort_values(RESULT_KEYS, kind="stable").reset_index(drop=True)
    checkpoint_matches_final = (
        len(checkpoint_sorted) == len(final_sorted)
        and runner._dataframe_digest(
            checkpoint_sorted, runner.FORMAL_RESULT_COLUMNS
        )
        == runner._dataframe_digest(final_sorted, runner.FORMAL_RESULT_COLUMNS)
    )

    group_conditions = final.groupby(["split", "pair_id"])["quality"].agg(
        lambda values: set(values.astype(str))
    )
    expected_condition_set = set(conditions)
    cross_condition_changes = int(
        sum(
            (
                final.groupby(["split", "pair_id"])[column].nunique(
                    dropna=False
                )
                != 1
            ).sum()
            for column in PAIR_STABLE_FIELDS
        )
    )
    successful = final["comparison_success"].map(runner._as_bool)
    similarities = pd.to_numeric(final["cosine_similarity"], errors="coerce")

    dev_original = final[
        (final["split"] == "development")
        & (final["quality"].astype(str) == "Original")
        & similarities.notna()
    ].copy()
    labels = dev_original["label"].astype(int).to_numpy()
    scores = pd.to_numeric(
        dev_original["cosine_similarity"], errors="raise"
    ).to_numpy()
    fpr, tpr, thresholds = roc_points(labels, scores)
    development_eer, independent_threshold = eer_intersection(
        fpr, tpr, thresholds
    )
    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    saved_metrics = pd.read_csv(metrics_path, dtype={"quality": str}).set_index(
        "quality"
    )
    metric_errors = {"FMR": 0.0, "FNMR": 0.0, "TAR": 0.0}
    decision_counts: Dict[str, Dict[str, int]] = {}
    for condition in conditions:
        group = final[
            (final["split"] == "evaluation")
            & (final["quality"].astype(str) == condition)
        ].copy()
        group_scores = pd.to_numeric(group["cosine_similarity"], errors="coerce")
        valid = group[group_scores.notna()].copy()
        valid_scores = pd.to_numeric(valid["cosine_similarity"], errors="raise")
        genuine = valid_scores[valid["label"].astype(int) == 1]
        impostor = valid_scores[valid["label"].astype(int) == 0]
        false_matches = int((impostor >= independent_threshold).sum())
        false_non_matches = int((genuine < independent_threshold).sum())
        values = {
            "FMR": float(false_matches / len(impostor)),
            "FNMR": float(false_non_matches / len(genuine)),
        }
        values["TAR"] = 1.0 - values["FNMR"]
        for metric in metric_errors:
            metric_errors[metric] = max(
                metric_errors[metric],
                abs(float(saved_metrics.loc[condition, metric]) - values[metric]),
            )
        decision_counts[condition] = {
            "false_matches": false_matches,
            "false_non_matches": false_non_matches,
        }

    impostor_pairs = pairs[pairs["label"].astype(int) == 0]
    impostor_group = impostor_pairs.groupby(
        ["split", "probe_subject_id", "probe_path"]
    )
    impostor_protocol_ok = True
    for (split_name, probe_subject, _), group in impostor_group:
        if len(group) != 10:
            impostor_protocol_ok = False
            break
        if group["reference_subject_id"].nunique() != 10:
            impostor_protocol_ok = False
            break
        if (group["reference_subject_id"].astype(str) == str(probe_subject)).any():
            impostor_protocol_ok = False
            break
        if (group["split"].astype(str) != str(split_name)).any():
            impostor_protocol_ok = False
            break

    stdout_text = (
        stdout_log.read_text(encoding="utf-16", errors="ignore")
        if stdout_log.exists()
        else ""
    )
    if "Embedding device: cuda:0" not in stdout_text and stdout_log.exists():
        stdout_text = stdout_log.read_text(encoding="utf-8", errors="ignore")
    exit_code_text = (
        exit_code_path.read_text(encoding="ascii").strip()
        if exit_code_path.exists()
        else ""
    )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".ppm",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
    raw_image_outputs = [
        str(path.relative_to(outputs))
        for path in outputs.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    ]
    legacy_path = outputs / "results_partial.csv"
    legacy_actual_hash = sha256(legacy_path) if legacy_path.exists() else ""
    legacy_row_count = (
        len(pd.read_csv(legacy_path, dtype=dtype)) if legacy_path.exists() else 0
    )

    config_conditions = [str(value) for value in config["selected_quality_conditions"]]
    dev_set = set(dev_subjects["subject_id"].astype(str))
    eval_set = set(eval_subjects["subject_id"].astype(str))
    checks = {
        "exit_code_zero": exit_code_text == "0",
        "results_final_exists": final_path.is_file(),
        "pair_count_19008": len(pairs) == expected_pair_count,
        "genuine_pair_count_1728": int((pairs["label"].astype(int) == 1).sum())
        == 1728,
        "impostor_pair_count_17280": int((pairs["label"].astype(int) == 0).sum())
        == 17280,
        "result_rows_95040": len(final) == expected_rows == 95040,
        "no_duplicate_result_keys": not final.duplicated(RESULT_KEYS).any(),
        "every_pair_has_exact_conditions": bool(
            len(group_conditions) == expected_pair_count
            and group_conditions.map(lambda value: value == expected_condition_set).all()
        ),
        "no_cross_condition_metadata_changes": cross_condition_changes == 0,
        "all_successful_cosines_complete": not similarities[successful].isna().any(),
        "single_matching_fingerprint": set(
            final["config_fingerprint"].astype(str)
        )
        == {fingerprint},
        "config_seed_123": int(config["random_seed"]) == 123,
        "config_conditions_exact": config_conditions == list(conditions),
        "config_full_experiment": config["run_mode"] == "full-experiment",
        "config_protocol_exact": (
            config["compression_mode"] == "probe-only"
            and config["preprocessing_order"] == "compress-before-detect"
            and int(config["requested_impostors_per_probe"]) == 10
            and int(config["bootstrap_iterations"]) == 1000
        ),
        "development_identities_199": len(dev_set) == 199,
        "evaluation_identities_794": len(eval_set) == 794,
        "subject_splits_disjoint": not (dev_set & eval_set),
        "assignment_subject_count_993": len(assignment) == 993,
        "impostor_sampling_valid": impostor_protocol_ok,
        "device_cuda0": "Embedding device: cuda:0" in stdout_text,
        "threshold_exact": float(threshold_payload["threshold"])
        == independent_threshold,
        "threshold_source_dev_original": (
            threshold_payload["threshold_source_condition"] == "Original"
            and int(threshold_payload["num_development_genuine_pairs"])
            == int((dev_original["label"].astype(int) == 1).sum())
            and int(threshold_payload["num_development_impostor_pairs"])
            == int((dev_original["label"].astype(int) == 0).sum())
        ),
        "independent_metrics_match": max(metric_errors.values()) <= 1e-15,
        "checkpoint_state_matches_final": checkpoint_matches_final,
        "no_checkpoint_part_files": not any(
            (outputs / "checkpoints").rglob("*.part")
        ),
        "no_raw_face_images": not raw_image_outputs,
        "legacy_checkpoint_unchanged": (
            (not legacy_hash and not legacy_path.exists())
            or (
                bool(legacy_hash)
                and legacy_actual_hash.upper() == legacy_hash.upper()
                and legacy_row_count == 9000
            )
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "run_dir": str(run_dir),
        "conditions": list(conditions),
        "checks": checks,
        "fingerprint": fingerprint,
        "pair_count": len(pairs),
        "result_rows": len(final),
        "valid_commits": valid_commits,
        "recoverable_checkpoint_rows": len(checkpoint_results),
        "skipped_rows": len(checkpoint_skips),
        "legacy_row_count": legacy_row_count,
        "legacy_sha256": legacy_actual_hash,
        "independent_threshold": independent_threshold,
        "saved_threshold": float(threshold_payload["threshold"]),
        "development_eer": development_eer,
        "metric_max_absolute_error": metric_errors,
        "decision_counts": decision_counts,
        "raw_image_outputs": raw_image_outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--exit-code-path", type=Path, required=True)
    parser.add_argument("--legacy-sha256", default="")
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.run_dir.resolve(),
        [str(value) for value in args.conditions],
        args.stdout_log.resolve(),
        args.exit_code_path.resolve(),
        str(args.legacy_sha256),
    )
    args.report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
