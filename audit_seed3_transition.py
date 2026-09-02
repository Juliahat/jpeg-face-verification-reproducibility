#!/usr/bin/env python3
"""Read-only audit and paired cluster bootstrap for the seed-3 experiment.

The audit reads only seed-3 aggregate protocol/result artifacts. It reuses
pure statistical helpers from the validated seed-456 audit implementation,
but does not read any seed-42, seed-123, or seed-456 result directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import audit_seed456_transition as statistical_core


PROJECT = Path(__file__).resolve().parent
RUN = PROJECT / "formal_transition_seed3_gpu_v1"
OUTPUTS = RUN / "outputs"
RUNNER = PROJECT / "feret_formal_experiment_10imp_chunk_checkpoint.py"
STDOUT_LOG = PROJECT / "seed3_chunk.stdout.log"
EXITCODE_FILE = PROJECT / "seed3_chunk.exitcode"

CONDITIONS = ["Original", "15", "11", "10", "9", "8", "7", "6", "5"]
ADJACENT = [
    ("15", "Original"),
    ("11", "15"),
    ("10", "11"),
    ("9", "10"),
    ("8", "9"),
    ("7", "8"),
    ("6", "7"),
    ("5", "6"),
]
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 3 + 4001
EXPECTED_FINGERPRINT = (
    "764b467260d98e0f5be9b673cc659374ab8a651f32c681417937a5986fa2d30c"
)
EXPECTED_RUNNER_SHA256 = (
    "ce742b52a22d94495cfd2506f449382dba47ec81bb600cad9c84299728979181"
)

RESULT_COLUMNS = statistical_core.RESULT_COLUMNS
SKIPPED_COLUMNS = statistical_core.SKIPPED_COLUMNS
ASSIGNMENT_COLUMNS = statistical_core.ASSIGNMENT_COLUMNS
PAIR_COLUMNS = statistical_core.PAIR_COLUMNS
SAVED_METRIC_COLUMNS = statistical_core.SAVED_METRIC_COLUMNS
METRICS = statistical_core.METRICS

METRICS_PATH = OUTPUTS / "seed3_transition_metrics_with_ci.csv"
PAIRED_PATH = OUTPUTS / "seed3_transition_paired_bootstrap.csv"
PROTOCOL_PATH = OUTPUTS / "seed3_protocol_digests.json"
SOFTWARE_PATH = OUTPUTS / "seed3_software_versions.json"
SCRIPT_HASH_PATH = OUTPUTS / "seed3_script_sha256.txt"
AUDIT_JSON_PATH = OUTPUTS / "seed3_transition_audit.json"
AUDIT_MD_PATH = OUTPUTS / "seed3_transition_audit.md"
NEW_ARTIFACTS = [
    METRICS_PATH,
    PAIRED_PATH,
    PROTOCOL_PATH,
    SOFTWARE_PATH,
    SCRIPT_HASH_PATH,
    AUDIT_JSON_PATH,
    AUDIT_MD_PATH,
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check(checks: Dict[str, bool], name: str, value: bool) -> None:
    checks[name] = bool(value)
    if not value:
        raise RuntimeError("Audit check failed: {}".format(name))


def exclusive_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
        handle.flush()
        os.fsync(handle.fileno())


def exclusive_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Cannot JSON-encode {}".format(type(value).__name__))


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    exclusive_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n",
    )


def read_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "config_fingerprint": str,
            "pair_id": str,
            "quality": str,
            "reference_subject_id": str,
            "probe_subject_id": str,
        },
        low_memory=False,
    )


def read_pairs(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "pair_id": str,
            "reference_subject_id": str,
            "probe_subject_id": str,
        },
        low_memory=False,
    )


def load_valid_commits(
    fingerprint: str, checks: Dict[str, bool]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    checkpoint_root = OUTPUTS / "checkpoints"
    result_dir = checkpoint_root / "result_chunks"
    skipped_dir = checkpoint_root / "skipped_chunks"
    commit_dir = checkpoint_root / "commits"
    commit_paths = sorted(commit_dir.glob("commit_*.json"))
    result_frames: List[pd.DataFrame] = []
    skipped_frames: List[pd.DataFrame] = []
    referenced_results = set()
    referenced_skipped = set()
    numbers: List[int] = []
    for commit_path in commit_paths:
        payload = json.loads(commit_path.read_text(encoding="utf-8"))
        if str(payload.get("config_fingerprint")) != fingerprint:
            raise RuntimeError(
                "Commit fingerprint mismatch: {}".format(commit_path)
            )
        number = int(payload["chunk_number"])
        token = "{:06d}".format(number)
        if commit_path.name != "commit_{}.json".format(token):
            raise RuntimeError(
                "Commit filename/number mismatch: {}".format(commit_path)
            )
        result_meta = payload["results"]
        skipped_meta = payload["skipped"]
        result_name = "results_chunk_{}.csv".format(token)
        skipped_name = "skipped_chunk_{}.csv".format(token)
        if result_meta["file"] != result_name:
            raise RuntimeError("Committed result filename mismatch")
        if skipped_meta["file"] != skipped_name:
            raise RuntimeError("Committed skipped filename mismatch")
        result_path = result_dir / result_name
        skipped_path = skipped_dir / skipped_name
        if statistical_core.sha256_file(result_path) != str(
            result_meta["sha256"]
        ):
            raise RuntimeError(
                "Committed result SHA-256 mismatch: {}".format(result_path)
            )
        if statistical_core.sha256_file(skipped_path) != str(
            skipped_meta["sha256"]
        ):
            raise RuntimeError(
                "Committed skipped SHA-256 mismatch: {}".format(skipped_path)
            )
        result = read_results(result_path)
        skipped = pd.read_csv(
            skipped_path,
            dtype={
                "config_fingerprint": str,
                "pair_id": str,
                "quality": str,
                "subject_id": str,
            },
            low_memory=False,
        )
        if list(result.columns) != RESULT_COLUMNS:
            raise RuntimeError("Committed result schema mismatch")
        if list(skipped.columns) != SKIPPED_COLUMNS:
            raise RuntimeError("Committed skipped schema mismatch")
        if len(result) != int(result_meta["row_count"]):
            raise RuntimeError("Committed result row-count mismatch")
        if len(skipped) != int(skipped_meta["row_count"]):
            raise RuntimeError("Committed skipped row-count mismatch")
        if set(result["config_fingerprint"].astype(str)) != {fingerprint}:
            raise RuntimeError("Result chunk fingerprint mismatch")
        if not skipped.empty and set(
            skipped["config_fingerprint"].astype(str)
        ) != {fingerprint}:
            raise RuntimeError("Skipped chunk fingerprint mismatch")
        result_frames.append(result)
        skipped_frames.append(skipped)
        referenced_results.add(result_name)
        referenced_skipped.add(skipped_name)
        numbers.append(number)

    committed = pd.concat(result_frames, ignore_index=True)
    skipped_all = pd.concat(skipped_frames, ignore_index=True)
    actual_results = {path.name for path in result_dir.glob("results_chunk_*.csv")}
    actual_skipped = {
        path.name for path in skipped_dir.glob("skipped_chunk_*.csv")
    }
    part_files = list(checkpoint_root.rglob("*.part"))
    check(checks, "valid_commit_count_172", len(commit_paths) == 172)
    check(
        checks,
        "commit_numbers_contiguous",
        numbers == list(range(1, 173)),
    )
    check(checks, "committed_result_rows_171072", len(committed) == 171072)
    check(
        checks,
        "committed_result_keys_unique",
        not committed.duplicated(["split", "pair_id", "quality"]).any(),
    )
    check(
        checks,
        "all_chunks_have_valid_commits",
        actual_results == referenced_results
        and actual_skipped == referenced_skipped,
    )
    check(checks, "no_uncommitted_part_files", not part_files)
    return (
        committed,
        skipped_all,
        {
            "valid_commit_count": len(commit_paths),
            "committed_result_rows": len(committed),
            "committed_skipped_rows": len(skipped_all),
            "uncommitted_part_files": len(part_files),
            "orphan_result_chunks": sorted(actual_results - referenced_results),
            "orphan_skipped_chunks": sorted(actual_skipped - referenced_skipped),
        },
    )


def compare_final_to_committed(
    final: pd.DataFrame,
    committed: pd.DataFrame,
    checks: Dict[str, bool],
) -> Dict[str, Any]:
    order = {condition: index for index, condition in enumerate(CONDITIONS)}

    def sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
        copy = frame.copy()
        copy["_order"] = copy["quality"].astype(str).map(order)
        return (
            copy.sort_values(["split", "pair_id", "_order"], kind="stable")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    left = sorted_frame(final)
    right = sorted_frame(committed)
    check(checks, "final_committed_row_count_match", len(left) == len(right))
    numeric_columns = [
        "cosine_similarity",
        "probe_size_bytes",
        "reference_size_bytes",
    ]
    maximum_differences: Dict[str, float] = {}
    numeric_match = True
    for column in numeric_columns:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        nan_masks_match = np.array_equal(np.isnan(a), np.isnan(b))
        finite = np.isfinite(a) & np.isfinite(b)
        maximum = (
            float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        )
        maximum_differences[column] = maximum
        numeric_match = numeric_match and nan_masks_match and maximum <= 1e-15
    non_numeric_mismatches: Dict[str, int] = {}
    for column in [
        value for value in RESULT_COLUMNS if value not in numeric_columns
    ]:
        a = left[column].fillna("").astype(str).to_numpy()
        b = right[column].fillna("").astype(str).to_numpy()
        non_numeric_mismatches[column] = int(np.sum(a != b))
    check(
        checks,
        "results_final_semantically_matches_committed_chunks",
        numeric_match and not any(non_numeric_mismatches.values()),
    )
    return {
        "numeric_max_absolute_difference": maximum_differences,
        "non_numeric_mismatches": non_numeric_mismatches,
        "numeric_tolerance": 1e-15,
    }


def main() -> int:
    checks: Dict[str, bool] = {}
    for path in NEW_ARTIFACTS:
        check(
            checks,
            "new_artifact_absent_{}".format(path.name),
            not path.exists(),
        )

    check(
        checks,
        "runner_exit_code_zero",
        EXITCODE_FILE.read_text(encoding="ascii").strip() == "0",
    )
    check(
        checks,
        "results_final_exists",
        (OUTPUTS / "results_final.csv").is_file(),
    )
    stdout = statistical_core.read_text_auto(STDOUT_LOG)
    check(
        checks,
        "embedding_device_cuda_0",
        "Embedding device: cuda:0" in stdout,
    )
    check(
        checks,
        "single_run_completed_without_resume",
        "Completed rows: 171,072; resumed rows: 0." in stdout,
    )

    config = json.loads(
        (OUTPUTS / "experiment_config.json").read_text(encoding="utf-8")
    )
    fingerprint = str(config["config_fingerprint"])
    stable_config = {
        key: value
        for key, value in config.items()
        if key not in {"config_fingerprint", "run_timestamp_utc"}
    }
    canonical = json.dumps(
        stable_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    recomputed_fingerprint = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    runner_sha256 = statistical_core.sha256_file(RUNNER)
    check(checks, "config_random_seed_3", int(config["random_seed"]) == 3)
    check(
        checks,
        "config_conditions_exact",
        [str(value) for value in config["selected_quality_conditions"]]
        == CONDITIONS,
    )
    check(
        checks,
        "config_full_experiment",
        config["run_mode"] == "full-experiment",
    )
    check(
        checks,
        "config_impostors_per_probe_10",
        int(config["requested_impostors_per_probe"]) == 10
        and config["effective_impostors_per_probe"]
        == {"development": 10, "evaluation": 10},
    )
    check(
        checks,
        "config_probe_only",
        config["compression_mode"] == "probe-only",
    )
    check(
        checks,
        "config_compress_before_detect",
        config["preprocessing_order"] == "compress-before-detect",
    )
    check(
        checks,
        "config_bootstrap_1000",
        int(config["bootstrap_iterations"]) == 1000,
    )
    check(
        checks,
        "config_threshold_strategy_seed_local",
        config["threshold_strategy"]
        == {
            "source_split": "development",
            "source_condition": "Original",
            "method": "empirical EER intersection",
            "applied_to": "all evaluation conditions",
        },
    )
    check(
        checks,
        "config_fingerprint_recomputed",
        fingerprint == recomputed_fingerprint == EXPECTED_FINGERPRINT,
    )
    check(
        checks,
        "fixed_runner_sha256_verified",
        runner_sha256 == EXPECTED_RUNNER_SHA256,
    )

    assignment = pd.read_csv(
        OUTPUTS / "reference_probe_assignment.csv",
        dtype={"subject_id": str},
        low_memory=False,
    )
    dev_subjects = pd.read_csv(
        OUTPUTS / "dev_subjects.csv",
        dtype={"subject_id": str},
        low_memory=False,
    )
    eval_subjects = pd.read_csv(
        OUTPUTS / "eval_subjects.csv",
        dtype={"subject_id": str},
        low_memory=False,
    )
    dev_ids = set(dev_subjects["subject_id"].astype(str))
    eval_ids = set(eval_subjects["subject_id"].astype(str))
    assignment_ids = set(assignment["subject_id"].astype(str))
    check(
        checks,
        "assignment_schema_exact",
        list(assignment.columns) == ASSIGNMENT_COLUMNS,
    )
    check(
        checks,
        "identity_count_993",
        len(assignment) == len(assignment_ids) == 993,
    )
    check(checks, "development_identity_count_199", len(dev_ids) == 199)
    check(checks, "evaluation_identity_count_794", len(eval_ids) == 794)
    check(checks, "subject_splits_disjoint", not (dev_ids & eval_ids))
    check(
        checks,
        "subject_split_union_complete",
        dev_ids | eval_ids == assignment_ids,
    )
    assignment_split = dict(
        zip(
            assignment["subject_id"].astype(str),
            assignment["split"].astype(str),
        )
    )

    dev_pairs = read_pairs(OUTPUTS / "dev_pairs.csv")
    eval_pairs = read_pairs(OUTPUTS / "eval_pairs.csv")
    pairs = pd.concat([dev_pairs, eval_pairs], ignore_index=True)
    pair_labels = pairs["label"].astype(int)
    check(checks, "dev_pair_schema_exact", list(dev_pairs.columns) == PAIR_COLUMNS)
    check(
        checks,
        "eval_pair_schema_exact",
        list(eval_pairs.columns) == PAIR_COLUMNS,
    )
    check(checks, "total_pairs_19008", len(pairs) == 19008)
    check(
        checks,
        "genuine_pairs_1728",
        int((pair_labels == 1).sum()) == 1728,
    )
    check(
        checks,
        "impostor_pairs_17280",
        int((pair_labels == 0).sum()) == 17280,
    )
    check(
        checks,
        "pair_keys_unique",
        not pairs.duplicated(["split", "pair_id"]).any(),
    )
    pair_subject_split_ok = all(
        assignment_split.get(str(row.probe_subject_id)) == str(row.split)
        and assignment_split.get(str(row.reference_subject_id))
        == str(row.split)
        for row in pairs.itertuples(index=False)
    )
    check(
        checks,
        "pair_subjects_in_declared_split",
        pair_subject_split_ok,
    )
    probe_groups = pairs.groupby(
        ["split", "probe_subject_id", "probe_path"],
        sort=False,
        dropna=False,
    )
    probe_counts = probe_groups.agg(
        total=("pair_id", "size"),
        genuine=(
            "label",
            lambda values: int((values.astype(int) == 1).sum()),
        ),
        impostor=(
            "label",
            lambda values: int((values.astype(int) == 0).sum()),
        ),
    )
    check(
        checks,
        "every_probe_one_genuine_ten_impostors",
        bool(
            (probe_counts["total"] == 11).all()
            and (probe_counts["genuine"] == 1).all()
            and (probe_counts["impostor"] == 10).all()
        ),
    )
    impostors = pairs[pair_labels == 0].copy()
    impostor_groups = impostors.groupby(
        ["split", "probe_subject_id", "probe_path"],
        sort=False,
        dropna=False,
    )
    check(
        checks,
        "impostor_sampling_without_replacement",
        bool(
            (
                impostor_groups["reference_subject_id"].nunique() == 10
            ).all()
        ),
    )
    check(
        checks,
        "impostor_subject_differs_from_probe",
        bool(
            (
                impostors["reference_subject_id"].astype(str)
                != impostors["probe_subject_id"].astype(str)
            ).all()
        ),
    )

    assignment_digest = statistical_core.dataframe_digest(
        assignment,
        ["subject_id", "reference_path", "probe_paths", "split"],
    )
    dev_pair_digest = statistical_core.dataframe_digest(
        dev_pairs, PAIR_COLUMNS
    )
    eval_pair_digest = statistical_core.dataframe_digest(
        eval_pairs, PAIR_COLUMNS
    )
    check(
        checks,
        "assignment_digest_matches_config",
        assignment_digest == config["assignment_digest_sha256"],
    )
    check(
        checks,
        "dev_pair_digest_matches_config",
        dev_pair_digest == config["dev_pair_digest_sha256"],
    )
    check(
        checks,
        "eval_pair_digest_matches_config",
        eval_pair_digest == config["eval_pair_digest_sha256"],
    )

    results = read_results(OUTPUTS / "results_final.csv")
    check(
        checks,
        "results_schema_exact",
        list(results.columns) == RESULT_COLUMNS,
    )
    check(checks, "result_rows_171072", len(results) == 171072)
    check(
        checks,
        "single_result_fingerprint",
        set(results["config_fingerprint"].astype(str)) == {fingerprint},
    )
    check(
        checks,
        "result_conditions_exact",
        set(results["quality"].astype(str)) == set(CONDITIONS),
    )
    condition_counts = (
        results["quality"]
        .astype(str)
        .value_counts()
        .reindex(CONDITIONS)
        .to_dict()
    )
    check(
        checks,
        "each_condition_19008_rows",
        all(int(value) == 19008 for value in condition_counts.values()),
    )
    check(
        checks,
        "pair_condition_keys_unique",
        not results.duplicated(["split", "pair_id", "quality"]).any(),
    )
    per_pair = results.groupby(["split", "pair_id"], sort=False)[
        "quality"
    ].agg(["size", "nunique"])
    check(
        checks,
        "each_pair_has_nine_conditions",
        bool(
            (per_pair["size"] == 9).all()
            and (per_pair["nunique"] == 9).all()
        ),
    )
    invariant_columns = [
        "label",
        "label_name",
        "reference_subject_id",
        "probe_subject_id",
        "reference_path",
        "probe_path",
        "reference_category",
        "probe_category",
        "reference_size_bytes",
    ]
    invariant_max = {
        column: int(
            results.groupby(["split", "pair_id"], sort=False)[column]
            .nunique(dropna=False)
            .max()
        )
        for column in invariant_columns
    }
    check(
        checks,
        "identity_path_split_label_invariant",
        all(value == 1 for value in invariant_max.values()),
    )
    original_rows = results[
        results["quality"].astype(str) == "Original"
    ].copy()
    protocol_merge = original_rows.merge(
        pairs,
        on=["split", "pair_id"],
        how="outer",
        suffixes=("_result", "_protocol"),
        indicator=True,
    )
    match_columns = [
        "label",
        "label_name",
        "reference_subject_id",
        "probe_subject_id",
        "reference_path",
        "probe_path",
        "reference_category",
        "probe_category",
    ]
    protocol_fields_match = all(
        (
            protocol_merge["{}_result".format(column)]
            .fillna("")
            .astype(str)
            == protocol_merge["{}_protocol".format(column)]
            .fillna("")
            .astype(str)
        ).all()
        for column in match_columns
    )
    check(
        checks,
        "results_match_frozen_pair_protocol",
        len(protocol_merge) == 19008
        and (protocol_merge["_merge"] == "both").all()
        and protocol_fields_match,
    )
    comparison_success = statistical_core.as_bool(
        results["comparison_success"]
    )
    finite_similarity = np.isfinite(
        pd.to_numeric(
            results["cosine_similarity"], errors="coerce"
        ).to_numpy(dtype=float)
    )
    check(
        checks,
        "comparison_success_matches_finite_similarity",
        np.array_equal(comparison_success, finite_similarity),
    )

    committed, skipped, checkpoint_detail = load_valid_commits(
        fingerprint, checks
    )
    final_chunk_comparison = compare_final_to_committed(
        results, committed, checks
    )

    threshold_payload = json.loads(
        (OUTPUTS / "thresholds.json").read_text(encoding="utf-8")
    )
    dev_original = results[
        (results["split"].astype(str) == "development")
        & (results["quality"].astype(str) == "Original")
    ].copy()
    dev_scores = pd.to_numeric(
        dev_original["cosine_similarity"], errors="coerce"
    )
    dev_valid = dev_original[dev_scores.notna()].copy()
    dev_fpr, dev_tpr, dev_thresholds = statistical_core.roc_curve_exact(
        dev_valid["label"].astype(int).to_numpy(),
        pd.to_numeric(dev_valid["cosine_similarity"]).to_numpy(dtype=float),
    )
    independent_dev_eer, independent_threshold = (
        statistical_core.eer_from_curve(
            dev_fpr, dev_tpr, dev_thresholds
        )
    )
    saved_threshold = float(threshold_payload["threshold"])
    check(
        checks,
        "threshold_source_seed3_development_original_only",
        threshold_payload["threshold_source_condition"] == "Original"
        and threshold_payload["method"]
        == "development_baseline_empirical_EER_intersection"
        and int(threshold_payload["num_development_genuine_pairs"]) == 366
        and int(threshold_payload["num_development_impostor_pairs"]) == 3660
        and threshold_payload["config_fingerprint"] == fingerprint,
    )
    check(
        checks,
        "threshold_independent_recalculation_exact",
        abs(saved_threshold - independent_threshold) <= 1e-15
        and abs(
            float(threshold_payload["development_eer"])
            - independent_dev_eer
        )
        <= 1e-15,
    )

    evaluation = results[
        results["split"].astype(str) == "evaluation"
    ].copy()
    subjects = sorted(evaluation["probe_subject_id"].astype(str).unique())
    subject_index = {
        subject: index for index, subject in enumerate(subjects)
    }
    condition_data = {
        condition: statistical_core.prepare_condition(
            evaluation[
                evaluation["quality"].astype(str) == condition
            ].copy(),
            subject_index,
        )
        for condition in CONDITIONS
    }
    check(
        checks,
        "evaluation_probe_clusters_794",
        len(subjects) == 794,
    )
    check(
        checks,
        "comparison_success_field_validated",
        all(
            np.array_equal(
                condition_data[condition]["comparison_success"],
                condition_data[condition]["comparison_success_field"],
            )
            for condition in CONDITIONS
        ),
    )
    unit_counts = np.ones(len(subjects), dtype=np.int32)
    original_size = statistical_core.original_average_size(
        condition_data["Original"], unit_counts
    )
    point_metrics = {
        condition: statistical_core.weighted_metrics(
            condition_data[condition],
            unit_counts,
            saved_threshold,
            original_size,
        )
        for condition in CONDITIONS
    }

    saved_metrics = pd.read_csv(
        OUTPUTS / "final_metrics_by_quality.csv",
        dtype={"quality": str},
        low_memory=False,
    )
    check(
        checks,
        "saved_metric_schema_exact",
        list(saved_metrics.columns) == SAVED_METRIC_COLUMNS,
    )
    check(
        checks,
        "fixed_threshold_applied_to_all_conditions",
        len(saved_metrics) == 9
        and set(saved_metrics["quality"].astype(str)) == set(CONDITIONS)
        and np.max(
            np.abs(
                saved_metrics["threshold"].astype(float).to_numpy()
                - saved_threshold
            )
        )
        <= 1e-15,
    )
    saved_lookup = saved_metrics.set_index("quality")
    metric_map = {
        "FMR": "FMR",
        "FNMR": "FNMR",
        "TAR": "TAR",
        "TNR": "TNR",
        "EER": "EER",
        "mean_similarity_genuine": "mean_genuine_similarity",
        "mean_similarity_impostor": "mean_impostor_similarity",
        "average_probe_size_bytes": "average_probe_size_bytes",
        "storage_saving_vs_original": "storage_saving_vs_original",
        "mtcnn_failure_rate": "detection_failure_rate",
    }
    metric_differences: Dict[str, Dict[str, float]] = {}
    saved_metrics_match = True
    for condition in CONDITIONS:
        metric_differences[condition] = {}
        for saved_name, independent_name in metric_map.items():
            difference = abs(
                float(saved_lookup.loc[condition, saved_name])
                - float(point_metrics[condition][independent_name])
            )
            tolerance = (
                1e-9
                if saved_name == "average_probe_size_bytes"
                else 1e-12
            )
            metric_differences[condition][saved_name] = difference
            saved_metrics_match = (
                saved_metrics_match and difference <= tolerance
            )
        data = condition_data[condition]
        attempted_count = len(data["scores_all"])
        reference_failure_rate = (
            point_metrics[condition][
                "reference_detection_failure_count"
            ]
            / attempted_count
        )
        probe_failure_rate = (
            point_metrics[condition]["probe_detection_failure_count"]
            / attempted_count
        )
        reference_difference = abs(
            float(
                saved_lookup.loc[
                    condition, "reference_detection_failure_rate"
                ]
            )
            - reference_failure_rate
        )
        probe_difference = abs(
            float(
                saved_lookup.loc[
                    condition, "probe_detection_failure_rate"
                ]
            )
            - probe_failure_rate
        )
        metric_differences[condition][
            "reference_detection_failure_rate"
        ] = reference_difference
        metric_differences[condition][
            "probe_detection_failure_rate"
        ] = probe_difference
        saved_metrics_match = (
            saved_metrics_match
            and reference_difference <= 1e-12
            and probe_difference <= 1e-12
            and int(saved_lookup.loc[condition, "num_eval_genuine"])
            == int((data["labels"] == 1).sum())
            and int(saved_lookup.loc[condition, "num_eval_impostor"])
            == int((data["labels"] == 0).sum())
        )
    check(
        checks,
        "independent_metrics_match_saved",
        saved_metrics_match,
    )
    runner_ci = pd.read_csv(
        OUTPUTS / "final_metrics_with_ci.csv",
        dtype={"quality": str},
        low_memory=False,
    )
    check(
        checks,
        "runner_bootstrap_output_complete",
        len(runner_ci) == 36
        and set(runner_ci["quality"].astype(str)) == set(CONDITIONS)
        and set(runner_ci["metric"].astype(str))
        == {"FMR", "FNMR", "TAR", "average_probe_size_bytes"},
    )

    print(
        "Independent seed-3 threshold and metrics verified; "
        "starting paired cluster bootstrap.",
        flush=True,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_counts = rng.multinomial(
        len(subjects),
        np.full(len(subjects), 1.0 / len(subjects)),
        size=BOOTSTRAP_ITERATIONS,
    ).astype(np.int32)
    distributions = {
        condition: {
            metric: np.full(
                BOOTSTRAP_ITERATIONS, np.nan, dtype=float
            )
            for metric in METRICS
        }
        for condition in CONDITIONS
    }
    for iteration, counts in enumerate(bootstrap_counts):
        bootstrap_original_size = (
            statistical_core.original_average_size(
                condition_data["Original"], counts
            )
        )
        for condition in CONDITIONS:
            values = statistical_core.weighted_metrics(
                condition_data[condition],
                counts,
                saved_threshold,
                bootstrap_original_size,
            )
            for metric in METRICS:
                distributions[condition][metric][iteration] = values[
                    metric
                ]
        if (iteration + 1) % 100 == 0:
            print(
                "Paired evaluation probe-subject bootstrap: {}/{}".format(
                    iteration + 1, BOOTSTRAP_ITERATIONS
                ),
                flush=True,
            )

    metric_rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        data = condition_data[condition]
        row: Dict[str, Any] = {
            "condition": condition,
            "source_run": RUN.name,
            "config_fingerprint": fingerprint,
            "threshold": saved_threshold,
            "threshold_source": "seed3 development Original",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_cluster": "evaluation probe_subject_id",
            "num_evaluation_probe_subjects": len(subjects),
            "num_eval_genuine_attempted": int(
                (data["labels_all"] == 1).sum()
            ),
            "num_eval_impostor_attempted": int(
                (data["labels_all"] == 0).sum()
            ),
            "num_eval_genuine_valid": int(
                (data["labels"] == 1).sum()
            ),
            "num_eval_impostor_valid": int(
                (data["labels"] == 0).sum()
            ),
        }
        for metric in METRICS:
            distribution = distributions[condition][metric]
            finite = distribution[np.isfinite(distribution)]
            row[metric] = float(point_metrics[condition][metric])
            row["{}_ci_low".format(metric)] = (
                float(np.percentile(finite, 2.5))
                if len(finite)
                else float("nan")
            )
            row["{}_ci_high".format(metric)] = (
                float(np.percentile(finite, 97.5))
                if len(finite)
                else float("nan")
            )
        metric_rows.append(row)
    metrics_table = pd.DataFrame(metric_rows)

    paired_rows: List[Dict[str, Any]] = []
    for lower_quality, prior_quality in ADJACENT:
        contrast = "Q{}-{}".format(
            lower_quality,
            (
                "Original"
                if prior_quality == "Original"
                else "Q{}".format(prior_quality)
            ),
        )
        for metric in METRICS:
            distribution = (
                distributions[lower_quality][metric]
                - distributions[prior_quality][metric]
            )
            finite = distribution[np.isfinite(distribution)]
            paired_rows.append(
                {
                    "contrast": contrast,
                    "lower_quality_condition": lower_quality,
                    "prior_condition": prior_quality,
                    "metric": metric,
                    "difference": float(
                        point_metrics[lower_quality][metric]
                        - point_metrics[prior_quality][metric]
                    ),
                    "ci_low": (
                        float(np.percentile(finite, 2.5))
                        if len(finite)
                        else float("nan")
                    ),
                    "ci_high": (
                        float(np.percentile(finite, 97.5))
                        if len(finite)
                        else float("nan")
                    ),
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_cluster": "evaluation probe_subject_id",
                    "threshold": saved_threshold,
                    "threshold_source": "seed3 development Original",
                }
            )
    paired_table = pd.DataFrame(paired_rows)
    expected_contrasts = {
        "Q15-Original",
        "Q11-Q15",
        "Q10-Q11",
        "Q9-Q10",
        "Q8-Q9",
        "Q7-Q8",
        "Q6-Q7",
        "Q5-Q6",
    }
    check(
        checks,
        "paired_cluster_bootstrap_complete",
        len(metrics_table) == 9
        and len(paired_table) == len(ADJACENT) * len(METRICS)
        and set(paired_table["contrast"]) == expected_contrasts
        and len(bootstrap_counts) == 1000,
    )

    allowed_aggregate_png = {
        "det_by_quality.png",
        "genuine_impostor_distribution_q30.png",
        "genuine_impostor_distribution_q95.png",
        "roc_by_quality.png",
        "similarity_boxplot_by_quality.png",
        "similarity_histogram_by_quality.png",
    }
    forbidden_images: List[str] = []
    unexpected_png: List[str] = []
    for path in RUN.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {
            ".jpg",
            ".jpeg",
            ".ppm",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }:
            forbidden_images.append(str(path.relative_to(RUN)))
        if suffix == ".png" and path.name not in allowed_aggregate_png:
            unexpected_png.append(str(path.relative_to(RUN)))
    check(
        checks,
        "no_raw_face_image_outputs",
        not forbidden_images and not unexpected_png,
    )

    protocol_files = [
        "reference_probe_assignment.csv",
        "dev_subjects.csv",
        "eval_subjects.csv",
        "dev_pairs.csv",
        "eval_pairs.csv",
    ]
    protocol_digests = {
        "config_fingerprint": fingerprint,
        "stable_dataframe_digests": {
            "reference_probe_assignment_selected_columns_sha256": assignment_digest,
            "dev_pairs_sha256": dev_pair_digest,
            "eval_pairs_sha256": eval_pair_digest,
        },
        "file_sha256": {
            name: statistical_core.sha256_file(OUTPUTS / name)
            for name in protocol_files
        },
        "config_recorded_digests": {
            "assignment_digest_sha256": config[
                "assignment_digest_sha256"
            ],
            "dev_pair_digest_sha256": config[
                "dev_pair_digest_sha256"
            ],
            "eval_pair_digest_sha256": config[
                "eval_pair_digest_sha256"
            ],
        },
    }
    versions = statistical_core.software_versions()

    check(checks, "all_required_checks_passed", all(checks.values()))
    exclusive_csv(METRICS_PATH, metrics_table)
    exclusive_csv(PAIRED_PATH, paired_table)
    exclusive_json(PROTOCOL_PATH, protocol_digests)
    exclusive_json(SOFTWARE_PATH, versions)
    exclusive_text(
        SCRIPT_HASH_PATH,
        "{}  {}\n".format(runner_sha256, RUNNER.name),
    )

    audit = {
        "status": "passed",
        "created_utc": now_utc(),
        "run_directory": str(RUN),
        "seed": 3,
        "conditions": CONDITIONS,
        "exit_code": 0,
        "embedding_device": "cuda:0",
        "config_fingerprint": fingerprint,
        "recomputed_config_fingerprint": recomputed_fingerprint,
        "runner_sha256": runner_sha256,
        "identity_counts": {
            "total": len(assignment_ids),
            "development": len(dev_ids),
            "evaluation": len(eval_ids),
            "overlap": len(dev_ids & eval_ids),
        },
        "pair_counts": {
            "total": len(pairs),
            "genuine": int((pair_labels == 1).sum()),
            "impostor": int((pair_labels == 0).sum()),
            "development": len(dev_pairs),
            "evaluation": len(eval_pairs),
        },
        "result_counts": {
            "total": len(results),
            "by_condition": condition_counts,
            "comparison_failures_all_splits": int(
                (~finite_similarity).sum()
            ),
            "skipped_records_all_splits": len(skipped),
        },
        "checkpoint": checkpoint_detail,
        "final_vs_committed": final_chunk_comparison,
        "threshold": {
            "saved": saved_threshold,
            "independent": independent_threshold,
            "absolute_difference": abs(
                saved_threshold - independent_threshold
            ),
            "development_eer_saved": float(
                threshold_payload["development_eer"]
            ),
            "development_eer_independent": independent_dev_eer,
            "source": "seed3 development Original only",
        },
        "saved_metric_absolute_differences": metric_differences,
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "cluster": "evaluation probe_subject_id",
            "cluster_count": len(subjects),
            "paired_across_all_conditions": True,
            "contrasts": sorted(expected_contrasts),
            "metric_count": len(METRICS),
        },
        "cross_condition_invariant_max_nunique": invariant_max,
        "image_output_audit": {
            "forbidden_raster_files": forbidden_images,
            "unexpected_png_files": unexpected_png,
            "allowed_aggregate_plots": sorted(allowed_aggregate_png),
        },
        "checks": checks,
        "artifacts": [path.name for path in NEW_ARTIFACTS],
    }
    exclusive_json(AUDIT_JSON_PATH, audit)

    lines = [
        "# Seed 3 transition experiment audit",
        "",
        "**Status: PASSED**",
        "",
        "This audit contains aggregate protocol and verification data only. "
        "No FERET face image was extracted, saved, or displayed.",
        "",
        "## Execution and protocol",
        "",
        "- Exit code: `0`",
        "- Embedding device: `cuda:0`",
        "- Seed: `3`",
        "- Config fingerprint: `{}`".format(fingerprint),
        "- Runner SHA-256: `{}`".format(runner_sha256),
        "- Identities: `993` (`199` development; `794` evaluation; overlap `0`)",
        "- Pairs: `19,008` (`1,728` genuine; `17,280` impostor)",
        "- Pair-condition rows: `171,072`",
        "- Valid immutable commits: `172`",
        "",
        "## Threshold",
        "",
        "- Source: seed 3 development Original only",
        "- Saved threshold: `{:.17g}`".format(saved_threshold),
        "- Independently recomputed threshold: `{:.17g}`".format(
            independent_threshold
        ),
        "- Absolute difference: `{:.3g}`".format(
            abs(saved_threshold - independent_threshold)
        ),
        "",
        "## Evaluation-only verification metrics",
        "",
        "| Condition | Genuine valid | Impostor valid | FMR (95% CI) | "
        "FNMR (95% CI) | TAR (95% CI) | TNR | EER (95% CI) | "
        "Mean genuine | Mean impostor | Avg probe bytes | Storage saving | "
        "Detection / embedding / comparison failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {condition} | {genuine:,} | {impostor:,} | "
            "{fmr:.6f} [{fmr_low:.6f}, {fmr_high:.6f}] | "
            "{fnmr:.6f} [{fnmr_low:.6f}, {fnmr_high:.6f}] | "
            "{tar:.6f} [{tar_low:.6f}, {tar_high:.6f}] | "
            "{tnr:.6f} | "
            "{eer:.6f} [{eer_low:.6f}, {eer_high:.6f}] | "
            "{mean_genuine:.6f} | {mean_impostor:.6f} | "
            "{average_size:.2f} | {saving:.6f} | "
            "{detection:.0f} / {embedding:.0f} / {comparison:.0f} |".format(
                condition=row["condition"],
                genuine=row["num_eval_genuine_valid"],
                impostor=row["num_eval_impostor_valid"],
                fmr=row["FMR"],
                fmr_low=row["FMR_ci_low"],
                fmr_high=row["FMR_ci_high"],
                fnmr=row["FNMR"],
                fnmr_low=row["FNMR_ci_low"],
                fnmr_high=row["FNMR_ci_high"],
                tar=row["TAR"],
                tar_low=row["TAR_ci_low"],
                tar_high=row["TAR_ci_high"],
                tnr=row["TNR"],
                eer=row["EER"],
                eer_low=row["EER_ci_low"],
                eer_high=row["EER_ci_high"],
                mean_genuine=row["mean_genuine_similarity"],
                mean_impostor=row["mean_impostor_similarity"],
                average_size=row["average_probe_size_bytes"],
                saving=row["storage_saving_vs_original"],
                detection=row["detection_failure_count"],
                embedding=row["embedding_failure_count"],
                comparison=row["comparison_failure_count"],
            )
        )
    lines.extend(
        [
            "",
            "All 95% confidence intervals use 1,000 paired cluster bootstrap "
            "replicates over evaluation `probe_subject_id`. The paired CSV "
            "contains all eight requested adjacent contrasts.",
            "",
            "## Conclusion",
            "",
            "All execution, configuration, protocol, immutable checkpoint, "
            "result-completeness, threshold, independent metric, paired "
            "bootstrap, and no-face-image-output checks passed.",
            "",
            "`SEED 3 TRANSITION EXPERIMENT AUDIT PASSED`",
            "",
        ]
    )
    exclusive_text(AUDIT_MD_PATH, "\n".join(lines))
    print("SEED 3 TRANSITION EXPERIMENT AUDIT PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
