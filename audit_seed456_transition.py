#!/usr/bin/env python3
"""Independent audit and paired cluster bootstrap for the seed-456 run.

This program reads aggregate protocol/result artifacts only. It never opens,
extracts, saves, or displays a FERET face image.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent
RUN = PROJECT / "formal_transition_seed456_gpu_v1"
OUTPUTS = RUN / "outputs"
RUNNER = PROJECT / "feret_formal_experiment_10imp_chunk_checkpoint.py"
FIRST_STDOUT = PROJECT / "seed456_chunk.stdout.log"
RESUME_STDOUT = PROJECT / "seed456_chunk_resume2.stdout.log"
RESUME_EXITCODE = PROJECT / "seed456_chunk_resume2.exitcode"

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
BOOTSTRAP_SEED = 456 + 4001
EXPECTED_FINGERPRINT = (
    "7de9248e78ba0781f97b4169f0d1b5ef60b5a9db76594cbe6c87d808f97dd644"
)
EXPECTED_SCRIPT_SHA256 = (
    "ce742b52a22d94495cfd2506f449382dba47ec81bb600cad9c84299728979181"
)

ASSIGNMENT_COLUMNS = [
    "subject_id",
    "reference_path",
    "reference_category",
    "probe_paths",
    "num_probe_images",
    "split",
]
PAIR_COLUMNS = [
    "pair_id",
    "split",
    "label",
    "label_name",
    "probe_subject_id",
    "reference_subject_id",
    "probe_path",
    "reference_path",
    "probe_category",
    "reference_category",
    "pair_type",
]
RESULT_COLUMNS = [
    "config_fingerprint",
    "split",
    "pair_id",
    "quality",
    "label",
    "label_name",
    "reference_subject_id",
    "probe_subject_id",
    "reference_path",
    "probe_path",
    "reference_category",
    "probe_category",
    "cosine_similarity",
    "probe_size_bytes",
    "reference_size_bytes",
    "reference_detected",
    "probe_detected",
    "comparison_success",
    "failure_reason",
]
SKIPPED_COLUMNS = [
    "config_fingerprint",
    "split",
    "pair_id",
    "quality",
    "image_path",
    "image_role",
    "subject_id",
    "category_code",
    "reason",
    "detail",
]
SAVED_METRIC_COLUMNS = [
    "quality",
    "num_eval_genuine",
    "num_eval_impostor",
    "threshold",
    "FMR",
    "FNMR",
    "TAR",
    "TNR",
    "EER",
    "mean_similarity_genuine",
    "mean_similarity_impostor",
    "median_similarity_genuine",
    "median_similarity_impostor",
    "average_probe_size_bytes",
    "storage_saving_vs_original",
    "mtcnn_failure_rate",
    "reference_detection_failure_rate",
    "probe_detection_failure_rate",
]
METRICS = [
    "FMR",
    "FNMR",
    "TAR",
    "TNR",
    "EER",
    "false_match_count",
    "false_non_match_count",
    "mean_genuine_similarity",
    "mean_impostor_similarity",
    "average_probe_size_bytes",
    "storage_saving_vs_original",
    "detection_failure_count",
    "detection_failure_rate",
    "reference_detection_failure_count",
    "probe_detection_failure_count",
    "embedding_failure_count",
    "embedding_failure_rate",
    "comparison_failure_count",
    "comparison_failure_rate",
]

ARTIFACT_NAMES = [
    "seed456_transition_metrics_with_ci.csv",
    "seed456_transition_metrics_bootstrap_long.csv",
    "seed456_transition_paired_bootstrap.csv",
    "seed456_protocol_digests.json",
    "seed456_software_versions.json",
    "seed456_script_sha256.txt",
    "seed456_artifact_sha256.json",
    "seed456_transition_audit.json",
    "seed456_transition_audit.md",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_digest(dataframe: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = dataframe.loc[:, list(columns)].copy()
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def as_bool(series: pd.Series) -> np.ndarray:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
        .to_numpy(dtype=bool)
    )


def exclusive_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def exclusive_csv(path: Path, dataframe: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        dataframe.to_csv(handle, index=False, lineterminator="\n")
        handle.flush()
        os.fsync(handle.fileno())


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    exclusive_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Cannot JSON-encode {}".format(type(value).__name__))


def record(checks: Dict[str, bool], name: str, condition: bool) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise RuntimeError("Audit check failed: {}".format(name))


def roc_curve_exact(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise RuntimeError("ROC requires genuine and impostor scores")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[
        np.where(sorted_scores[:-1] != sorted_scores[1:])[0],
        len(sorted_scores) - 1,
    ]
    cumulative_positive = np.cumsum(sorted_labels == 1)[ends]
    cumulative_negative = np.cumsum(sorted_labels == 0)[ends]
    return (
        np.r_[0.0, cumulative_negative / negatives],
        np.r_[0.0, cumulative_positive / positives],
        np.r_[np.inf, sorted_scores[ends]],
    )


def eer_from_curve(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> Tuple[float, float]:
    fnr = 1.0 - tpr
    difference = fpr - fnr
    crossings = np.where(difference[:-1] * difference[1:] <= 0)[0]
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


def prepare_condition(
    group: pd.DataFrame, subject_index: Mapping[str, int]
) -> Dict[str, np.ndarray]:
    scores_all = pd.to_numeric(
        group["cosine_similarity"], errors="coerce"
    ).to_numpy(dtype=float)
    sizes = pd.to_numeric(group["probe_size_bytes"], errors="coerce").to_numpy(
        dtype=float
    )
    labels_all = group["label"].astype(int).to_numpy(dtype=np.int8)
    subject_all = np.asarray(
        [subject_index[str(value)] for value in group["probe_subject_id"]],
        dtype=np.int32,
    )
    reference_detected = as_bool(group["reference_detected"])
    probe_detected = as_bool(group["probe_detected"])
    comparison_success_field = as_bool(group["comparison_success"])
    comparison_success = np.isfinite(scores_all)
    failure_reason = group["failure_reason"].fillna("").astype(str).str.lower()
    embedding_failure = failure_reason.str.contains(
        "embedding_failed", regex=False
    ).to_numpy(dtype=bool)
    valid = np.isfinite(scores_all)
    scores = scores_all[valid]
    labels = labels_all[valid]
    subjects = subject_all[valid]
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_subjects = subjects[order]
    ends = np.r_[
        np.where(sorted_scores[:-1] != sorted_scores[1:])[0],
        len(sorted_scores) - 1,
    ]
    return {
        "scores_all": scores_all,
        "sizes": sizes,
        "labels_all": labels_all,
        "subject_all": subject_all,
        "reference_detected": reference_detected,
        "probe_detected": probe_detected,
        "comparison_success": comparison_success,
        "comparison_success_field": comparison_success_field,
        "embedding_failure": embedding_failure,
        "scores": scores,
        "labels": labels,
        "subjects": subjects,
        "sorted_scores": sorted_scores,
        "sorted_labels": sorted_labels,
        "sorted_subjects": sorted_subjects,
        "ends": ends,
        "threshold_values": sorted_scores[ends],
    }


def weighted_eer(data: Mapping[str, np.ndarray], counts: np.ndarray) -> float:
    weights = counts[data["sorted_subjects"]].astype(float)
    positive = data["sorted_labels"] == 1
    negative = data["sorted_labels"] == 0
    positive_total = float(weights[positive].sum())
    negative_total = float(weights[negative].sum())
    if positive_total <= 0 or negative_total <= 0:
        return float("nan")
    cumulative_positive = np.cumsum(weights * positive)[data["ends"]]
    cumulative_negative = np.cumsum(weights * negative)[data["ends"]]
    fpr = np.r_[0.0, cumulative_negative / negative_total]
    tpr = np.r_[0.0, cumulative_positive / positive_total]
    thresholds = np.r_[np.inf, data["threshold_values"]]
    return eer_from_curve(fpr, tpr, thresholds)[0]


def original_average_size(
    original_data: Mapping[str, np.ndarray], counts: np.ndarray
) -> float:
    weights = counts[original_data["subject_all"]].astype(float)
    finite = np.isfinite(original_data["sizes"])
    denominator = float(weights[finite].sum())
    if denominator <= 0:
        return float("nan")
    return float(
        np.sum(weights[finite] * original_data["sizes"][finite]) / denominator
    )


def weighted_metrics(
    data: Mapping[str, np.ndarray],
    counts: np.ndarray,
    threshold: float,
    original_size: float,
) -> Dict[str, float]:
    all_weights = counts[data["subject_all"]].astype(float)
    valid_weights = counts[data["subjects"]].astype(float)
    genuine = data["labels"] == 1
    impostor = data["labels"] == 0
    genuine_total = float(valid_weights[genuine].sum())
    impostor_total = float(valid_weights[impostor].sum())
    false_matches = float(
        valid_weights[impostor & (data["scores"] >= threshold)].sum()
    )
    false_non_matches = float(
        valid_weights[genuine & (data["scores"] < threshold)].sum()
    )
    fmr = false_matches / impostor_total if impostor_total else float("nan")
    fnmr = false_non_matches / genuine_total if genuine_total else float("nan")
    genuine_mean = (
        float(
            np.sum(valid_weights[genuine] * data["scores"][genuine])
            / genuine_total
        )
        if genuine_total
        else float("nan")
    )
    impostor_mean = (
        float(
            np.sum(valid_weights[impostor] * data["scores"][impostor])
            / impostor_total
        )
        if impostor_total
        else float("nan")
    )
    finite_size = np.isfinite(data["sizes"])
    size_total = float(all_weights[finite_size].sum())
    average_size = (
        float(
            np.sum(all_weights[finite_size] * data["sizes"][finite_size])
            / size_total
        )
        if size_total
        else float("nan")
    )
    attempted_total = float(all_weights.sum())
    detection_failure = ~(
        data["reference_detected"] & data["probe_detected"]
    )
    reference_failure = ~data["reference_detected"]
    probe_failure = ~data["probe_detected"]
    detection_count = float(all_weights[detection_failure].sum())
    reference_count = float(all_weights[reference_failure].sum())
    probe_count = float(all_weights[probe_failure].sum())
    embedding_count = float(all_weights[data["embedding_failure"]].sum())
    comparison_count = float(all_weights[~data["comparison_success"]].sum())
    return {
        "FMR": fmr,
        "FNMR": fnmr,
        "TAR": 1.0 - fnmr if np.isfinite(fnmr) else float("nan"),
        "TNR": 1.0 - fmr if np.isfinite(fmr) else float("nan"),
        "EER": weighted_eer(data, counts),
        "false_match_count": false_matches,
        "false_non_match_count": false_non_matches,
        "mean_genuine_similarity": genuine_mean,
        "mean_impostor_similarity": impostor_mean,
        "average_probe_size_bytes": average_size,
        "storage_saving_vs_original": (
            1.0 - average_size / original_size
            if np.isfinite(original_size) and original_size > 0
            else float("nan")
        ),
        "detection_failure_count": detection_count,
        "detection_failure_rate": (
            detection_count / attempted_total if attempted_total else float("nan")
        ),
        "reference_detection_failure_count": reference_count,
        "probe_detection_failure_count": probe_count,
        "embedding_failure_count": embedding_count,
        "embedding_failure_rate": (
            embedding_count / attempted_total if attempted_total else float("nan")
        ),
        "comparison_failure_count": comparison_count,
        "comparison_failure_rate": (
            comparison_count / attempted_total if attempted_total else float("nan")
        ),
    }


def load_and_validate_commits(
    fingerprint: str, checks: Dict[str, bool]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    root = OUTPUTS / "checkpoints"
    result_dir = root / "result_chunks"
    skipped_dir = root / "skipped_chunks"
    commit_dir = root / "commits"
    commits = sorted(commit_dir.glob("commit_*.json"))
    result_frames: List[pd.DataFrame] = []
    skipped_frames: List[pd.DataFrame] = []
    referenced_results = set()
    referenced_skipped = set()
    commit_numbers: List[int] = []
    for commit_path in commits:
        payload = json.loads(commit_path.read_text(encoding="utf-8"))
        number = int(payload["chunk_number"])
        token = "{:06d}".format(number)
        if commit_path.name != "commit_{}.json".format(token):
            raise RuntimeError("Commit filename mismatch: {}".format(commit_path))
        if str(payload["config_fingerprint"]) != fingerprint:
            raise RuntimeError("Commit fingerprint mismatch: {}".format(commit_path))
        result_meta = payload["results"]
        skipped_meta = payload["skipped"]
        expected_result_name = "results_chunk_{}.csv".format(token)
        expected_skipped_name = "skipped_chunk_{}.csv".format(token)
        if result_meta["file"] != expected_result_name:
            raise RuntimeError("Result chunk name mismatch")
        if skipped_meta["file"] != expected_skipped_name:
            raise RuntimeError("Skipped chunk name mismatch")
        result_path = result_dir / expected_result_name
        skipped_path = skipped_dir / expected_skipped_name
        if sha256_file(result_path) != str(result_meta["sha256"]):
            raise RuntimeError("Result chunk hash mismatch: {}".format(result_path))
        if sha256_file(skipped_path) != str(skipped_meta["sha256"]):
            raise RuntimeError("Skipped chunk hash mismatch: {}".format(skipped_path))
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
            raise RuntimeError("Result chunk schema mismatch")
        if list(skipped.columns) != SKIPPED_COLUMNS:
            raise RuntimeError("Skipped chunk schema mismatch")
        if len(result) != int(result_meta["row_count"]):
            raise RuntimeError("Result chunk row count mismatch")
        if len(skipped) != int(skipped_meta["row_count"]):
            raise RuntimeError("Skipped chunk row count mismatch")
        result_frames.append(result)
        skipped_frames.append(skipped)
        referenced_results.add(result_path.name)
        referenced_skipped.add(skipped_path.name)
        commit_numbers.append(number)
    committed = pd.concat(result_frames, ignore_index=True)
    skipped_all = pd.concat(skipped_frames, ignore_index=True)
    actual_results = {path.name for path in result_dir.glob("results_chunk_*.csv")}
    actual_skipped = {path.name for path in skipped_dir.glob("skipped_chunk_*.csv")}
    parts = list(root.rglob("*.part"))
    record(checks, "immutable_commits_count_172", len(commits) == 172)
    record(
        checks,
        "immutable_commit_numbers_contiguous",
        commit_numbers == list(range(1, len(commits) + 1)),
    )
    record(checks, "committed_result_rows_171072", len(committed) == 171072)
    record(
        checks,
        "committed_result_keys_unique",
        not committed.duplicated(["split", "pair_id", "quality"]).any(),
    )
    record(
        checks,
        "all_chunks_have_valid_commits",
        actual_results == referenced_results and actual_skipped == referenced_skipped,
    )
    record(checks, "no_uncommitted_part_files", len(parts) == 0)
    record(
        checks,
        "all_commit_fingerprints_match",
        set(committed["config_fingerprint"].astype(str)) == {fingerprint}
        and (
            skipped_all.empty
            or set(skipped_all["config_fingerprint"].astype(str)) == {fingerprint}
        ),
    )
    detail = {
        "valid_commit_count": len(commits),
        "committed_result_rows": len(committed),
        "committed_skipped_rows": len(skipped_all),
        "uncommitted_part_files": len(parts),
        "orphan_result_chunks": sorted(actual_results - referenced_results),
        "orphan_skipped_chunks": sorted(actual_skipped - referenced_skipped),
    }
    return committed, skipped_all, detail


def compare_final_to_committed(
    final: pd.DataFrame, committed: pd.DataFrame, checks: Dict[str, bool]
) -> Dict[str, Any]:
    quality_order = {quality: index for index, quality in enumerate(CONDITIONS)}

    def sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
        copy = frame.copy()
        copy["_quality_order"] = copy["quality"].map(quality_order)
        return (
            copy.sort_values(
                ["split", "pair_id", "_quality_order"], kind="stable"
            )
            .drop(columns="_quality_order")
            .reset_index(drop=True)
        )

    left = sorted_frame(final)
    right = sorted_frame(committed)
    record(checks, "final_and_committed_row_counts_match", len(left) == len(right))
    numeric_columns = [
        "cosine_similarity",
        "probe_size_bytes",
        "reference_size_bytes",
    ]
    max_differences: Dict[str, float] = {}
    numeric_ok = True
    for column in numeric_columns:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        finite = np.isfinite(a) & np.isfinite(b)
        maximum = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        max_differences[column] = maximum
        numeric_ok = numeric_ok and same_nan and maximum <= 1e-15
    string_mismatches: Dict[str, int] = {}
    for column in [name for name in RESULT_COLUMNS if name not in numeric_columns]:
        a = left[column].fillna("").astype(str).to_numpy()
        b = right[column].fillna("").astype(str).to_numpy()
        string_mismatches[column] = int(np.sum(a != b))
    record(checks, "final_semantically_matches_committed_chunks", numeric_ok and not any(string_mismatches.values()))
    return {
        "numeric_max_absolute_differences": max_differences,
        "non_numeric_mismatches": string_mismatches,
        "tolerance_for_csv_float_roundtrip": 1e-15,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def read_text_auto(path: Path) -> str:
    """Read PowerShell-captured logs without assuming their text encoding."""
    payload = path.read_bytes()
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")
    if payload.startswith(b"\xef\xbb\xbf"):
        return payload.decode("utf-8-sig")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("mbcs", errors="replace")


def software_versions() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in [
                "numpy",
                "pandas",
                "Pillow",
                "torch",
                "torchvision",
                "facenet-pytorch",
                "matplotlib",
            ]
        },
    }
    try:
        import torch

        payload["torch_runtime"] = {
            "torch_version": torch.__version__,
            "compiled_cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            ),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_0_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:
        payload["torch_runtime_error"] = "{}: {}".format(type(exc).__name__, exc)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        payload["nvidia_smi"] = output.strip().splitlines()
    except Exception as exc:
        payload["nvidia_smi_error"] = "{}: {}".format(type(exc).__name__, exc)
    return payload


def main() -> int:
    checks: Dict[str, bool] = {}
    for name in ARTIFACT_NAMES:
        record(
            checks,
            "new_artifact_absent_{}".format(name),
            not (OUTPUTS / name).exists(),
        )

    config = json.loads((OUTPUTS / "experiment_config.json").read_text(encoding="utf-8"))
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
    recomputed_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    record(checks, "config_seed_456", int(config["random_seed"]) == 456)
    record(
        checks,
        "config_conditions_exact",
        [str(value) for value in config["selected_quality_conditions"]]
        == CONDITIONS,
    )
    record(checks, "config_full_experiment", config["run_mode"] == "full-experiment")
    record(
        checks,
        "config_impostors_10",
        int(config["requested_impostors_per_probe"]) == 10
        and config["effective_impostors_per_probe"]
        == {"development": 10, "evaluation": 10},
    )
    record(checks, "config_probe_only", config["compression_mode"] == "probe-only")
    record(
        checks,
        "config_compress_before_detect",
        config["preprocessing_order"] == "compress-before-detect",
    )
    record(
        checks,
        "config_model_exact",
        config["model_name"] == "InceptionResnetV1"
        and config["facenet_checkpoint"] == "vggface2",
    )
    record(
        checks,
        "config_bootstrap_1000", int(config["bootstrap_iterations"]) == 1000
    )
    record(
        checks,
        "config_threshold_strategy_exact",
        config["threshold_strategy"]
        == {
            "source_split": "development",
            "source_condition": "Original",
            "method": "empirical EER intersection",
            "applied_to": "all evaluation conditions",
        },
    )
    record(
        checks,
        "config_fingerprint_recomputed",
        fingerprint == recomputed_fingerprint == EXPECTED_FINGERPRINT,
    )
    script_sha256 = sha256_file(RUNNER)
    record(
        checks,
        "fixed_script_sha256",
        script_sha256 == EXPECTED_SCRIPT_SHA256,
    )
    record(
        checks,
        "resume_exit_code_zero",
        RESUME_EXITCODE.read_text(encoding="utf-8").strip() == "0",
    )
    first_log = read_text_auto(FIRST_STDOUT)
    resume_log = read_text_auto(RESUME_STDOUT)
    record(checks, "embedding_device_cuda_0", "Embedding device: cuda:0" in first_log)
    record(
        checks,
        "resume_skipped_all_committed_rows",
        "Resume: skipping 171,072 completed pair-condition rows." in resume_log,
    )

    assignment = pd.read_csv(
        OUTPUTS / "reference_probe_assignment.csv",
        dtype={"subject_id": str},
        low_memory=False,
    )
    dev_subjects = pd.read_csv(
        OUTPUTS / "dev_subjects.csv", dtype={"subject_id": str}, low_memory=False
    )
    eval_subjects = pd.read_csv(
        OUTPUTS / "eval_subjects.csv", dtype={"subject_id": str}, low_memory=False
    )
    record(
        checks,
        "assignment_schema_exact",
        list(assignment.columns) == ASSIGNMENT_COLUMNS,
    )
    dev_ids = set(dev_subjects["subject_id"].astype(str))
    eval_ids = set(eval_subjects["subject_id"].astype(str))
    assignment_ids = set(assignment["subject_id"].astype(str))
    record(checks, "total_identities_993", len(assignment_ids) == len(assignment) == 993)
    record(checks, "development_identities_199", len(dev_ids) == 199)
    record(checks, "evaluation_identities_794", len(eval_ids) == 794)
    record(checks, "split_subject_disjoint", not (dev_ids & eval_ids))
    record(
        checks,
        "split_subject_union_complete",
        dev_ids | eval_ids == assignment_ids,
    )
    assignment_split = dict(
        zip(assignment["subject_id"].astype(str), assignment["split"].astype(str))
    )
    record(
        checks,
        "subject_csv_matches_assignment",
        dev_ids
        == set(
            assignment.loc[
                assignment["split"].astype(str) == "development", "subject_id"
            ].astype(str)
        )
        and eval_ids
        == set(
            assignment.loc[
                assignment["split"].astype(str) == "evaluation", "subject_id"
            ].astype(str)
        ),
    )

    dev_pairs = read_pairs(OUTPUTS / "dev_pairs.csv")
    eval_pairs = read_pairs(OUTPUTS / "eval_pairs.csv")
    record(checks, "dev_pair_schema_exact", list(dev_pairs.columns) == PAIR_COLUMNS)
    record(checks, "eval_pair_schema_exact", list(eval_pairs.columns) == PAIR_COLUMNS)
    pairs = pd.concat([dev_pairs, eval_pairs], ignore_index=True)
    labels = pairs["label"].astype(int)
    record(checks, "pair_rows_19008", len(pairs) == 19008)
    record(checks, "genuine_pairs_1728", int((labels == 1).sum()) == 1728)
    record(checks, "impostor_pairs_17280", int((labels == 0).sum()) == 17280)
    record(
        checks,
        "pair_ids_unique_within_split",
        not pairs.duplicated(["split", "pair_id"]).any(),
    )
    record(
        checks,
        "pair_split_files_correct",
        set(dev_pairs["split"].astype(str)) == {"development"}
        and set(eval_pairs["split"].astype(str)) == {"evaluation"},
    )
    pair_subject_split_ok = all(
        assignment_split.get(str(row.probe_subject_id)) == str(row.split)
        and assignment_split.get(str(row.reference_subject_id)) == str(row.split)
        for row in pairs.itertuples(index=False)
    )
    record(checks, "all_pair_subjects_in_same_declared_split", pair_subject_split_ok)
    probe_groups = pairs.groupby(
        ["split", "probe_subject_id", "probe_path"], sort=False, dropna=False
    )
    probe_audit = probe_groups.agg(
        rows=("pair_id", "size"),
        genuine=("label", lambda values: int((values.astype(int) == 1).sum())),
        impostor=("label", lambda values: int((values.astype(int) == 0).sum())),
    )
    record(
        checks,
        "every_probe_has_one_genuine_and_ten_impostors",
        bool(
            (probe_audit["rows"] == 11).all()
            and (probe_audit["genuine"] == 1).all()
            and (probe_audit["impostor"] == 10).all()
        ),
    )
    impostors = pairs[labels == 0].copy()
    impostor_groups = impostors.groupby(
        ["split", "probe_subject_id", "probe_path"], sort=False, dropna=False
    )
    unique_without_replacement = impostor_groups["reference_subject_id"].nunique()
    record(
        checks,
        "impostor_references_unique_without_replacement",
        bool((unique_without_replacement == 10).all()),
    )
    record(
        checks,
        "impostor_reference_subject_differs_from_probe",
        bool(
            (
                impostors["reference_subject_id"].astype(str)
                != impostors["probe_subject_id"].astype(str)
            ).all()
        ),
    )

    assignment_protocol_digest = dataframe_digest(
        assignment, ["subject_id", "reference_path", "probe_paths", "split"]
    )
    dev_pair_digest = dataframe_digest(dev_pairs, PAIR_COLUMNS)
    eval_pair_digest = dataframe_digest(eval_pairs, PAIR_COLUMNS)
    record(
        checks,
        "assignment_digest_matches_config",
        assignment_protocol_digest == config["assignment_digest_sha256"],
    )
    record(
        checks,
        "dev_pair_digest_matches_config",
        dev_pair_digest == config["dev_pair_digest_sha256"],
    )
    record(
        checks,
        "eval_pair_digest_matches_config",
        eval_pair_digest == config["eval_pair_digest_sha256"],
    )

    results = read_results(OUTPUTS / "results_final.csv")
    record(checks, "results_final_schema_exact", list(results.columns) == RESULT_COLUMNS)
    record(checks, "result_rows_171072", len(results) == 171072)
    record(
        checks,
        "single_config_fingerprint",
        set(results["config_fingerprint"].astype(str)) == {fingerprint},
    )
    record(
        checks,
        "result_conditions_exact",
        set(results["quality"].astype(str)) == set(CONDITIONS),
    )
    condition_counts = (
        results["quality"].astype(str).value_counts().reindex(CONDITIONS).to_dict()
    )
    record(
        checks,
        "each_condition_has_19008_rows",
        all(int(value) == 19008 for value in condition_counts.values()),
    )
    key_columns = ["split", "pair_id", "quality"]
    record(checks, "result_pair_condition_unique", not results.duplicated(key_columns).any())
    per_pair = results.groupby(["split", "pair_id"], sort=False)["quality"].agg(
        ["size", "nunique"]
    )
    record(
        checks,
        "each_pair_has_exactly_nine_conditions",
        bool((per_pair["size"] == 9).all() and (per_pair["nunique"] == 9).all()),
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
    record(
        checks,
        "identity_path_split_label_invariant_across_conditions",
        all(value == 1 for value in invariant_max.values()),
    )
    original_rows = results[results["quality"].astype(str) == "Original"].copy()
    original_protocol = original_rows.merge(
        pairs,
        on=["split", "pair_id"],
        how="outer",
        suffixes=("_result", "_protocol"),
        indicator=True,
    )
    protocol_match_columns = [
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
            original_protocol["{}_result".format(column)].fillna("").astype(str)
            == original_protocol["{}_protocol".format(column)].fillna("").astype(str)
        ).all()
        for column in protocol_match_columns
    )
    record(
        checks,
        "results_match_frozen_pair_protocol",
        len(original_protocol) == 19008
        and (original_protocol["_merge"] == "both").all()
        and protocol_fields_match,
    )
    success_field = as_bool(results["comparison_success"])
    finite_score = np.isfinite(
        pd.to_numeric(results["cosine_similarity"], errors="coerce").to_numpy()
    )
    record(
        checks,
        "comparison_success_matches_finite_similarity",
        np.array_equal(success_field, finite_score),
    )

    committed, skipped, commit_detail = load_and_validate_commits(
        fingerprint, checks
    )
    final_commit_comparison = compare_final_to_committed(results, committed, checks)

    threshold_payload = json.loads(
        (OUTPUTS / "thresholds.json").read_text(encoding="utf-8")
    )
    dev_original = results[
        (results["split"].astype(str) == "development")
        & (results["quality"].astype(str) == "Original")
    ].copy()
    dev_original_scores = pd.to_numeric(
        dev_original["cosine_similarity"], errors="coerce"
    )
    dev_valid = dev_original[dev_original_scores.notna()].copy()
    dev_fpr, dev_tpr, dev_thresholds = roc_curve_exact(
        dev_valid["label"].astype(int).to_numpy(),
        pd.to_numeric(dev_valid["cosine_similarity"]).to_numpy(dtype=float),
    )
    independent_dev_eer, independent_threshold = eer_from_curve(
        dev_fpr, dev_tpr, dev_thresholds
    )
    saved_threshold = float(threshold_payload["threshold"])
    record(
        checks,
        "threshold_source_seed456_development_original_only",
        threshold_payload["threshold_source_condition"] == "Original"
        and threshold_payload["method"]
        == "development_baseline_empirical_EER_intersection"
        and int(threshold_payload["num_development_genuine_pairs"]) == 367
        and int(threshold_payload["num_development_impostor_pairs"]) == 3670
        and threshold_payload["config_fingerprint"] == fingerprint,
    )
    record(
        checks,
        "threshold_independent_recalculation_exact",
        abs(saved_threshold - independent_threshold) <= 1e-15
        and abs(
            float(threshold_payload["development_eer"]) - independent_dev_eer
        )
        <= 1e-15,
    )

    evaluation = results[results["split"].astype(str) == "evaluation"].copy()
    subjects = sorted(evaluation["probe_subject_id"].astype(str).unique())
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    condition_data = {
        condition: prepare_condition(
            evaluation[evaluation["quality"].astype(str) == condition].copy(),
            subject_index,
        )
        for condition in CONDITIONS
    }
    record(
        checks,
        "comparison_success_field_validated_all_conditions",
        all(
            np.array_equal(
                condition_data[condition]["comparison_success"],
                condition_data[condition]["comparison_success_field"],
            )
            for condition in CONDITIONS
        ),
    )
    unit_counts = np.ones(len(subjects), dtype=np.int32)
    original_point_size = original_average_size(
        condition_data["Original"], unit_counts
    )
    point_metrics = {
        condition: weighted_metrics(
            condition_data[condition],
            unit_counts,
            saved_threshold,
            original_point_size,
        )
        for condition in CONDITIONS
    }

    saved_metrics = pd.read_csv(
        OUTPUTS / "final_metrics_by_quality.csv",
        dtype={"quality": str},
        low_memory=False,
    )
    record(
        checks,
        "saved_metric_schema_exact",
        list(saved_metrics.columns) == SAVED_METRIC_COLUMNS,
    )
    record(
        checks,
        "fixed_threshold_applied_to_all_evaluation_conditions",
        len(saved_metrics) == 9
        and set(saved_metrics["quality"].astype(str)) == set(CONDITIONS)
        and np.max(
            np.abs(saved_metrics["threshold"].astype(float).to_numpy() - saved_threshold)
        )
        <= 1e-15,
    )
    metric_name_map = {
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
    saved_lookup = saved_metrics.set_index("quality")
    metric_differences: Dict[str, Dict[str, float]] = {}
    all_saved_metric_matches = True
    for condition in CONDITIONS:
        metric_differences[condition] = {}
        for saved_name, independent_name in metric_name_map.items():
            difference = abs(
                float(saved_lookup.loc[condition, saved_name])
                - float(point_metrics[condition][independent_name])
            )
            tolerance = (
                1e-9 if saved_name == "average_probe_size_bytes" else 1e-12
            )
            metric_differences[condition][saved_name] = difference
            all_saved_metric_matches = (
                all_saved_metric_matches and difference <= tolerance
            )
        reference_failure_rate = (
            point_metrics[condition]["reference_detection_failure_count"]
            / len(condition_data[condition]["scores_all"])
        )
        probe_failure_rate = (
            point_metrics[condition]["probe_detection_failure_count"]
            / len(condition_data[condition]["scores_all"])
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
            float(saved_lookup.loc[condition, "probe_detection_failure_rate"])
            - probe_failure_rate
        )
        metric_differences[condition][
            "reference_detection_failure_rate"
        ] = reference_difference
        metric_differences[condition][
            "probe_detection_failure_rate"
        ] = probe_difference
        all_saved_metric_matches = (
            all_saved_metric_matches
            and reference_difference <= 1e-12
            and probe_difference <= 1e-12
        )
        data = condition_data[condition]
        expected_valid_genuine = int((data["labels"] == 1).sum())
        expected_valid_impostor = int((data["labels"] == 0).sum())
        all_saved_metric_matches = (
            all_saved_metric_matches
            and int(saved_lookup.loc[condition, "num_eval_genuine"])
            == expected_valid_genuine
            and int(saved_lookup.loc[condition, "num_eval_impostor"])
            == expected_valid_impostor
        )
    if not all_saved_metric_matches:
        print(
            "Saved metric comparison diagnostics:\n{}".format(
                json.dumps(metric_differences, indent=2)
            ),
            flush=True,
        )
        print(
            "Saved valid counts:\n{}".format(
                saved_metrics[
                    ["quality", "num_eval_genuine", "num_eval_impostor"]
                ].to_string(index=False)
            ),
            flush=True,
        )
        print(
            "Independent valid counts: {}".format(
                {
                    condition: {
                        "genuine": int(
                            (condition_data[condition]["labels"] == 1).sum()
                        ),
                        "impostor": int(
                            (condition_data[condition]["labels"] == 0).sum()
                        ),
                    }
                    for condition in CONDITIONS
                }
            ),
            flush=True,
        )
    record(
        checks,
        "independent_evaluation_metrics_match_saved",
        all_saved_metric_matches,
    )
    runner_ci = pd.read_csv(
        OUTPUTS / "final_metrics_with_ci.csv",
        dtype={"quality": str},
        low_memory=False,
    )
    record(
        checks,
        "runner_bootstrap_output_complete",
        len(runner_ci) == 36
        and set(runner_ci["quality"].astype(str)) == set(CONDITIONS)
        and set(runner_ci["metric"].astype(str))
        == {"FMR", "FNMR", "TAR", "average_probe_size_bytes"},
    )

    print(
        "Independent fixed-threshold metrics verified; starting paired cluster bootstrap.",
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
            metric: np.full(BOOTSTRAP_ITERATIONS, np.nan, dtype=float)
            for metric in METRICS
        }
        for condition in CONDITIONS
    }
    for iteration, counts in enumerate(bootstrap_counts):
        bootstrap_original_size = original_average_size(
            condition_data["Original"], counts
        )
        for condition in CONDITIONS:
            values = weighted_metrics(
                condition_data[condition],
                counts,
                saved_threshold,
                bootstrap_original_size,
            )
            for metric in METRICS:
                distributions[condition][metric][iteration] = values[metric]
        if (iteration + 1) % 100 == 0:
            print(
                "Paired evaluation probe-subject bootstrap: {}/{}".format(
                    iteration + 1, BOOTSTRAP_ITERATIONS
                ),
                flush=True,
            )

    wide_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        data = condition_data[condition]
        row: Dict[str, Any] = {
            "condition": condition,
            "source_run": RUN.name,
            "config_fingerprint": fingerprint,
            "threshold": saved_threshold,
            "threshold_source": "seed456 development Original",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_cluster": "evaluation probe_subject_id",
            "num_evaluation_probe_subjects": len(subjects),
            "num_eval_genuine_attempted": int((data["labels_all"] == 1).sum()),
            "num_eval_impostor_attempted": int((data["labels_all"] == 0).sum()),
            "num_eval_genuine_valid": int((data["labels"] == 1).sum()),
            "num_eval_impostor_valid": int((data["labels"] == 0).sum()),
        }
        for metric in METRICS:
            distribution = distributions[condition][metric]
            finite = distribution[np.isfinite(distribution)]
            value = float(point_metrics[condition][metric])
            ci_low = (
                float(np.percentile(finite, 2.5)) if len(finite) else float("nan")
            )
            ci_high = (
                float(np.percentile(finite, 97.5)) if len(finite) else float("nan")
            )
            row[metric] = value
            row["{}_ci_low".format(metric)] = ci_low
            row["{}_ci_high".format(metric)] = ci_high
            long_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "source_run": RUN.name,
                    "config_fingerprint": fingerprint,
                    "threshold": saved_threshold,
                    "threshold_source": "seed456 development Original",
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_cluster": "evaluation probe_subject_id",
                }
            )
        wide_rows.append(row)
    metrics_wide = pd.DataFrame(wide_rows)
    metrics_long = pd.DataFrame(long_rows)

    paired_rows: List[Dict[str, Any]] = []
    for higher, lower in ADJACENT:
        contrast = "Q{}-{}".format(
            higher, "Original" if lower == "Original" else "Q{}".format(lower)
        )
        for metric in METRICS:
            difference_distribution = (
                distributions[higher][metric] - distributions[lower][metric]
            )
            finite = difference_distribution[
                np.isfinite(difference_distribution)
            ]
            paired_rows.append(
                {
                    "contrast": contrast,
                    "higher_condition": higher,
                    "lower_condition": lower,
                    "metric": metric,
                    "difference": float(
                        point_metrics[higher][metric]
                        - point_metrics[lower][metric]
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
                    "threshold_source": "seed456 development Original",
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
    record(
        checks,
        "paired_cluster_bootstrap_1000_complete",
        len(metrics_long) == len(CONDITIONS) * len(METRICS)
        and len(paired_table) == len(ADJACENT) * len(METRICS)
        and set(paired_table["contrast"]) == expected_contrasts
        and len(bootstrap_counts) == 1000,
    )

    known_aggregate_png = {
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
        if suffix in {".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff", ".webp"}:
            forbidden_images.append(str(path.relative_to(RUN)))
        if suffix == ".png" and path.name not in known_aggregate_png:
            unexpected_png.append(str(path.relative_to(RUN)))
    record(
        checks,
        "no_original_face_image_outputs",
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
            "reference_probe_assignment_selected_columns_sha256": assignment_protocol_digest,
            "dev_pairs_sha256": dev_pair_digest,
            "eval_pairs_sha256": eval_pair_digest,
        },
        "file_sha256": {
            name: sha256_file(OUTPUTS / name) for name in protocol_files
        },
        "config_recorded_digests": {
            "assignment_digest_sha256": config["assignment_digest_sha256"],
            "dev_pair_digest_sha256": config["dev_pair_digest_sha256"],
            "eval_pair_digest_sha256": config["eval_pair_digest_sha256"],
        },
    }
    versions = software_versions()

    metrics_path = OUTPUTS / ARTIFACT_NAMES[0]
    metrics_long_path = OUTPUTS / ARTIFACT_NAMES[1]
    paired_path = OUTPUTS / ARTIFACT_NAMES[2]
    protocol_path = OUTPUTS / ARTIFACT_NAMES[3]
    versions_path = OUTPUTS / ARTIFACT_NAMES[4]
    script_hash_path = OUTPUTS / ARTIFACT_NAMES[5]
    artifact_hash_path = OUTPUTS / ARTIFACT_NAMES[6]
    audit_json_path = OUTPUTS / ARTIFACT_NAMES[7]
    audit_md_path = OUTPUTS / ARTIFACT_NAMES[8]
    exclusive_csv(metrics_path, metrics_wide)
    exclusive_csv(metrics_long_path, metrics_long)
    exclusive_csv(paired_path, paired_table)
    exclusive_json(protocol_path, protocol_digests)
    exclusive_json(versions_path, versions)
    exclusive_text(
        script_hash_path,
        "{}  {}\n".format(script_sha256, RUNNER.name),
    )

    artifact_targets = [
        "experiment_config.json",
        "thresholds.json",
        "results_final.csv",
        "final_metrics_by_quality.csv",
        "final_metrics_with_ci.csv",
        "formal_experiment_report.md",
        metrics_path.name,
        metrics_long_path.name,
        paired_path.name,
        protocol_path.name,
        versions_path.name,
        script_hash_path.name,
    ]
    artifact_hashes = {
        "created_utc": now_utc(),
        "config_fingerprint": fingerprint,
        "artifacts": {
            name: {
                "sha256": sha256_file(OUTPUTS / name),
                "bytes": (OUTPUTS / name).stat().st_size,
            }
            for name in artifact_targets
        },
        "runner": {
            "path": str(RUNNER),
            "sha256": script_sha256,
        },
    }
    exclusive_json(artifact_hash_path, artifact_hashes)

    record(checks, "all_required_checks_passed", all(checks.values()))
    audit = {
        "status": "passed" if all(checks.values()) else "failed",
        "created_utc": now_utc(),
        "run_directory": str(RUN),
        "output_directory": str(OUTPUTS),
        "seed": 456,
        "conditions": CONDITIONS,
        "config_fingerprint": fingerprint,
        "recomputed_config_fingerprint": recomputed_fingerprint,
        "runner_sha256": script_sha256,
        "embedding_device_evidence": {
            "device": "cuda:0",
            "log": str(FIRST_STDOUT),
            "resume_log": str(RESUME_STDOUT),
        },
        "exit_code": 0,
        "identity_counts": {
            "total": len(assignment_ids),
            "development": len(dev_ids),
            "evaluation": len(eval_ids),
            "overlap": len(dev_ids & eval_ids),
        },
        "pair_counts": {
            "total": len(pairs),
            "genuine": int((labels == 1).sum()),
            "impostor": int((labels == 0).sum()),
            "development": len(dev_pairs),
            "evaluation": len(eval_pairs),
        },
        "result_counts": {
            "total": len(results),
            "by_condition": condition_counts,
            "comparison_failures_total_all_splits": int((~finite_score).sum()),
            "skipped_records_total_all_splits": len(skipped),
        },
        "checkpoint": commit_detail,
        "final_vs_committed": final_commit_comparison,
        "threshold": {
            "saved": saved_threshold,
            "independent": independent_threshold,
            "absolute_difference": abs(saved_threshold - independent_threshold),
            "development_eer_saved": float(
                threshold_payload["development_eer"]
            ),
            "development_eer_independent": independent_dev_eer,
            "source": "seed456 development Original only",
        },
        "independent_saved_metric_absolute_differences": metric_differences,
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
            "allowed_aggregate_plot_names": sorted(known_aggregate_png),
        },
        "checks": checks,
        "artifacts": artifact_targets + [artifact_hash_path.name],
    }
    exclusive_json(audit_json_path, audit)

    markdown_lines = [
        "# Seed 456 transition experiment audit",
        "",
        "**Status: PASSED**",
        "",
        "This audit contains aggregate protocol and verification data only. No FERET face image was extracted, saved, or displayed.",
        "",
        "## Execution and provenance",
        "",
        "- Exit code: `0`",
        "- Embedding device: `cuda:0` (recorded by the original embedding run)",
        "- Seed: `456`",
        "- Config fingerprint: `{}`".format(fingerprint),
        "- Fixed runner SHA-256: `{}`".format(script_sha256),
        "- Immutable commits: `172` (`171,072` result rows)",
        "- Resume verification: all `171,072` committed rows were skipped",
        "",
        "## Frozen protocol",
        "",
        "- Identities: `993` total; `199` development; `794` evaluation; overlap `0`",
        "- Pairs: `19,008` total; `1,728` genuine; `17,280` impostor",
        "- Conditions: `Original, 15, 11, 10, 9, 8, 7, 6, 5`",
        "- Pair-condition rows: `171,072`",
        "- Each probe: one genuine pair and 10 unique, same-split, different-subject impostor references",
        "",
        "## Threshold verification",
        "",
        "- Source: seed 456 development Original only",
        "- Saved threshold: `{:.17g}`".format(saved_threshold),
        "- Independently recomputed threshold: `{:.17g}`".format(
            independent_threshold
        ),
        "- Absolute difference: `{:.3g}`".format(
            abs(saved_threshold - independent_threshold)
        ),
        "- The same fixed threshold was verified in all nine evaluation conditions.",
        "",
        "## Evaluation-only verification metrics",
        "",
        "| Condition | Genuine valid | Impostor valid | FMR (95% CI) | FNMR (95% CI) | TAR (95% CI) | TNR | EER (95% CI) | Mean genuine | Mean impostor | Avg probe bytes | Storage saving | Detection / embedding / comparison failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in wide_rows:
        markdown_lines.append(
            "| {condition} | {genuine:,} | {impostor:,} | {fmr:.6f} [{fmr_lo:.6f}, {fmr_hi:.6f}] | "
            "{fnmr:.6f} [{fnmr_lo:.6f}, {fnmr_hi:.6f}] | "
            "{tar:.6f} [{tar_lo:.6f}, {tar_hi:.6f}] | {tnr:.6f} | "
            "{eer:.6f} [{eer_lo:.6f}, {eer_hi:.6f}] | {mean_g:.6f} | "
            "{mean_i:.6f} | {size:.2f} | {saving:.6f} | {detect:.0f} / {embed:.0f} / {compare:.0f} |".format(
                condition=row["condition"],
                genuine=row["num_eval_genuine_valid"],
                impostor=row["num_eval_impostor_valid"],
                fmr=row["FMR"],
                fmr_lo=row["FMR_ci_low"],
                fmr_hi=row["FMR_ci_high"],
                fnmr=row["FNMR"],
                fnmr_lo=row["FNMR_ci_low"],
                fnmr_hi=row["FNMR_ci_high"],
                tar=row["TAR"],
                tar_lo=row["TAR_ci_low"],
                tar_hi=row["TAR_ci_high"],
                tnr=row["TNR"],
                eer=row["EER"],
                eer_lo=row["EER_ci_low"],
                eer_hi=row["EER_ci_high"],
                mean_g=row["mean_genuine_similarity"],
                mean_i=row["mean_impostor_similarity"],
                size=row["average_probe_size_bytes"],
                saving=row["storage_saving_vs_original"],
                detect=row["detection_failure_count"],
                embed=row["embedding_failure_count"],
                compare=row["comparison_failure_count"],
            )
        )
    markdown_lines.extend(
        [
            "",
            "All confidence intervals above use 1,000 paired cluster bootstrap replicates over evaluation `probe_subject_id`. The paired-difference CSV contains all requested adjacent contrasts and all audited metrics.",
            "",
            "## Audit conclusion",
            "",
            "All protocol, checkpoint, fingerprint, device, threshold, metric-recalculation, bootstrap, and no-image-output checks passed.",
            "",
            "`SEED 456 TRANSITION EXPERIMENT AUDIT PASSED`",
            "",
        ]
    )
    exclusive_text(audit_md_path, "\n".join(markdown_lines))
    print("SEED 456 TRANSITION EXPERIMENT AUDIT PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
