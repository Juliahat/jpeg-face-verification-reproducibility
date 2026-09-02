#!/usr/bin/env python3
"""Merge and audit the seed-123 Part A/Part B chunk-checkpoint experiments."""

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent
PART_A_NAME = "formal_transition_seed123_partA_gpu_v1"
PART_B_NAME = "formal_transition_seed123_partB_gpu_v1"
PART_A = PROJECT / PART_A_NAME
PART_B = PROJECT / PART_B_NAME
OUTPUT_ROOT = PROJECT / "formal_transition_seed123_chunk_combined_v1"
OUTPUTS = OUTPUT_ROOT / "outputs"
CONDITIONS = ["Original", "15", "11", "10", "9", "8", "7", "6", "5"]
PART_A_CONDITIONS = {"Original", "15", "11", "10", "9"}
PART_B_CONDITIONS = {"8", "7", "6", "5"}
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
PROTOCOL_FILES = [
    "reference_probe_assignment.csv",
    "dev_subjects.csv",
    "eval_subjects.csv",
    "dev_pairs.csv",
    "eval_pairs.csv",
]
RESULT_KEYS = ["split", "pair_id", "quality"]
PAIR_KEYS = ["split", "pair_id"]
STABLE_FIELDS = [
    "label",
    "label_name",
    "reference_subject_id",
    "probe_subject_id",
    "reference_path",
    "probe_path",
    "reference_category",
    "probe_category",
]
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 4124
THRESHOLD_SOURCE = (
    "formal_transition_seed123_partA_gpu_v1:development:Original"
)
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
    "mtcnn_failure_count",
    "mtcnn_failure_rate",
    "comparison_failure_count",
    "comparison_failure_rate",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(values: pd.Series) -> np.ndarray:
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y"])
        .to_numpy(dtype=bool)
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


def eer_from_curve(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> Tuple[float, float]:
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


def prepare_condition(
    group: pd.DataFrame,
    subject_index: Mapping[str, int],
) -> Dict[str, np.ndarray]:
    scores_all = pd.to_numeric(group["cosine_similarity"], errors="coerce").to_numpy()
    sizes = pd.to_numeric(group["probe_size_bytes"], errors="coerce").to_numpy()
    labels_all = group["label"].astype(int).to_numpy()
    subject_all = np.asarray(
        [subject_index[str(value)] for value in group["probe_subject_id"]],
        dtype=np.int32,
    )
    reference_detected = as_bool(group["reference_detected"])
    probe_detected = as_bool(group["probe_detected"])
    comparison_success = np.isfinite(scores_all)
    valid = np.isfinite(scores_all)
    scores = scores_all[valid]
    labels = labels_all[valid]
    subjects = subject_all[valid]
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_subjects = subjects[order]
    if len(sorted_scores):
        ends = np.r_[np.where(sorted_scores[:-1] != sorted_scores[1:])[0], len(sorted_scores) - 1]
        threshold_values = sorted_scores[ends]
    else:
        ends = np.asarray([], dtype=int)
        threshold_values = np.asarray([], dtype=float)
    return {
        "scores_all": scores_all,
        "sizes": sizes,
        "labels_all": labels_all,
        "subject_all": subject_all,
        "reference_detected": reference_detected,
        "probe_detected": probe_detected,
        "comparison_success": comparison_success,
        "scores": scores,
        "labels": labels,
        "subjects": subjects,
        "sorted_scores": sorted_scores,
        "sorted_labels": sorted_labels,
        "sorted_subjects": sorted_subjects,
        "ends": ends,
        "threshold_values": threshold_values,
    }


def weighted_eer(data: Mapping[str, np.ndarray], cluster_counts: np.ndarray) -> float:
    if not len(data["sorted_scores"]):
        return float("nan")
    weights = cluster_counts[data["sorted_subjects"]]
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
    eer, _ = eer_from_curve(fpr, tpr, thresholds)
    return eer


def weighted_metrics(
    data: Mapping[str, np.ndarray],
    cluster_counts: np.ndarray,
    threshold: float,
    original_average_size: float,
) -> Dict[str, float]:
    all_weights = cluster_counts[data["subject_all"]].astype(float)
    valid_weights = cluster_counts[data["subjects"]].astype(float)
    genuine = data["labels"] == 1
    impostor = data["labels"] == 0
    genuine_weight = float(valid_weights[genuine].sum())
    impostor_weight = float(valid_weights[impostor].sum())
    false_match_count = float(
        valid_weights[impostor & (data["scores"] >= threshold)].sum()
    )
    false_non_match_count = float(
        valid_weights[genuine & (data["scores"] < threshold)].sum()
    )
    fmr = (
        false_match_count / impostor_weight if impostor_weight > 0 else float("nan")
    )
    fnmr = (
        false_non_match_count / genuine_weight if genuine_weight > 0 else float("nan")
    )
    genuine_mean = (
        float(
            np.sum(valid_weights[genuine] * data["scores"][genuine])
            / genuine_weight
        )
        if genuine_weight > 0
        else float("nan")
    )
    impostor_mean = (
        float(
            np.sum(valid_weights[impostor] * data["scores"][impostor])
            / impostor_weight
        )
        if impostor_weight > 0
        else float("nan")
    )
    finite_size = np.isfinite(data["sizes"])
    size_weight = float(all_weights[finite_size].sum())
    average_size = (
        float(
            np.sum(all_weights[finite_size] * data["sizes"][finite_size])
            / size_weight
        )
        if size_weight > 0
        else float("nan")
    )
    both_detected = data["reference_detected"] & data["probe_detected"]
    attempted_weight = float(all_weights.sum())
    mtcnn_failure_count = float(all_weights[~both_detected].sum())
    comparison_failure_count = float(
        all_weights[~data["comparison_success"]].sum()
    )
    metrics = {
        "FMR": fmr,
        "FNMR": fnmr,
        "TAR": 1.0 - fnmr if np.isfinite(fnmr) else float("nan"),
        "TNR": 1.0 - fmr if np.isfinite(fmr) else float("nan"),
        "EER": weighted_eer(data, cluster_counts),
        "false_match_count": false_match_count,
        "false_non_match_count": false_non_match_count,
        "mean_genuine_similarity": genuine_mean,
        "mean_impostor_similarity": impostor_mean,
        "average_probe_size_bytes": average_size,
        "storage_saving_vs_original": (
            1.0 - average_size / original_average_size
            if np.isfinite(average_size)
            and np.isfinite(original_average_size)
            and original_average_size > 0
            else float("nan")
        ),
        "mtcnn_failure_count": mtcnn_failure_count,
        "mtcnn_failure_rate": (
            mtcnn_failure_count / attempted_weight
            if attempted_weight > 0
            else float("nan")
        ),
        "comparison_failure_count": comparison_failure_count,
        "comparison_failure_rate": (
            comparison_failure_count / attempted_weight
            if attempted_weight > 0
            else float("nan")
        ),
    }
    return metrics


def original_average_size(
    original_data: Mapping[str, np.ndarray], cluster_counts: np.ndarray
) -> float:
    weights = cluster_counts[original_data["subject_all"]].astype(float)
    finite = np.isfinite(original_data["sizes"])
    denominator = float(weights[finite].sum())
    return (
        float(np.sum(weights[finite] * original_data["sizes"][finite]) / denominator)
        if denominator > 0
        else float("nan")
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def main() -> int:
    if OUTPUT_ROOT.exists():
        final_audit = OUTPUTS / "combined_transition_audit_seed123_q15_q5.json"
        if not OUTPUTS.is_dir() or final_audit.exists():
            raise RuntimeError(
                "Combined output directory is not an incomplete resumable output: {}".format(
                    OUTPUT_ROOT
                )
            )
        print(
            "Resuming deterministic finalization of incomplete combined outputs: {}".format(
                OUTPUT_ROOT
            ),
            flush=True,
        )
    part_a_outputs = PART_A / "outputs"
    part_b_outputs = PART_B / "outputs"
    part_a_audit = json.loads(
        (PROJECT / "seed123_partA_chunk_audit.json").read_text(encoding="utf-8")
    )
    part_b_audit = json.loads(
        (PROJECT / "seed123_partB_chunk_audit.json").read_text(encoding="utf-8")
    )
    if part_a_audit.get("status") != "passed" or part_b_audit.get("status") != "passed":
        raise RuntimeError("Both per-run audits must pass before merge")

    config_a = json.loads(
        (part_a_outputs / "experiment_config.json").read_text(encoding="utf-8")
    )
    config_b = json.loads(
        (part_b_outputs / "experiment_config.json").read_text(encoding="utf-8")
    )
    fingerprint_a = str(config_a["config_fingerprint"])
    fingerprint_b = str(config_b["config_fingerprint"])
    threshold_payload_a = json.loads(
        (part_a_outputs / "thresholds.json").read_text(encoding="utf-8")
    )
    threshold_a = float(threshold_payload_a["threshold"])

    protocol_digests: Dict[str, Dict[str, Any]] = {}
    for name in PROTOCOL_FILES:
        digest_a = sha256(part_a_outputs / name)
        digest_b = sha256(part_b_outputs / name)
        protocol_digests[name] = {
            "part_a_sha256": digest_a,
            "part_b_sha256": digest_b,
            "match": digest_a == digest_b,
        }
    if not all(item["match"] for item in protocol_digests.values()):
        raise RuntimeError("Part A and Part B protocol digests differ")

    results_a = read_results(part_a_outputs / "results_final.csv")
    results_b = read_results(part_b_outputs / "results_final.csv")
    original_a = results_a[results_a["quality"].astype(str) == "Original"].copy()
    original_b = results_b[results_b["quality"].astype(str) == "Original"].copy()
    original_merge = original_a.merge(
        original_b,
        on=["split", "pair_id", "quality"],
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    original_keys_exact = bool((original_merge["_merge"] == "both").all())
    original_metadata_changes = 0
    for column in STABLE_FIELDS + [
        "probe_size_bytes",
        "reference_size_bytes",
        "reference_detected",
        "probe_detected",
        "comparison_success",
        "failure_reason",
    ]:
        left = original_merge["{}_a".format(column)].fillna("<NA>").astype(str)
        right = original_merge["{}_b".format(column)].fillna("<NA>").astype(str)
        original_metadata_changes += int((left != right).sum())
    similarity_a = pd.to_numeric(
        original_merge["cosine_similarity_a"], errors="coerce"
    )
    similarity_b = pd.to_numeric(
        original_merge["cosine_similarity_b"], errors="coerce"
    )
    original_difference = (similarity_b - similarity_a).abs()
    valid_difference = original_difference.dropna()
    decision_a = similarity_a >= threshold_a
    decision_b = similarity_b >= threshold_a
    decision_changes = int((decision_a != decision_b).sum())
    original_consistency = {
        "row_count_part_a": int(len(original_a)),
        "row_count_part_b": int(len(original_b)),
        "keys_one_to_one": original_keys_exact,
        "metadata_field_changes": original_metadata_changes,
        "cosine_absolute_difference_mean": float(valid_difference.mean()),
        "cosine_absolute_difference_median": float(valid_difference.median()),
        "cosine_absolute_difference_p99": float(valid_difference.quantile(0.99)),
        "cosine_absolute_difference_maximum": float(valid_difference.max()),
        "cosine_nonzero_differences": int((valid_difference != 0.0).sum()),
        "part_a_threshold_decision_changes": decision_changes,
        "part_a_fixed_threshold": threshold_a,
    }
    original_check = (
        len(original_a) == 19008
        and len(original_b) == 19008
        and original_keys_exact
        and original_metadata_changes == 0
        and len(valid_difference) == 19008
        and bool((valid_difference == 0.0).all())
        and decision_changes == 0
    )
    if not original_check:
        raise RuntimeError("Part A and Part B Original consistency check failed")

    selected_a = results_a[
        results_a["quality"].astype(str).isin(PART_A_CONDITIONS)
    ].copy()
    selected_b = results_b[
        results_b["quality"].astype(str).isin(PART_B_CONDITIONS)
    ].copy()
    selected_a["source_run"] = PART_A_NAME
    selected_b["source_run"] = PART_B_NAME
    selected_a["threshold_source"] = THRESHOLD_SOURCE
    selected_b["threshold_source"] = THRESHOLD_SOURCE
    combined = pd.concat([selected_a, selected_b], ignore_index=True)
    order = {quality: index for index, quality in enumerate(CONDITIONS)}
    combined["_quality_order"] = (
        combined["quality"].astype(str).map(order).astype(int)
    )
    combined = combined.sort_values(
        ["split", "pair_id", "_quality_order"], kind="stable"
    ).drop(columns="_quality_order")

    expected_rows = 19008 * len(CONDITIONS)
    combined_group_conditions = combined.groupby(PAIR_KEYS)["quality"].agg(
        lambda values: set(values.astype(str))
    )
    cross_condition_changes = int(
        sum(
            (
                combined.groupby(PAIR_KEYS)[column].nunique(dropna=False) != 1
            ).sum()
            for column in STABLE_FIELDS
        )
    )
    combined_integrity = {
        "rows_171072": len(combined) == expected_rows == 171072,
        "no_duplicate_keys": not combined.duplicated(RESULT_KEYS).any(),
        "pair_count_19008": len(combined_group_conditions) == 19008,
        "every_pair_has_nine_conditions": bool(
            combined_group_conditions.map(
                lambda values: values == set(CONDITIONS)
            ).all()
        ),
        "no_cross_condition_metadata_changes": cross_condition_changes == 0,
        "original_only_from_part_a": set(
            combined.loc[
                combined["quality"].astype(str) == "Original", "source_run"
            ]
        )
        == {PART_A_NAME},
        "threshold_source_uniform_part_a": set(combined["threshold_source"])
        == {THRESHOLD_SOURCE},
    }
    if not all(combined_integrity.values()):
        raise RuntimeError("Combined result integrity check failed")

    evaluation = combined[combined["split"] == "evaluation"].copy()
    eval_keys = evaluation.groupby("quality")["pair_id"].agg(lambda values: set(values))
    paired_eval_keys = all(
        eval_keys[str(condition)] == eval_keys["Original"]
        for condition in CONDITIONS
    )
    subjects = sorted(
        evaluation["probe_subject_id"].astype(str).unique().tolist()
    )
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    condition_data = {
        condition: prepare_condition(
            evaluation[evaluation["quality"].astype(str) == condition].copy(),
            subject_index,
        )
        for condition in CONDITIONS
    }
    unit_counts = np.ones(len(subjects), dtype=np.int32)
    original_point_average = original_average_size(
        condition_data["Original"], unit_counts
    )
    point_metrics = {
        condition: weighted_metrics(
            condition_data[condition],
            unit_counts,
            threshold_a,
            original_point_average,
        )
        for condition in CONDITIONS
    }

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
        original_average = original_average_size(
            condition_data["Original"], counts
        )
        for condition in CONDITIONS:
            values = weighted_metrics(
                condition_data[condition],
                counts,
                threshold_a,
                original_average,
            )
            for metric in METRICS:
                distributions[condition][metric][iteration] = values[metric]
        if (iteration + 1) % 100 == 0:
            print(
                "Paired cluster bootstrap: {}/{}".format(
                    iteration + 1, BOOTSTRAP_ITERATIONS
                ),
                flush=True,
            )

    metric_rows: List[Dict[str, Any]] = []
    metric_long_rows: List[Dict[str, Any]] = []
    source_by_condition = {
        condition: (PART_A_NAME if condition in PART_A_CONDITIONS else PART_B_NAME)
        for condition in CONDITIONS
    }
    fingerprint_by_condition = {
        condition: (fingerprint_a if condition in PART_A_CONDITIONS else fingerprint_b)
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        data = condition_data[condition]
        row: Dict[str, Any] = {
            "condition": condition,
            "source_run": source_by_condition[condition],
            "config_fingerprint": fingerprint_by_condition[condition],
            "threshold": threshold_a,
            "threshold_source": THRESHOLD_SOURCE,
            "threshold_source_config_fingerprint": fingerprint_a,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_cluster": "evaluation probe_subject_id",
            "num_evaluation_probe_subjects": len(subjects),
            "num_eval_genuine_valid": int((data["labels"] == 1).sum()),
            "num_eval_impostor_valid": int((data["labels"] == 0).sum()),
            "num_eval_attempted": int(len(data["scores_all"])),
        }
        for metric in METRICS:
            distribution = distributions[condition][metric]
            finite = distribution[np.isfinite(distribution)]
            value = point_metrics[condition][metric]
            ci_low = (
                float(np.percentile(finite, 2.5)) if len(finite) else float("nan")
            )
            ci_high = (
                float(np.percentile(finite, 97.5)) if len(finite) else float("nan")
            )
            row[metric] = value
            row["{}_ci_low".format(metric)] = ci_low
            row["{}_ci_high".format(metric)] = ci_high
            metric_long_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "source_run": source_by_condition[condition],
                    "config_fingerprint": fingerprint_by_condition[condition],
                    "threshold": threshold_a,
                    "threshold_source": THRESHOLD_SOURCE,
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_cluster": "evaluation probe_subject_id",
                }
            )
        metric_rows.append(row)
    metrics_table = pd.DataFrame(metric_rows)
    metrics_long = pd.DataFrame(metric_long_rows)

    paired_rows: List[Dict[str, Any]] = []
    for higher, lower in ADJACENT:
        contrast = "Q{}-{}".format(
            higher, "Original" if lower == "Original" else "Q{}".format(lower)
        )
        for metric in METRICS:
            point_difference = (
                point_metrics[higher][metric] - point_metrics[lower][metric]
            )
            distribution = (
                distributions[higher][metric] - distributions[lower][metric]
            )
            finite = distribution[np.isfinite(distribution)]
            paired_rows.append(
                {
                    "contrast": contrast,
                    "higher_condition": higher,
                    "lower_condition": lower,
                    "metric": metric,
                    "difference": point_difference,
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
                    "threshold": threshold_a,
                    "threshold_source": THRESHOLD_SOURCE,
                }
            )
    paired_table = pd.DataFrame(paired_rows)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        OUTPUTS / "combined_transition_results_seed123_q15_q5.csv",
        index=False,
        lineterminator="\n",
    )
    metrics_table.to_csv(
        OUTPUTS / "combined_transition_metrics_seed123_q15_q5.csv",
        index=False,
        lineterminator="\n",
    )
    metrics_long.to_csv(
        OUTPUTS / "combined_transition_metrics_bootstrap_long_seed123_q15_q5.csv",
        index=False,
        lineterminator="\n",
    )
    paired_table.to_csv(
        OUTPUTS / "combined_transition_paired_bootstrap_seed123_q15_q5.csv",
        index=False,
        lineterminator="\n",
    )
    (OUTPUTS / "protocol_digests.json").write_text(
        json.dumps(protocol_digests, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUTS / "original_consistency.json").write_text(
        json.dumps(original_consistency, indent=2) + "\n", encoding="utf-8"
    )
    software_versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "Pillow": package_version("Pillow"),
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "facenet-pytorch": package_version("facenet-pytorch"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "chunk_runner_sha256": sha256(
            PROJECT / "feret_formal_experiment_10imp_chunk_checkpoint.py"
        ),
    }
    (OUTPUTS / "software_versions.json").write_text(
        json.dumps(software_versions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    checks = {
        "part_a_audit_passed": part_a_audit["status"] == "passed",
        "part_b_audit_passed": part_b_audit["status"] == "passed",
        "protocol_digests_match": all(
            item["match"] for item in protocol_digests.values()
        ),
        "original_consistency_exact": original_check,
        "combined_integrity": all(combined_integrity.values()),
        "evaluation_pair_keys_paired": paired_eval_keys,
        "evaluation_probe_subjects_794": len(subjects) == 794,
        "part_a_threshold_used": bool(
            (metrics_table["threshold"] == threshold_a).all()
            and set(metrics_table["threshold_source"]) == {THRESHOLD_SOURCE}
        ),
        "bootstrap_1000": BOOTSTRAP_ITERATIONS == 1000,
        "all_conditions_have_all_metric_intervals": (
            len(metrics_long) == len(CONDITIONS) * len(METRICS)
            and metrics_long[["value", "ci_low", "ci_high"]].notna().all(axis=None)
        ),
        "all_adjacent_paired_differences_present": (
            len(paired_table) == len(ADJACENT) * len(METRICS)
            and set(paired_table["contrast"])
            == {
                "Q15-Original",
                "Q11-Q15",
                "Q10-Q11",
                "Q9-Q10",
                "Q8-Q9",
                "Q7-Q8",
                "Q6-Q7",
                "Q5-Q6",
            }
        ),
        "source_fields_retained": all(
            column in combined.columns
            for column in ["source_run", "config_fingerprint", "threshold_source"]
        ),
    }
    audit = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "conditions": CONDITIONS,
        "part_a_config_fingerprint": fingerprint_a,
        "part_b_config_fingerprint": fingerprint_b,
        "fixed_threshold": threshold_a,
        "threshold_source": THRESHOLD_SOURCE,
        "combined_rows": len(combined),
        "combined_pair_count": len(combined_group_conditions),
        "evaluation_probe_subjects": len(subjects),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "protocol_digests": protocol_digests,
        "original_consistency": original_consistency,
        "combined_integrity": combined_integrity,
        "output_files": {},
    }
    for path in sorted(OUTPUTS.iterdir()):
        if path.is_file():
            audit["output_files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    (OUTPUTS / "combined_transition_audit_seed123_q15_q5.json").write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Seed 123 chunk-checkpoint split transition audit",
        "",
        "Status: **{}**".format(audit["status"].upper()),
        "",
        "- Combined conditions: {}".format(", ".join(CONDITIONS)),
        "- Combined pair-condition rows: {:,}".format(len(combined)),
        "- Unique pairs: {:,}".format(len(combined_group_conditions)),
        "- Fixed threshold: {:.16f}".format(threshold_a),
        "- Threshold source: `{}`".format(THRESHOLD_SOURCE),
        "- Evaluation bootstrap clusters: {:,} probe subjects".format(
            len(subjects)
        ),
        "- Paired cluster bootstrap iterations: {:,}".format(
            BOOTSTRAP_ITERATIONS
        ),
        "- Original cosine absolute difference mean/median/p99/max: "
        "{:.16g} / {:.16g} / {:.16g} / {:.16g}".format(
            original_consistency["cosine_absolute_difference_mean"],
            original_consistency["cosine_absolute_difference_median"],
            original_consistency["cosine_absolute_difference_p99"],
            original_consistency["cosine_absolute_difference_maximum"],
        ),
        "- Original fixed-threshold decision changes: {}".format(
            original_consistency["part_a_threshold_decision_changes"]
        ),
        "- Protocol digest files matching: {}/{}".format(
            sum(item["match"] for item in protocol_digests.values()),
            len(protocol_digests),
        ),
        "",
        "No original FERET face image was extracted, saved, or displayed.",
        "",
    ]
    (OUTPUTS / "combined_transition_audit_seed123_q15_q5.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    if audit["status"] != "passed":
        print(
            json.dumps(
                audit,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            )
        )
        return 2
    print("SEED 123 CHUNK-CHECKPOINT SPLIT TRANSITION AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
