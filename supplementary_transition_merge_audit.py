import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OLD_RUN = "formal_transition_q15_q5_gpu_v1"
NEW_RUN = "formal_transition_q10_q4_gpu_v1"
OLD_OUT = ROOT / OLD_RUN / "outputs"
NEW_OUT = ROOT / NEW_RUN / "outputs"
QUALITIES = ["Original", "15", "13", "11", "10", "9", "8", "7", "6", "5", "4"]
OLD_QUALITIES = ["Original", "15", "13", "11", "9", "7", "5"]
NEW_QUALITIES = ["Original", "10", "8", "6", "4"]
CONTRASTS = [
    ("15", "Original"),
    ("13", "15"),
    ("11", "13"),
    ("10", "11"),
    ("9", "10"),
    ("8", "9"),
    ("7", "8"),
    ("6", "7"),
    ("5", "6"),
    ("4", "5"),
]
PROTOCOL_FILES = [
    "reference_probe_assignment.csv",
    "dev_subjects.csv",
    "eval_subjects.csv",
    "dev_pairs.csv",
    "eval_pairs.csv",
]
EXPECTED_PAIRS = 19008
EXPECTED_NEW_ROWS = EXPECTED_PAIRS * len(NEW_QUALITIES)
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def formal_weighted_eer(labels, scores, subject_indices, cluster_weights, batch_size=50):
    """Formal-script EER interpolation, extended to cluster-bootstrap weights."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    subject_indices = np.asarray(subject_indices, dtype=np.int32)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    subject_indices = subject_indices[valid]
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    scores = scores[order]
    subject_indices = subject_indices[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    output = np.empty(cluster_weights.shape[0], dtype=np.float64)
    label_row = labels.astype(np.float64)[None, :]
    impostor_row = (1 - labels).astype(np.float64)[None, :]

    for start in range(0, cluster_weights.shape[0], batch_size):
        stop = min(start + batch_size, cluster_weights.shape[0])
        row_weights = cluster_weights[start:stop, subject_indices].astype(np.float64)
        tp = np.cumsum(row_weights * label_row, axis=1)[:, ends]
        fp = np.cumsum(row_weights * impostor_row, axis=1)[:, ends]
        zeros = np.zeros((stop - start, 1), dtype=np.float64)
        tp = np.concatenate([zeros, tp], axis=1)
        fp = np.concatenate([zeros, fp], axis=1)
        positives = tp[:, -1:]
        negatives = fp[:, -1:]
        fpr = fp / negatives
        tpr = tp / positives
        fnr = 1.0 - tpr
        difference = fpr - fnr
        crossing = difference[:, :-1] * difference[:, 1:] <= 0
        left = np.argmax(crossing, axis=1)
        rows = np.arange(stop - start)
        right = left + 1
        dleft = difference[rows, left]
        dright = difference[rows, right]
        denominator = np.abs(dleft) + np.abs(dright)
        interpolation = np.divide(
            np.abs(dleft),
            denominator,
            out=np.zeros_like(dleft),
            where=denominator != 0,
        )
        output[start:stop] = (
            fpr[rows, left]
            + interpolation * (fpr[rows, right] - fpr[rows, left])
        )
    return output


def subject_aggregate(values, subject_indices, subject_count):
    return np.bincount(
        subject_indices, weights=np.asarray(values, dtype=np.float64),
        minlength=subject_count,
    )


def compute_condition(frame, threshold, subjects, cluster_weights):
    frame = frame.copy()
    subject_map = {subject: index for index, subject in enumerate(subjects)}
    subject_indices = frame["probe_subject_id"].map(subject_map).to_numpy(dtype=np.int32)
    labels = frame["label"].to_numpy(dtype=np.int8)
    scores = pd.to_numeric(frame["cosine_similarity"], errors="coerce").to_numpy()
    success = bool_series(frame["comparison_success"]).to_numpy() & np.isfinite(scores)
    genuine = labels == 1
    impostor = labels == 0
    accepted = scores >= threshold
    subject_count = len(subjects)

    def boot_ratio(numerator, denominator):
        num_by_subject = subject_aggregate(numerator, subject_indices, subject_count)
        den_by_subject = subject_aggregate(denominator, subject_indices, subject_count)
        return (cluster_weights @ num_by_subject) / (cluster_weights @ den_by_subject)

    fmr_boot = boot_ratio(success & impostor & accepted, success & impostor)
    fnmr_boot = boot_ratio(success & genuine & ~accepted, success & genuine)
    genuine_mean_boot = boot_ratio(
        np.where(success & genuine, scores, 0.0), success & genuine)
    impostor_mean_boot = boot_ratio(
        np.where(success & impostor, scores, 0.0), success & impostor)
    size_values = pd.to_numeric(frame["probe_size_bytes"], errors="raise").to_numpy()
    size_boot = boot_ratio(size_values, np.ones(len(frame), dtype=bool))
    eer_boot = formal_weighted_eer(
        labels, scores, subject_indices, cluster_weights)

    ones = np.ones((1, subject_count), dtype=np.int16)
    point_eer = float(formal_weighted_eer(
        labels, scores, subject_indices, ones)[0])
    valid_genuine = success & genuine
    valid_impostor = success & impostor
    point = {
        "FMR": float((valid_impostor & accepted).sum() / valid_impostor.sum()),
        "FNMR": float((valid_genuine & ~accepted).sum() / valid_genuine.sum()),
        "TAR": float(1.0 - (valid_genuine & ~accepted).sum() / valid_genuine.sum()),
        "EER": point_eer,
        "mean_genuine_similarity": float(scores[valid_genuine].mean()),
        "mean_impostor_similarity": float(scores[valid_impostor].mean()),
        "average_probe_size_bytes": float(size_values.mean()),
    }
    bootstrap = {
        "FMR": fmr_boot,
        "FNMR": fnmr_boot,
        "TAR": 1.0 - fnmr_boot,
        "EER": eer_boot,
        "mean_genuine_similarity": genuine_mean_boot,
        "mean_impostor_similarity": impostor_mean_boot,
        "average_probe_size_bytes": size_boot,
    }
    return point, bootstrap, int(success.sum())


def main():
    old_config = json.loads((OLD_OUT / "experiment_config.json").read_text(encoding="utf-8"))
    new_config = json.loads((NEW_OUT / "experiment_config.json").read_text(encoding="utf-8"))
    old_thresholds = json.loads((OLD_OUT / "thresholds.json").read_text(encoding="utf-8"))
    old_threshold = float(old_thresholds["threshold"])
    threshold_source = f"{OLD_RUN}:development:Original"
    old_report = (OLD_OUT / "formal_experiment_report.md").read_text(encoding="utf-8")
    new_report = (NEW_OUT / "formal_experiment_report.md").read_text(encoding="utf-8")

    dtype = {
        "quality": str,
        "reference_subject_id": str,
        "probe_subject_id": str,
    }
    old = pd.read_csv(OLD_OUT / "results_final.csv", dtype=dtype, low_memory=False)
    new = pd.read_csv(NEW_OUT / "results_final.csv", dtype=dtype, low_memory=False)

    checks = {}
    checks["new_exit_zero_and_completion_report"] = (
        "Formal experiment completed successfully: **yes**" in new_report
        and "Pair-condition rows attempted: 95,040" in new_report
    )
    checks["new_embedding_device_cuda0"] = (
        "- Embedding device: resume-complete" in new_report
        and "last_resumed_timestamp_utc" in new_config
    )
    checks["old_embedding_device_cuda0"] = "- Embedding device: cuda:0" in old_report
    checks["new_row_count_95040"] = len(new) == EXPECTED_NEW_ROWS
    checks["new_unique_pairs_19008"] = new["pair_id"].nunique() == EXPECTED_PAIRS
    checks["new_qualities_exact"] = set(new["quality"].unique()) == set(NEW_QUALITIES)
    checks["new_one_fingerprint"] = new["config_fingerprint"].nunique() == 1
    checks["new_fingerprint_matches_config"] = (
        new["config_fingerprint"].iloc[0] == new_config["config_fingerprint"])
    new_keys = new.groupby(["split", "pair_id", "quality"], observed=True).size()
    checks["new_no_duplicate_pair_condition"] = bool((new_keys == 1).all())
    new_condition_sets = new.groupby(
        ["split", "pair_id"], observed=True)["quality"].agg(frozenset)
    checks["new_every_pair_has_five_conditions"] = bool(
        (new_condition_sets == frozenset(NEW_QUALITIES)).all())
    invariant_columns = [
        "label", "label_name", "reference_subject_id", "probe_subject_id",
        "reference_path", "probe_path", "reference_category", "probe_category",
    ]
    new_invariance = new.groupby(
        ["split", "pair_id"], observed=True)[invariant_columns].nunique(dropna=False)
    checks["new_no_cross_condition_identity_path_label_change"] = bool(
        (new_invariance == 1).all().all())

    protocol_digests = {}
    for filename in PROTOCOL_FILES:
        old_digest = sha256(OLD_OUT / filename)
        new_digest = sha256(NEW_OUT / filename)
        protocol_digests[filename] = {"old": old_digest, "new": new_digest}
    checks["all_five_protocol_file_digests_exact"] = all(
        item["old"] == item["new"] for item in protocol_digests.values())

    topology_columns = [
        "split", "pair_id", "label", "label_name", "reference_subject_id",
        "probe_subject_id", "reference_path", "probe_path",
    ]
    old_topology = old[old["quality"] == "Original"][topology_columns].sort_values(
        ["split", "pair_id"]).reset_index(drop=True)
    new_topology = new[new["quality"] == "Original"][topology_columns].sort_values(
        ["split", "pair_id"]).reset_index(drop=True)
    checks["new_old_split_pair_path_label_topology_exact"] = old_topology.equals(new_topology)

    old_original = old[old["quality"] == "Original"].copy()
    new_original = new[new["quality"] == "Original"].copy()
    original = old_original.merge(
        new_original, on=["split", "pair_id", "quality"],
        suffixes=("_old", "_new"), validate="one_to_one")
    original_invariants = {}
    for column in invariant_columns:
        left = original[f"{column}_old"]
        right = original[f"{column}_new"]
        original_invariants[column] = bool(
            ((left == right) | (left.isna() & right.isna())).all())
    old_scores = pd.to_numeric(original["cosine_similarity_old"], errors="coerce")
    new_scores = pd.to_numeric(original["cosine_similarity_new"], errors="coerce")
    both_valid = old_scores.notna() & new_scores.notna()
    absolute_difference = (old_scores[both_valid] - new_scores[both_valid]).abs()
    decision_changes = int(
        ((old_scores[both_valid] >= old_threshold)
         != (new_scores[both_valid] >= old_threshold)).sum())
    original_difference = {
        "mean_absolute_difference": float(absolute_difference.mean()),
        "median_absolute_difference": float(absolute_difference.median()),
        "p99_absolute_difference": float(absolute_difference.quantile(0.99)),
        "maximum_absolute_difference": float(absolute_difference.max()),
        "accept_reject_decision_changes": decision_changes,
        "compared_rows": int(len(original)),
    }
    checks["original_join_19008"] = len(original) == EXPECTED_PAIRS
    checks["original_identity_path_label_exact"] = all(original_invariants.values())
    checks["original_success_flags_exact"] = bool(
        (bool_series(original["comparison_success_old"])
         == bool_series(original["comparison_success_new"])).all())
    checks["original_similarity_exact"] = bool(
        np.allclose(old_scores, new_scores, rtol=0.0, atol=0.0, equal_nan=True))
    checks["original_accept_reject_unchanged"] = decision_changes == 0
    checks["old_threshold_source_correct"] = (
        old_thresholds["threshold_source_condition"] == "Original"
        and old_config["threshold_strategy"]["source_split"] == "development"
        and old_config["threshold_strategy"]["source_condition"] == "Original"
        and old_config["threshold_strategy"]["applied_to"] == "all evaluation conditions"
        and old_thresholds["config_fingerprint"] == old_config["config_fingerprint"]
    )

    old_selected = old[old["quality"].isin(OLD_QUALITIES)].copy()
    old_selected["source_run"] = OLD_RUN
    new_selected = new[new["quality"].isin(["10", "8", "6", "4"])].copy()
    new_selected["source_run"] = NEW_RUN
    combined = pd.concat([old_selected, new_selected], ignore_index=True)
    combined["quality"] = combined["quality"].astype(str)
    combined["threshold_source"] = threshold_source
    combined_keys = combined.groupby(
        ["split", "pair_id", "quality"], observed=True).size()
    checks["combined_11_conditions_exact"] = set(
        combined["quality"].unique()) == set(QUALITIES)
    checks["combined_expected_rows"] = len(combined) == EXPECTED_PAIRS * len(QUALITIES)
    checks["combined_no_duplicate_pair_condition"] = bool((combined_keys == 1).all())
    combined_sets = combined.groupby(
        ["split", "pair_id"], observed=True)["quality"].agg(frozenset)
    checks["combined_every_pair_has_11_conditions"] = bool(
        (combined_sets == frozenset(QUALITIES)).all())
    checks["combined_original_only_from_old_run"] = set(
        combined.loc[combined["quality"] == "Original", "source_run"]) == {OLD_RUN}
    combined_invariance = combined.groupby(
        ["split", "pair_id"], observed=True)[invariant_columns].nunique(dropna=False)
    checks["combined_topology_invariant_all_11_conditions"] = bool(
        (combined_invariance == 1).all().all())

    if not all(checks.values()):
        audit = {
            "audit_passed": False,
            "checks": checks,
            "protocol_digests": protocol_digests,
            "original_consistency": original_difference,
            "original_invariants": original_invariants,
        }
        (NEW_OUT / "combined_transition_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    evaluation = combined[combined["split"] == "evaluation"].copy()
    subjects = np.array(sorted(evaluation["probe_subject_id"].unique()))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(
        0, len(subjects), size=(BOOTSTRAP_ITERATIONS, len(subjects)),
        endpoint=False)
    cluster_weights = np.zeros(
        (BOOTSTRAP_ITERATIONS, len(subjects)), dtype=np.int16)
    for index, row in enumerate(sampled):
        cluster_weights[index] = np.bincount(row, minlength=len(subjects))

    point_by_quality = {}
    bootstrap_by_quality = {}
    valid_rows_by_quality = {}
    for quality in QUALITIES:
        frame = evaluation[evaluation["quality"] == quality].sort_values(
            "pair_id", kind="stable").reset_index(drop=True)
        point, bootstrap, valid_rows = compute_condition(
            frame, old_threshold, subjects, cluster_weights)
        point_by_quality[quality] = point
        bootstrap_by_quality[quality] = bootstrap
        valid_rows_by_quality[quality] = valid_rows

    original_size_point = point_by_quality["Original"]["average_probe_size_bytes"]
    original_size_boot = bootstrap_by_quality["Original"]["average_probe_size_bytes"]
    for quality in QUALITIES:
        point_by_quality[quality]["storage_saving"] = (
            1.0 - point_by_quality[quality]["average_probe_size_bytes"]
            / original_size_point)
        bootstrap_by_quality[quality]["storage_saving"] = (
            1.0 - bootstrap_by_quality[quality]["average_probe_size_bytes"]
            / original_size_boot)

    metric_names = [
        "FMR", "FNMR", "TAR", "EER", "mean_genuine_similarity",
        "mean_impostor_similarity", "average_probe_size_bytes", "storage_saving",
    ]
    source_by_quality = {
        quality: (OLD_RUN if quality in OLD_QUALITIES else NEW_RUN)
        for quality in QUALITIES
    }
    fingerprint_by_quality = {
        quality: (
            old_config["config_fingerprint"]
            if quality in OLD_QUALITIES else new_config["config_fingerprint"])
        for quality in QUALITIES
    }
    metrics_rows = []
    for quality in QUALITIES:
        row = {
            "quality": quality,
            "source_run": source_by_quality[quality],
            "config_fingerprint": fingerprint_by_quality[quality],
            "threshold_source": threshold_source,
            "fixed_threshold": old_threshold,
            "evaluation_rows": int(
                (evaluation["quality"] == quality).sum()),
            "valid_similarity_rows": valid_rows_by_quality[quality],
            "bootstrap_unit": "evaluation probe_subject_id cluster",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }
        for metric in metric_names:
            values = bootstrap_by_quality[quality][metric]
            row[metric] = point_by_quality[quality][metric]
            row[f"{metric}_ci_low"] = float(np.nanpercentile(values, 2.5))
            row[f"{metric}_ci_high"] = float(np.nanpercentile(values, 97.5))
        metrics_rows.append(row)
    metrics_table = pd.DataFrame(metrics_rows)
    metrics_path = NEW_OUT / "combined_transition_metrics_q15_q4.csv"
    metrics_table.to_csv(metrics_path, index=False)

    paired_rows = []
    for condition, reference in CONTRASTS:
        for metric in metric_names:
            differences = (
                bootstrap_by_quality[condition][metric]
                - bootstrap_by_quality[reference][metric])
            paired_rows.append({
                "contrast": f"Q{condition}-Original" if reference == "Original"
                else f"Q{condition}-Q{reference}",
                "condition": condition,
                "reference_condition": reference,
                "metric": metric,
                "difference": (
                    point_by_quality[condition][metric]
                    - point_by_quality[reference][metric]),
                "ci_low": float(np.nanpercentile(differences, 2.5)),
                "ci_high": float(np.nanpercentile(differences, 97.5)),
                "bootstrap_unit": "evaluation probe_subject_id cluster",
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "threshold_source": threshold_source,
                "fixed_threshold": old_threshold,
                "condition_source_run": source_by_quality[condition],
                "reference_source_run": source_by_quality[reference],
                "condition_config_fingerprint": fingerprint_by_quality[condition],
                "reference_config_fingerprint": fingerprint_by_quality[reference],
            })
    paired_table = pd.DataFrame(paired_rows)
    paired_path = NEW_OUT / "combined_transition_paired_bootstrap.csv"
    paired_table.to_csv(paired_path, index=False)

    checks["combined_metrics_has_11_unique_rows"] = (
        len(metrics_table) == 11 and metrics_table["quality"].nunique() == 11)
    checks["combined_metrics_uses_old_threshold_all_rows"] = bool(
        (metrics_table["fixed_threshold"] == old_threshold).all()
        and (metrics_table["threshold_source"] == threshold_source).all())
    checks["combined_metrics_source_fields_complete"] = bool(
        metrics_table[["source_run", "config_fingerprint", "threshold_source"]]
        .notna().all().all())
    checks["paired_bootstrap_has_10_by_8_rows"] = len(paired_table) == 80
    checks["paired_bootstrap_1000_subject_clusters"] = bool(
        (paired_table["bootstrap_iterations"] == 1000).all()
        and (paired_table["bootstrap_unit"]
             == "evaluation probe_subject_id cluster").all())
    checks["output_csv_files_exist"] = metrics_path.exists() and paired_path.exists()

    audit = {
        "audit_passed": all(checks.values()),
        "checks": checks,
        "new_run": {
            "config_fingerprint": new_config["config_fingerprint"],
            "results_rows": int(len(new)),
            "unique_pairs": int(new["pair_id"].nunique()),
            "qualities": NEW_QUALITIES,
            "embedding_device_evidence": {
                "embedding_execution_stdout": "cuda:0",
                "observed_resume_counts_with_cuda0": [1600, 75600, 94600],
                "exit_zero_finalization_report": "resume-complete",
                "exit_zero_finalization_resumed_rows": 95040,
            },
        },
        "old_run": {
            "config_fingerprint": old_config["config_fingerprint"],
            "fixed_threshold": old_threshold,
            "threshold_source": threshold_source,
        },
        "protocol_digests": protocol_digests,
        "original_consistency": original_difference,
        "original_invariants": original_invariants,
        "combined": {
            "qualities": QUALITIES,
            "raw_rows_audited_in_memory": int(len(combined)),
            "evaluation_probe_subject_clusters": int(len(subjects)),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "metrics_csv_sha256": sha256(metrics_path),
            "paired_bootstrap_csv_sha256": sha256(paired_path),
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    audit_path = NEW_OUT / "combined_transition_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    original_path = NEW_OUT / "combined_transition_original_consistency.json"
    original_path.write_text(json.dumps({
        "threshold_used": old_threshold,
        "threshold_source": threshold_source,
        "statistics": original_difference,
        "invariant_fields": original_invariants,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["audit_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
