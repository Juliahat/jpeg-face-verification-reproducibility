#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**32 - 1)


def cache_path(cache_dir: Path, role: str, condition: str) -> Path:
    safe = "original" if condition == "Original" else f"q{condition}"
    return cache_dir / f"{role}_embeddings_{safe}.npz"


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required embedding cache is absent: {path}")
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name].copy() for name in data.files}
    embeddings = result["embeddings"].astype(np.float32, copy=False)
    valid = result["valid"].astype(bool, copy=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512 or valid.shape != embeddings.shape[:1]:
        raise RuntimeError(f"Invalid cache shapes in {path}")
    if np.any(np.abs(np.linalg.norm(embeddings[valid], axis=1) - 1.0) > 2e-5):
        raise RuntimeError(f"Valid embeddings are not L2 normalised: {path}")
    return result


def load_assignment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    frame["subject_id"] = frame["subject_id"].astype(str)
    if set(frame["split"].unique()) != {"development", "evaluation"}:
        raise RuntimeError(f"Unexpected split labels in {path}")
    return frame


def mapping_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    columns = ["subject_id", "reference_path", "probe_paths", "num_probe_images"]
    result = frame.loc[:, columns].sort_values("subject_id", kind="stable").to_dict("records")
    for row in result:
        row["subject_id"] = str(row["subject_id"])
        row["num_probe_images"] = int(row["num_probe_images"])
    return result


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap_ratio(
    numerators: np.ndarray,
    denominators: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    n = len(numerators)
    if n == 0 or denominators.sum() == 0 or iterations <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=iterations)
    sampled_den = weights @ denominators
    sampled_num = weights @ numerators
    ratios = np.divide(
        sampled_num,
        sampled_den,
        out=np.full(iterations, np.nan, dtype=float),
        where=sampled_den > 0,
    )
    return tuple(np.nanquantile(ratios, [0.025, 0.975]).astype(float))


def cluster_bootstrap_delta(
    differences: np.ndarray,
    denominators: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    return cluster_bootstrap_ratio(differences, denominators, iterations, seed)


def calibrate_target_fmr(scores: np.ndarray, target: float) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("No valid Development impostor scores")
    if not 0.0 < target < 1.0:
        raise ValueError("Target FMR must lie strictly between zero and one")
    allowed = int(math.floor(target * len(scores)))
    descending = np.sort(scores)[::-1]
    if allowed == 0:
        threshold = float(np.nextafter(descending[0], np.inf))
        tie_adjusted = True
    else:
        candidate = float(descending[allowed - 1])
        at_or_above = int(np.count_nonzero(scores >= candidate))
        if at_or_above > allowed:
            threshold = float(np.nextafter(candidate, np.inf))
            tie_adjusted = True
        else:
            threshold = candidate
            tie_adjusted = False
    false_matches = int(np.count_nonzero(scores >= threshold))
    return {
        "target_fmr": float(target),
        "threshold": threshold,
        "valid_development_impostor_scores": int(len(scores)),
        "allowed_false_matches": allowed,
        "achieved_false_matches": false_matches,
        "achieved_fmr": float(false_matches / len(scores)),
        "empirical_resolution": float(1.0 / len(scores)),
        "tie_adjusted_upward": bool(tie_adjusted),
        "accept_rule": "cosine_similarity >= threshold",
    }


def score_matrix(
    probe_cache: Mapping[str, np.ndarray],
    reference_cache: Mapping[str, np.ndarray],
    probe_indices: np.ndarray,
    reference_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    probes = probe_cache["embeddings"][probe_indices]
    references = reference_cache["embeddings"][reference_indices]
    valid = (
        probe_cache["valid"][probe_indices, None].astype(bool)
        & reference_cache["valid"][reference_indices][None, :].astype(bool)
    )
    scores = probes @ references.T
    scores = scores.astype(np.float32, copy=False)
    scores[~valid] = np.nan
    return scores, valid


def nonmate_mask(probe_subjects: np.ndarray, reference_subjects: np.ndarray) -> np.ndarray:
    return probe_subjects[:, None] != reference_subjects[None, :]


def threshold_label(target: float) -> str:
    if math.isclose(target, 0.001):
        return "full_dev_fmr_0.1pct"
    if math.isclose(target, 0.0001):
        return "full_dev_fmr_0.01pct"
    return f"full_dev_fmr_{target:.8g}"


def metric_row(
    *,
    seed: int,
    condition: str,
    cohort: str,
    policy: str,
    threshold: float,
    attempted: np.ndarray,
    valid: np.ndarray,
    accepted: np.ndarray,
    bootstrap_iterations: int,
    bootstrap_base_seed: int,
) -> Dict[str, Any]:
    per_probe_attempted = attempted.sum(axis=1).astype(np.int64)
    per_probe_valid = valid.sum(axis=1).astype(np.int64)
    per_probe_false_matches = (accepted & valid).sum(axis=1).astype(np.int64)
    n_attempted = int(per_probe_attempted.sum())
    n_valid = int(per_probe_valid.sum())
    false_matches = int(per_probe_false_matches.sum())
    fmr = false_matches / n_valid if n_valid else float("nan")
    wilson_low, wilson_high = wilson_interval(false_matches, n_valid)
    cluster_low, cluster_high = cluster_bootstrap_ratio(
        per_probe_false_matches,
        per_probe_valid,
        bootstrap_iterations,
        stable_seed(bootstrap_base_seed, seed, condition, cohort, policy, "fmr"),
    )
    return {
        "seed": seed,
        "split": "evaluation",
        "condition": condition,
        "impostor_cohort": cohort,
        "threshold_policy": policy,
        "threshold": threshold,
        "probe_count": int(len(per_probe_attempted)),
        "attempted_impostor_pairs": n_attempted,
        "valid_impostor_scores": n_valid,
        "failed_impostor_comparisons": n_attempted - n_valid,
        "false_matches": false_matches,
        "FMR": fmr,
        "empirical_resolution": 1.0 / n_valid if n_valid else float("nan"),
        "score_level_wilson_ci_low": wilson_low,
        "score_level_wilson_ci_high": wilson_high,
        "probe_cluster_bootstrap_ci_low": cluster_low,
        "probe_cluster_bootstrap_ci_high": cluster_high,
        "probes_with_at_least_one_false_match": int(np.count_nonzero(per_probe_false_matches > 0)),
        "bootstrap_cluster": "probe image",
        "bootstrap_iterations": bootstrap_iterations,
    }


def genuine_row(
    *,
    seed: int,
    condition: str,
    policy: str,
    threshold: float,
    scores: np.ndarray,
    valid: np.ndarray,
    bootstrap_iterations: int,
    bootstrap_base_seed: int,
) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(scores)
    false_nonmatches = valid & (scores < threshold)
    n = len(scores)
    n_valid = int(valid.sum())
    fn = int(false_nonmatches.sum())
    failed = n - n_valid
    score_fnmr = fn / n_valid if n_valid else float("nan")
    attempt_fnmr = (fn + failed) / n if n else float("nan")
    low, high = cluster_bootstrap_ratio(
        false_nonmatches.astype(int),
        valid.astype(int),
        bootstrap_iterations,
        stable_seed(bootstrap_base_seed, seed, condition, policy, "fnmr"),
    )
    return {
        "seed": seed,
        "split": "evaluation",
        "condition": condition,
        "threshold_policy": policy,
        "threshold": threshold,
        "genuine_attempts": n,
        "valid_genuine_scores": n_valid,
        "failed_genuine_attempts": failed,
        "false_nonmatches": fn,
        "FNMR": score_fnmr,
        "FNMR_attempt": attempt_fnmr,
        "probe_cluster_bootstrap_ci_low": low,
        "probe_cluster_bootstrap_ci_high": high,
    }


def delta_row(
    *,
    seed: int,
    condition: str,
    cohort: str,
    policy: str,
    baseline_scores: np.ndarray,
    condition_scores: np.ndarray,
    attempted: np.ndarray,
    threshold: float,
    bootstrap_iterations: int,
    bootstrap_base_seed: int,
) -> Dict[str, Any]:
    common = attempted & np.isfinite(baseline_scores) & np.isfinite(condition_scores)
    base_accept = common & (baseline_scores >= threshold)
    cond_accept = common & (condition_scores >= threshold)
    gained = cond_accept & ~base_accept
    lost = base_accept & ~cond_accept
    per_probe_difference = gained.sum(axis=1).astype(float) - lost.sum(axis=1).astype(float)
    per_probe_common = common.sum(axis=1).astype(float)
    total_common = int(per_probe_common.sum())
    gain_count = int(gained.sum())
    loss_count = int(lost.sum())
    delta = (gain_count - loss_count) / total_common if total_common else float("nan")
    low, high = cluster_bootstrap_delta(
        per_probe_difference,
        per_probe_common,
        bootstrap_iterations,
        stable_seed(bootstrap_base_seed, seed, condition, cohort, policy, "delta"),
    )
    return {
        "seed": seed,
        "split": "evaluation",
        "condition": condition,
        "baseline_condition": "Original",
        "impostor_cohort": cohort,
        "threshold_policy": policy,
        "threshold": threshold,
        "common_valid_pairs": total_common,
        "reject_to_accept": gain_count,
        "accept_to_reject": loss_count,
        "paired_FMR_delta": delta,
        "probe_cluster_bootstrap_ci_low": low,
        "probe_cluster_bootstrap_ci_high": high,
        "bootstrap_iterations": bootstrap_iterations,
    }


def make_hard_indices(
    original_scores: np.ndarray,
    nonmates: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    eligible = nonmates & np.isfinite(original_scores)
    n_probes = original_scores.shape[0]
    indices = np.full((n_probes, top_k), -1, dtype=np.int64)
    valid = np.zeros((n_probes, top_k), dtype=bool)
    for row in range(n_probes):
        candidates = np.flatnonzero(eligible[row])
        if len(candidates) == 0:
            continue
        candidate_scores = original_scores[row, candidates]
        order = np.lexsort((candidates, -candidate_scores))
        selected = candidates[order[:top_k]]
        indices[row, : len(selected)] = selected
        valid[row, : len(selected)] = True
    return indices, valid


def gather_hard(scores: np.ndarray, indices: np.ndarray, membership: np.ndarray) -> np.ndarray:
    safe = indices.copy()
    safe[~membership] = 0
    gathered = scores[np.arange(scores.shape[0])[:, None], safe]
    gathered = gathered.astype(np.float32, copy=False)
    gathered[~membership] = np.nan
    return gathered


def validate_against_legacy(
    analysis_dir: Path,
    config: Mapping[str, Any],
    reference_cache: Mapping[str, np.ndarray],
    probe_caches: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    reference_lookup = {
        str(path): idx for idx, path in enumerate(reference_cache["image_path"].astype(str))
    }
    probe_lookup = {
        str(path): idx for idx, path in enumerate(probe_caches["Original"]["image_path"].astype(str))
    }
    loaded: Dict[Path, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []
    for condition in config["conditions"]:
        source = (analysis_dir / config["legacy_score_sources"][condition]).resolve()
        if source not in loaded:
            frame = pd.read_csv(
                source,
                dtype={"quality": str, "reference_subject_id": str, "probe_subject_id": str},
                usecols=[
                    "quality", "label", "reference_path", "probe_path",
                    "cosine_similarity", "comparison_success",
                ],
            )
            frame["cosine_similarity"] = pd.to_numeric(frame["cosine_similarity"], errors="coerce")
            loaded[source] = frame
        genuine = loaded[source][
            (loaded[source]["quality"].astype(str) == condition)
            & (loaded[source]["label"].astype(int) == 1)
        ].copy()
        computed = []
        for row in genuine.itertuples(index=False):
            ri = reference_lookup[str(row.reference_path)]
            pi = probe_lookup[str(row.probe_path)]
            if reference_cache["valid"][ri] and probe_caches[condition]["valid"][pi]:
                computed.append(float(reference_cache["embeddings"][ri] @ probe_caches[condition]["embeddings"][pi]))
            else:
                computed.append(float("nan"))
        saved = genuine["cosine_similarity"].to_numpy(dtype=float)
        computed_array = np.asarray(computed, dtype=float)
        jointly_finite = np.isfinite(saved) & np.isfinite(computed_array)
        nan_pattern_equal = bool(np.array_equal(np.isfinite(saved), np.isfinite(computed_array)))
        differences = np.abs(saved[jointly_finite] - computed_array[jointly_finite])
        rows.append({
            "condition": condition,
            "legacy_source": source.name,
            "genuine_rows_checked": int(len(genuine)),
            "jointly_finite_rows": int(jointly_finite.sum()),
            "finite_nan_pattern_equal": nan_pattern_equal,
            "max_absolute_cosine_difference": float(differences.max()) if len(differences) else float("nan"),
            "mean_absolute_cosine_difference": float(differences.mean()) if len(differences) else float("nan"),
            "within_1e_6": bool(nan_pattern_equal and (len(differences) == 0 or differences.max() <= 1e-6)),
        })
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, output_dir: Path, conditions: Sequence[str]) -> None:
    order = {condition: i for i, condition in enumerate(conditions)}
    selected = metrics[metrics["threshold_policy"] == "full_dev_fmr_0.1pct"].copy()
    grouped = selected.groupby(["condition", "impostor_cohort"], as_index=False)["FMR"].mean()
    grouped["order"] = grouped["condition"].map(order)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    panels = [
        ("all_nonmate", "All same-split non-mates", "#1f77b4"),
        ("hard_top10_fixed_on_original", "Fixed Original-selected hard top-10", "#ff7f0e"),
    ]
    for ax, (cohort, title, colour) in zip(axes, panels):
        group = grouped[grouped["impostor_cohort"] == cohort].sort_values("order")
        values = group["FMR"].to_numpy(dtype=float) * 100.0
        ax.plot(group["condition"], values, marker="o", linewidth=2.2, color=colour)
        spread = max(float(values.max() - values.min()), 1e-4)
        ax.set_ylim(max(0.0, float(values.min() - 0.25 * spread)), float(values.max() + 0.25 * spread))
        ax.set_ylabel("Mean seed-level FMR (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Probe condition")
    fig.suptitle("False-match analysis at full-Development 0.1% target", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "fmr_all_vs_hard_top10.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = analysis_dir / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((analysis_dir / "analysis_config.json").read_text(encoding="utf-8"))
    conditions: List[str] = list(config["conditions"])
    top_k = int(config["hard_impostor_top_k"])
    bootstrap_iterations = int(config["bootstrap_iterations"])
    bootstrap_base_seed = int(config["bootstrap_seed"])
    cache_dir = analysis_dir / "embedding_cache"

    reference_cache = load_cache(cache_path(cache_dir, "reference", "Original"))
    probe_caches = {
        condition: load_cache(cache_path(cache_dir, "probe", condition))
        for condition in conditions
    }
    reference_subjects_all = reference_cache["subject_id"].astype(str)
    probe_subjects_all = probe_caches["Original"]["subject_id"].astype(str)
    reference_lookup = {subject: idx for idx, subject in enumerate(reference_subjects_all)}
    reference_path_all = reference_cache["image_path"].astype(str)
    probe_path_all = probe_caches["Original"]["image_path"].astype(str)
    if len(reference_lookup) != len(reference_subjects_all):
        raise RuntimeError("Reference subject IDs are not unique")

    canonical_mapping = None
    threshold_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    genuine_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    hard_rows: List[Dict[str, Any]] = []
    protocol_rows: List[Dict[str, Any]] = []

    for seed_text, seed_config in config["seeds"].items():
        seed = int(seed_text)
        assignment_path = (analysis_dir / seed_config["assignment_relative_to_analysis_dir"]).resolve()
        assignment = load_assignment(assignment_path)
        mapping = mapping_records(assignment)
        if canonical_mapping is None:
            canonical_mapping = mapping
        elif mapping != canonical_mapping:
            raise RuntimeError(f"Reference/probe mapping differs for seed {seed}")
        split_by_subject = dict(zip(assignment["subject_id"], assignment["split"]))
        if set(reference_subjects_all) != set(split_by_subject):
            raise RuntimeError(f"Embedding subjects do not match assignment for seed {seed}")

        split_indices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for split in ("development", "evaluation"):
            ref_indices = np.asarray([
                i for i, subject in enumerate(reference_subjects_all)
                if split_by_subject[subject] == split
            ], dtype=np.int64)
            probe_indices = np.asarray([
                i for i, subject in enumerate(probe_subjects_all)
                if split_by_subject[subject] == split
            ], dtype=np.int64)
            split_indices[split] = (probe_indices, ref_indices)

        dev_probe_indices, dev_ref_indices = split_indices["development"]
        dev_probe_subjects = probe_subjects_all[dev_probe_indices]
        dev_ref_subjects = reference_subjects_all[dev_ref_indices]
        dev_scores, _ = score_matrix(
            probe_caches["Original"], reference_cache, dev_probe_indices, dev_ref_indices
        )
        dev_nonmate = nonmate_mask(dev_probe_subjects, dev_ref_subjects)
        dev_valid_impostors = dev_nonmate & np.isfinite(dev_scores)
        dev_impostor_scores = dev_scores[dev_valid_impostors]
        old_threshold_payload = json.loads(
            (analysis_dir / seed_config["threshold_relative_to_analysis_dir"]).resolve().read_text(encoding="utf-8")
        )
        policies: Dict[str, float] = {
            "legacy_sampled_eer": float(old_threshold_payload["threshold"]),
        }
        threshold_rows.append({
            "seed": seed,
            "threshold_policy": "legacy_sampled_eer",
            "target_fmr": float("nan"),
            "threshold": policies["legacy_sampled_eer"],
            "valid_development_impostor_scores": int(len(dev_impostor_scores)),
            "allowed_false_matches": float("nan"),
            "achieved_false_matches": int(np.count_nonzero(dev_impostor_scores >= policies["legacy_sampled_eer"])),
            "achieved_fmr": float(np.mean(dev_impostor_scores >= policies["legacy_sampled_eer"])),
            "empirical_resolution": float(1.0 / len(dev_impostor_scores)),
            "tie_adjusted_upward": False,
            "calibration_note": "Threshold copied from original 10-random-impostor Development EER analysis; achieved FMR recomputed here on all Development non-mates.",
        })
        for target in config["target_fmrs"]:
            calibration = calibrate_target_fmr(dev_impostor_scores, float(target))
            label = threshold_label(float(target))
            policies[label] = float(calibration["threshold"])
            threshold_rows.append({
                "seed": seed,
                "threshold_policy": label,
                **calibration,
                "calibration_note": "Calibrated on all valid Development/Original non-mate scores only.",
            })

        eval_probe_indices, eval_ref_indices = split_indices["evaluation"]
        eval_probe_subjects = probe_subjects_all[eval_probe_indices]
        eval_ref_subjects = reference_subjects_all[eval_ref_indices]
        eval_nonmate = nonmate_mask(eval_probe_subjects, eval_ref_subjects)
        original_scores, _ = score_matrix(
            probe_caches["Original"], reference_cache, eval_probe_indices, eval_ref_indices
        )
        hard_indices, hard_membership = make_hard_indices(original_scores, eval_nonmate, top_k)
        original_hard_scores = gather_hard(original_scores, hard_indices, hard_membership)
        for probe_row, global_probe_index in enumerate(eval_probe_indices):
            for rank in range(top_k):
                if not hard_membership[probe_row, rank]:
                    continue
                local_ref_index = int(hard_indices[probe_row, rank])
                global_ref_index = int(eval_ref_indices[local_ref_index])
                hard_rows.append({
                    "seed": seed,
                    "split": "evaluation",
                    "selection_condition": "Original",
                    "selection_method": "probe-to-reference cosine nearest non-mates",
                    "rank": rank + 1,
                    "probe_subject_id": str(eval_probe_subjects[probe_row]),
                    "probe_path": str(probe_path_all[global_probe_index]),
                    "reference_subject_id": str(eval_ref_subjects[local_ref_index]),
                    "reference_path": str(reference_path_all[global_ref_index]),
                    "original_cosine_similarity": float(original_scores[probe_row, local_ref_index]),
                })

        protocol_rows.append({
            "seed": seed,
            "development_references": int(len(dev_ref_indices)),
            "development_probes": int(len(dev_probe_indices)),
            "development_all_nonmate_attempts": int(dev_nonmate.sum()),
            "development_valid_original_nonmate_scores": int(dev_valid_impostors.sum()),
            "evaluation_references": int(len(eval_ref_indices)),
            "evaluation_probes": int(len(eval_probe_indices)),
            "evaluation_all_nonmate_attempts": int(eval_nonmate.sum()),
            "evaluation_hard_top_k_attempts": int(hard_membership.sum()),
        })

        original_by_cohort = {
            "all_nonmate": (original_scores, eval_nonmate),
            f"hard_top{top_k}_fixed_on_original": (original_hard_scores, hard_membership),
        }
        for condition in conditions:
            condition_scores, _ = score_matrix(
                probe_caches[condition], reference_cache, eval_probe_indices, eval_ref_indices
            )
            hard_scores = gather_hard(condition_scores, hard_indices, hard_membership)
            cohort_data = {
                "all_nonmate": (condition_scores, eval_nonmate),
                f"hard_top{top_k}_fixed_on_original": (hard_scores, hard_membership),
            }
            mate_local_indices = np.asarray(
                [{subject: i for i, subject in enumerate(eval_ref_subjects)}[subject]
                 for subject in eval_probe_subjects],
                dtype=np.int64,
            )
            genuine_scores = condition_scores[np.arange(len(eval_probe_indices)), mate_local_indices]
            genuine_valid = np.isfinite(genuine_scores)
            for policy, threshold in policies.items():
                for cohort, (scores, attempted) in cohort_data.items():
                    valid = attempted & np.isfinite(scores)
                    accepted = valid & (scores >= threshold)
                    metric_rows.append(metric_row(
                        seed=seed,
                        condition=condition,
                        cohort=cohort,
                        policy=policy,
                        threshold=threshold,
                        attempted=attempted,
                        valid=valid,
                        accepted=accepted,
                        bootstrap_iterations=bootstrap_iterations,
                        bootstrap_base_seed=bootstrap_base_seed,
                    ))
                    baseline_scores, baseline_attempted = original_by_cohort[cohort]
                    delta_rows.append(delta_row(
                        seed=seed,
                        condition=condition,
                        cohort=cohort,
                        policy=policy,
                        baseline_scores=baseline_scores,
                        condition_scores=scores,
                        attempted=attempted & baseline_attempted,
                        threshold=threshold,
                        bootstrap_iterations=bootstrap_iterations,
                        bootstrap_base_seed=bootstrap_base_seed,
                    ))
                genuine_rows.append(genuine_row(
                    seed=seed,
                    condition=condition,
                    policy=policy,
                    threshold=threshold,
                    scores=genuine_scores,
                    valid=genuine_valid,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_base_seed=bootstrap_base_seed,
                ))
        print(f"Completed seed {seed}", flush=True)

    thresholds = pd.DataFrame(threshold_rows).sort_values(["seed", "threshold_policy"])
    metrics = pd.DataFrame(metric_rows)
    genuine = pd.DataFrame(genuine_rows)
    deltas = pd.DataFrame(delta_rows)
    hard_pairs = pd.DataFrame(hard_rows)
    protocols = pd.DataFrame(protocol_rows).sort_values("seed")
    condition_order = {condition: i for i, condition in enumerate(conditions)}
    for frame in (metrics, genuine, deltas):
        frame["_condition_order"] = frame["condition"].map(condition_order)
    metrics = metrics.sort_values(["seed", "threshold_policy", "impostor_cohort", "_condition_order"]).drop(columns="_condition_order")
    genuine = genuine.sort_values(["seed", "threshold_policy", "_condition_order"]).drop(columns="_condition_order")
    deltas = deltas.sort_values(["seed", "threshold_policy", "impostor_cohort", "_condition_order"]).drop(columns="_condition_order")

    thresholds.to_csv(output_dir / "development_thresholds_all_nonmates.csv", index=False)
    protocols.to_csv(output_dir / "protocol_pair_counts.csv", index=False)
    hard_pairs.to_csv(output_dir / "hard_impostor_pairs_fixed_on_original.csv", index=False)
    metrics.to_csv(output_dir / "evaluation_fmr_all_and_hard.csv", index=False)
    genuine.to_csv(output_dir / "evaluation_fnmr_same_thresholds.csv", index=False)
    deltas.to_csv(output_dir / "paired_fmr_delta_vs_original.csv", index=False)

    cross_seed = metrics.groupby(
        ["condition", "impostor_cohort", "threshold_policy"], as_index=False
    ).agg(
        seed_count=("seed", "nunique"),
        mean_seed_FMR=("FMR", "mean"),
        min_seed_FMR=("FMR", "min"),
        max_seed_FMR=("FMR", "max"),
        min_false_matches=("false_matches", "min"),
        max_false_matches=("false_matches", "max"),
        min_valid_scores=("valid_impostor_scores", "min"),
        max_valid_scores=("valid_impostor_scores", "max"),
    )
    cross_seed["_condition_order"] = cross_seed["condition"].map(condition_order)
    cross_seed = cross_seed.sort_values(
        ["threshold_policy", "impostor_cohort", "_condition_order"]
    ).drop(columns="_condition_order")
    cross_seed.to_csv(output_dir / "cross_seed_fmr_summary_descriptive.csv", index=False)

    legacy_validation = validate_against_legacy(
        analysis_dir, config, reference_cache, probe_caches
    )
    legacy_validation.to_csv(output_dir / "legacy_genuine_score_reproduction.csv", index=False)
    plot_metrics(metrics, output_dir, conditions)

    manifest = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "output_manifest_sha256.json":
            manifest.append({
                "filename": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            })
    summary = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "all same-split non-mate references plus fixed Original-selected hard top-k stress cohort",
        "seeds": [int(value) for value in config["seeds"]],
        "conditions": conditions,
        "hard_impostor_top_k": top_k,
        "hard_selection": "For each Evaluation probe and seed, rank valid different-identity references by Original cosine similarity; use top-k with reference-index tie break; freeze exact pairs for every JPEG condition.",
        "hard_condition_interpretation": "Stress test conditional on difficult non-mates; not a population FMR estimate.",
        "threshold_source": "Seed-local Development Original only; Evaluation never calibrates thresholds.",
        "threshold_policies": ["legacy_sampled_eer", "full_dev_fmr_0.1pct", "full_dev_fmr_0.01pct"],
        "fmr_denominator": "valid impostor scores; failed comparisons are separately counted",
        "fnmr_denominators": "FNMR uses valid genuine scores; FNMR_attempt counts failed genuine attempts as non-matches",
        "uncertainty": "Wilson score interval is descriptive at score level; primary dependence-aware interval resamples probe images as clusters.",
        "cross_seed_note": "Five seed splits reuse identities and embeddings; seed summaries are descriptive and are not pooled as independent trials.",
        "legacy_reproduction_all_conditions_within_1e_6": bool(legacy_validation["within_1e_6"].all()),
        "raw_images_or_face_crops_saved": False,
    }
    atomic_json(output_dir / "analysis_run_summary.json", summary)
    manifest.append({
        "filename": "analysis_run_summary.json",
        "size_bytes": int((output_dir / "analysis_run_summary.json").stat().st_size),
        "sha256": sha256_file(output_dir / "analysis_run_summary.json"),
    })
    atomic_json(output_dir / "output_manifest_sha256.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
