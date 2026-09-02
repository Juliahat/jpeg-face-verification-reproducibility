#!/usr/bin/env python3
"""Formal, local-only Color FERET JPEG face-verification experiment.

This runner builds on ``feret_feasibility_check.py``. It reuses the proven tar
inspection, filename parsing, and frontal reference/probe eligibility logic,
then adds subject-disjoint development/evaluation splits, a fixed development
threshold, an Original baseline, resumable embeddings, biometric metrics, and
formal plots. It never saves or displays a face image.
"""

import argparse
import bz2
import hashlib
import io
import json
import math
import os
import random
import sys
import tarfile
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from feret_feasibility_check import (
    FRONTAL_CODES,
    NON_FRONTAL_CODES,
    build_manifest,
    inspect_tar,
    read_feret_partitions,
    select_eligible_subjects,
)


ASSIGNMENT_COLUMNS = [
    "subject_id",
    "reference_path",
    "reference_category",
    "probe_paths",
    "num_probe_images",
    "split",
]
FORMAL_PAIR_COLUMNS = [
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
FORMAL_RESULT_COLUMNS = [
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
FORMAL_SKIPPED_COLUMNS = [
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
FINAL_METRIC_COLUMNS = [
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
FAILURE_REASONS = {
    "image_read_failed",
    "jpeg_compression_failed",
    "reference_detection_failed",
    "probe_detection_failed",
    "embedding_failed",
    "cosine_nan",
    "unknown_error",
}
IMPOSTOR_SAMPLING_RULE = (
    "for each genuine probe, seeded sampling without replacement from "
    "different-subject references within the same split"
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _decode_probe_paths(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        decoded = json.loads(str(value))
        return [str(item) for item in decoded] if isinstance(decoded, list) else []
    except (json.JSONDecodeError, TypeError):
        return [part for part in str(value).split(";") if part]


def _atomic_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def normalize_quality_conditions(values: Sequence[str]) -> List[str]:
    conditions: List[str] = []
    for value in values:
        token = str(value).strip()
        if token.lower() == "original":
            condition = "Original"
        else:
            try:
                quality = int(token)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    "Quality conditions must be Original or integers from 1 to 100"
                ) from exc
            if not 1 <= quality <= 100:
                raise argparse.ArgumentTypeError("JPEG qualities must be between 1 and 100")
            condition = str(quality)
        if condition not in conditions:
            conditions.append(condition)
    if not conditions:
        raise argparse.ArgumentTypeError("At least one quality condition is required")
    return conditions


def _condition_key(condition: str) -> Tuple[int, int]:
    return (0, 0) if condition == "Original" else (1, -int(condition))


def _condition_order_map(conditions: Sequence[str]) -> Dict[str, int]:
    return {condition: index for index, condition in enumerate(conditions)}


def _dataframe_digest(dataframe: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = dataframe.loc[:, list(columns)].copy() if not dataframe.empty else pd.DataFrame(columns=columns)
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_reference_probe_assignment(
    eligible_subjects: pd.DataFrame,
    seed: int,
    max_identities: Optional[int],
    full_experiment: bool,
) -> Tuple[pd.DataFrame, Set[str]]:
    """Select eligible identities without changing their reference/probe rule."""
    eligible = eligible_subjects[eligible_subjects["eligible"].map(_as_bool)].copy()
    eligible["subject_id"] = eligible["subject_id"].astype(str)
    eligible = eligible.sort_values("subject_id", kind="stable").reset_index(drop=True)
    all_eligible_ids = set(eligible["subject_id"].tolist())
    if not full_experiment:
        limit = 50 if max_identities is None else int(max_identities)
        if limit <= 0:
            raise ValueError("--max-identities must be positive")
        if len(eligible) > limit:
            indices = sorted(random.Random(int(seed)).sample(range(len(eligible)), limit))
            eligible = eligible.iloc[indices].copy().reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for record in eligible.to_dict(orient="records"):
        probes = _decode_probe_paths(record.get("probe_paths"))
        rows.append(
            {
                "subject_id": str(record["subject_id"]),
                "reference_path": str(record["reference_path"]),
                "reference_category": str(record.get("reference_category", "unknown")),
                "probe_paths": json.dumps(probes, ensure_ascii=False),
                "num_probe_images": int(len(probes)),
                "split": "unassigned",
            }
        )
    assignment = pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    return assignment, all_eligible_ids


def split_subjects_dev_eval(
    assignment: pd.DataFrame,
    seed: int,
    dev_ratio: float = 0.20,
) -> pd.DataFrame:
    """Create a reproducible identity-disjoint development/evaluation split."""
    if not 0.0 < float(dev_ratio) < 1.0:
        raise ValueError("Development split ratio must be between 0 and 1")
    if len(assignment) < 4:
        raise ValueError(
            "At least four eligible identities are required for two identity-disjoint splits"
        )
    subject_ids = sorted(assignment["subject_id"].astype(str).tolist())
    random.Random(int(seed)).shuffle(subject_ids)
    dev_count = int(math.floor(len(subject_ids) * float(dev_ratio) + 0.5))
    dev_count = max(2, min(dev_count, len(subject_ids) - 2))
    dev_ids = set(subject_ids[:dev_count])
    result = assignment.copy()
    result["split"] = result["subject_id"].map(
        lambda subject_id: "development" if str(subject_id) in dev_ids else "evaluation"
    )
    if set(result.loc[result["split"] == "development", "subject_id"]) & set(
        result.loc[result["split"] == "evaluation", "subject_id"]
    ):
        raise AssertionError("Subject leakage detected between development and evaluation")
    return result.sort_values(["split", "subject_id"], kind="stable").reset_index(drop=True)


def _path_category_lookup(manifest: pd.DataFrame) -> Dict[str, str]:
    return {
        str(row["internal_path"]): str(row.get("category_code", "unknown"))
        for row in manifest.to_dict(orient="records")
    }


def build_impostor_sampled_pairs(
    split_assignment: pd.DataFrame,
    split_name: str,
    seed: int,
    path_categories: Mapping[str, str],
    impostors_per_probe: int = 1,
) -> Tuple[pd.DataFrame, int]:
    """Generate one genuine and up to K unique-subject impostors per probe."""
    requested_k = int(impostors_per_probe)
    if requested_k <= 0:
        raise ValueError("impostors_per_probe must be positive")
    selected = split_assignment[split_assignment["split"] == split_name].copy()
    selected = selected.sort_values("subject_id", kind="stable").reset_index(drop=True)
    if len(selected) < 2:
        raise ValueError("{} split needs at least two identities".format(split_name))
    effective_k = min(requested_k, len(selected) - 1)
    if effective_k < requested_k:
        print(
            "Warning: {} split has only {} identities; requested {} impostors per probe, "
            "so effective_k={} (no cross-split or repeated identities).".format(
                split_name, len(selected), requested_k, effective_k
            ),
            file=sys.stderr,
        )
    references = {
        str(row["subject_id"]): str(row["reference_path"])
        for row in selected.to_dict(orient="records")
    }
    reference_categories = {
        str(row["subject_id"]): str(row["reference_category"])
        for row in selected.to_dict(orient="records")
    }
    rng = random.Random(int(seed))
    rows: List[Dict[str, Any]] = []
    pair_index = 0
    prefix = "dev" if split_name == "development" else "eval"
    for record in selected.to_dict(orient="records"):
        probe_subject = str(record["subject_id"])
        other_subjects = sorted(subject for subject in references if subject != probe_subject)
        for probe_path in _decode_probe_paths(record["probe_paths"]):
            pair_index += 1
            rows.append(
                {
                    "pair_id": "{}_P{:07d}".format(prefix, pair_index),
                    "split": split_name,
                    "label": 1,
                    "label_name": "genuine",
                    "probe_subject_id": probe_subject,
                    "reference_subject_id": probe_subject,
                    "probe_path": probe_path,
                    "reference_path": references[probe_subject],
                    "probe_category": path_categories.get(probe_path, "unknown"),
                    "reference_category": reference_categories[probe_subject],
                    "pair_type": "genuine",
                }
            )
            for impostor_subject in rng.sample(other_subjects, effective_k):
                pair_index += 1
                rows.append(
                    {
                        "pair_id": "{}_P{:07d}".format(prefix, pair_index),
                        "split": split_name,
                        "label": 0,
                        "label_name": "impostor",
                        "probe_subject_id": probe_subject,
                        "reference_subject_id": impostor_subject,
                        "probe_path": probe_path,
                        "reference_path": references[impostor_subject],
                        "probe_category": path_categories.get(probe_path, "unknown"),
                        "reference_category": reference_categories[impostor_subject],
                        "pair_type": "impostor",
                    }
                )
    pairs = pd.DataFrame(rows, columns=FORMAL_PAIR_COLUMNS)
    if pairs["pair_id"].duplicated().any():
        raise AssertionError("Pair IDs are not unique")
    duplicate_links = pairs.duplicated(
        subset=["split", "probe_path", "reference_subject_id"], keep=False
    )
    if duplicate_links.any():
        raise AssertionError(
            "Duplicate (split, probe_path, reference_subject_id) comparisons detected"
        )
    return pairs, effective_k


def build_excluded_images(
    manifest: pd.DataFrame,
    assignment: pd.DataFrame,
    all_eligible_ids: Set[str],
) -> pd.DataFrame:
    """Record why every image not used by the frozen assignment was excluded."""
    selected_subjects = set(assignment["subject_id"].astype(str))
    selected_paths: Set[str] = set(assignment["reference_path"].astype(str))
    for value in assignment["probe_paths"]:
        selected_paths.update(_decode_probe_paths(value))
    has_primary_tree = (
        "is_primary_dataset_image" in manifest.columns
        and manifest["is_primary_dataset_image"].map(_as_bool).any()
    )
    rows: List[Dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        path = str(record["internal_path"])
        subject = str(record.get("subject_id", "unknown"))
        category = str(record.get("category_code", "unknown"))
        frontal = str(record.get("is_frontal_candidate", "unknown")).lower()
        primary = _as_bool(record.get("is_primary_dataset_image", False))
        reason: Optional[str] = None
        if has_primary_tree and not primary:
            reason = "non_primary_dataset_image"
        elif subject == "unknown":
            reason = "subject_id_unparsed"
        elif path in selected_paths:
            continue
        elif subject in all_eligible_ids and subject not in selected_subjects:
            reason = "subject_not_selected_debug_limit"
        elif frontal == "false":
            reason = "non_frontal_category"
        elif frontal == "unknown":
            reason = "uncertain_frontal_category"
        elif subject not in all_eligible_ids:
            reason = "subject_insufficient_frontal_images"
        else:
            reason = "unassigned_image"
        rows.append(
            {
                "image_path": path,
                "subject_id": subject,
                "category_code": category,
                "reason": reason,
                "dataset_role": str(record.get("dataset_role", "unknown")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["image_path", "subject_id", "category_code", "reason", "dataset_role"],
    )


def write_pair_generation_summary(
    assignment: pd.DataFrame,
    dev_pairs: pd.DataFrame,
    eval_pairs: pd.DataFrame,
    seed: int,
    requested_impostors_per_probe: int,
    effective_impostors_per_probe: Mapping[str, int],
    output_path: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, pairs in (("development", dev_pairs), ("evaluation", eval_pairs)):
        genuine_count = int((pairs["label"] == 1).sum())
        impostor_count = int((pairs["label"] == 0).sum())
        rows.append(
            {
                "split": split_name,
                "num_identities": int((assignment["split"] == split_name).sum()),
                "requested_impostors_per_probe": int(requested_impostors_per_probe),
                "effective_impostors_per_probe": int(
                    effective_impostors_per_probe[split_name]
                ),
                "num_genuine_pairs": genuine_count,
                "num_impostor_pairs": impostor_count,
                "num_total_pairs": int(len(pairs)),
                "realised_impostor_to_genuine_ratio": (
                    float(impostor_count / genuine_count) if genuine_count else np.nan
                ),
                "seed": int(seed),
                "impostor_sampling_rule": IMPOSTOR_SAMPLING_RULE,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


def write_experiment_config(
    output_path: Path,
    stable_config: Mapping[str, Any],
    partial_results_path: Path,
) -> Dict[str, Any]:
    canonical = json.dumps(stable_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    timestamp = _now_utc()
    payload: Dict[str, Any] = dict(stable_config)
    payload["config_fingerprint"] = fingerprint
    payload["run_timestamp_utc"] = timestamp
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_fingerprint = existing.get("config_fingerprint")
        if existing_fingerprint != fingerprint and partial_results_path.exists():
            raise RuntimeError(
                "The work directory contains partial results from a different configuration. "
                "Use a new --work-dir to avoid mixing experiments."
            )
        if existing_fingerprint == fingerprint:
            payload["run_timestamp_utc"] = existing.get("run_timestamp_utc", timestamp)
            payload["last_resumed_timestamp_utc"] = timestamp
    _atomic_json(payload, output_path)
    return payload


class ArchiveImageStore:
    """Read selected images and exact uncompressed member sizes without saving them."""

    def __init__(self, tar_path: Path, cache_size: int = 8):
        self.tar_path = Path(tar_path)
        self.cache_size = max(1, int(cache_size))
        self.archive: Optional[tarfile.TarFile] = None
        self.cache: "OrderedDict[str, Tuple[Image.Image, int]]" = OrderedDict()

    def __enter__(self) -> "ArchiveImageStore":
        self.archive = tarfile.open(str(self.tar_path), mode="r:*")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.archive is not None:
            self.archive.close()
        self.archive = None
        self.cache.clear()

    def load(self, internal_path: str) -> Tuple[Image.Image, int]:
        if internal_path in self.cache:
            image, size = self.cache.pop(internal_path)
            self.cache[internal_path] = (image, size)
            return image.copy(), int(size)
        if self.archive is None:
            raise RuntimeError("ArchiveImageStore must be used as a context manager")
        member = self.archive.getmember(internal_path)
        if not member.isfile():
            raise OSError("Tar member is not a regular file")
        source = self.archive.extractfile(member)
        if source is None:
            raise OSError("Tar member could not be read")
        with source:
            payload = source.read()
        if internal_path.lower().endswith(".bz2"):
            payload = bz2.decompress(payload)
        original_size = len(payload)
        with Image.open(io.BytesIO(payload)) as loaded:
            image = loaded.convert("RGB")
            image.load()
        self.cache[internal_path] = (image, original_size)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return image.copy(), int(original_size)


def apply_jpeg_condition(
    image: Image.Image,
    condition: str,
    original_size_bytes: int,
) -> Tuple[Image.Image, int]:
    """Apply Original or a reproducible JPEG condition entirely in memory."""
    if condition == "Original":
        return image.copy(), int(original_size_bytes)
    quality = int(condition)
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    size_bytes = buffer.tell()
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        result = decoded.convert("RGB")
        result.load()
    return result, int(size_bytes)


def extract_facenet_embedding(face: Any, model: Any, device: Any) -> np.ndarray:
    """Embed one MTCNN crop and L2-normalize it."""
    import torch

    if face is None:
        raise ValueError("No face tensor supplied")
    if face.ndim == 4:
        face = face[0]
    with torch.no_grad():
        vector = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Embedding norm is zero or non-finite")
    return vector.astype(np.float32) / norm


def _largest_box(boxes: Any) -> Optional[np.ndarray]:
    if boxes is None:
        return None
    array = np.asarray(boxes, dtype=float)
    if array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    areas = np.maximum(0.0, array[:, 2] - array[:, 0]) * np.maximum(
        0.0, array[:, 3] - array[:, 1]
    )
    return array[[int(np.argmax(areas))]]


def _process_image_condition(
    store: ArchiveImageStore,
    image_path: str,
    condition: str,
    image_role: str,
    mtcnn: Any,
    model: Any,
    device: Any,
    preprocess_order: str,
) -> Dict[str, Any]:
    result = {
        "embedding": None,
        "detected": False,
        "size_bytes": np.nan,
        "failure_reason": "",
        "detail": "",
    }
    try:
        original, original_size = store.load(image_path)
    except Exception as exc:
        result["failure_reason"] = "image_read_failed"
        result["detail"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
        return result

    try:
        if preprocess_order == "detect-before-compress":
            try:
                boxes, _probabilities = mtcnn.detect(original)
                selected_box = _largest_box(boxes)
            except Exception as exc:
                selected_box = None
                result["detail"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
            if selected_box is None:
                result["failure_reason"] = "{}_detection_failed".format(image_role)
                return result
            try:
                conditioned, size_bytes = apply_jpeg_condition(
                    original, condition, original_size
                )
                result["size_bytes"] = size_bytes
            except Exception as exc:
                result["failure_reason"] = "jpeg_compression_failed"
                result["detail"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
                return result
            try:
                face = mtcnn.extract(conditioned, selected_box, save_path=None)
            except Exception as exc:
                result["failure_reason"] = "{}_detection_failed".format(image_role)
                result["detail"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
                return result
        else:
            try:
                conditioned, size_bytes = apply_jpeg_condition(
                    original, condition, original_size
                )
                result["size_bytes"] = size_bytes
            except Exception as exc:
                result["failure_reason"] = "jpeg_compression_failed"
                result["detail"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
                return result
            try:
                face = mtcnn(conditioned)
            except Exception as exc:
                result["failure_reason"] = "{}_detection_failed".format(image_role)
                result["detail"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
                return result

        if face is None or (hasattr(face, "numel") and face.numel() == 0):
            result["failure_reason"] = "{}_detection_failed".format(image_role)
            return result
        result["detected"] = True
        try:
            result["embedding"] = extract_facenet_embedding(face, model, device)
        except Exception as exc:
            result["failure_reason"] = "embedding_failed"
            result["detail"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
            return result
        return result
    except Exception as exc:
        result["failure_reason"] = "unknown_error"
        result["detail"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
        return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _load_resume_dataframe(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    dtype = {
        "subject_id": str,
        "reference_subject_id": str,
        "probe_subject_id": str,
        "pair_id": str,
        "quality": str,
        "config_fingerprint": str,
    }
    loaded = pd.read_csv(path, dtype=dtype)
    for column in columns:
        if column not in loaded.columns:
            loaded[column] = np.nan
    return loaded.loc[:, list(columns)]


def _checkpoint_results(
    rows: List[Dict[str, Any]],
    skipped_rows: List[Dict[str, Any]],
    partial_path: Path,
    skipped_path: Path,
) -> None:
    results = pd.DataFrame(rows, columns=FORMAL_RESULT_COLUMNS)
    skipped = pd.DataFrame(skipped_rows, columns=FORMAL_SKIPPED_COLUMNS)
    _atomic_csv(results, partial_path)
    _atomic_csv(skipped, skipped_path)


def run_formal_embedding_experiment(
    pairs: pd.DataFrame,
    conditions: Sequence[str],
    tar_path: Path,
    output_dir: Path,
    compression_mode: str,
    preprocess_order: str,
    config_fingerprint: str,
    checkpoint_every: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, int]:
    """Run or resume all pair-condition embeddings with progressive CSV checkpoints."""
    partial_path = output_dir / "results_partial.csv"
    final_path = output_dir / "results_final.csv"
    skipped_path = output_dir / "skipped_images.csv"
    resume_source = partial_path if partial_path.exists() else final_path
    existing = _load_resume_dataframe(resume_source, FORMAL_RESULT_COLUMNS)
    existing_skips = _load_resume_dataframe(skipped_path, FORMAL_SKIPPED_COLUMNS)
    if not existing.empty:
        fingerprints = set(existing["config_fingerprint"].dropna().astype(str))
        if fingerprints and fingerprints != {config_fingerprint}:
            raise RuntimeError("Partial results do not match the current experiment config")
    completed = set(
        zip(
            existing["split"].astype(str),
            existing["pair_id"].astype(str),
            existing["quality"].astype(str),
        )
    )
    expected = [
        (str(pair["split"]), str(pair["pair_id"]), str(condition))
        for pair in pairs.to_dict(orient="records")
        for condition in conditions
    ]
    pending_count = sum(key not in completed for key in expected)
    resumed_count = len(expected) - pending_count
    if resumed_count:
        print("  Resume: skipping {:,} completed pair-condition rows.".format(resumed_count))
    if pending_count == 0:
        final = existing.copy()
        _atomic_csv(final, final_path)
        _atomic_csv(final, partial_path)
        return final, existing_skips, "resume-complete", resumed_count

    try:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN
    except Exception as exc:
        raise RuntimeError("PyTorch/facenet-pytorch dependencies are unavailable") from exc

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("  Embedding device: {}".format(device))
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        keep_all=False,
        select_largest=True,
        post_process=True,
        device=device,
    )
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    result_rows = existing.to_dict(orient="records")
    skipped_rows = existing_skips.to_dict(orient="records")
    embedding_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    new_since_checkpoint = 0
    try:
        from tqdm import tqdm

        progress: Optional[Any] = tqdm(
            total=pending_count, desc="Formal embeddings", unit="comparison"
        )
    except Exception:
        progress = None

    try:
        with ArchiveImageStore(tar_path) as store:
            for pair in pairs.to_dict(orient="records"):
                for condition in conditions:
                    key = (str(pair["split"]), str(pair["pair_id"]), str(condition))
                    if key in completed:
                        continue
                    reference_condition = (
                        str(condition) if compression_mode == "both" else "Original"
                    )
                    reference_key = (
                        str(pair["reference_path"]),
                        reference_condition,
                        "reference",
                    )
                    probe_key = (str(pair["probe_path"]), str(condition), "probe")
                    if reference_key not in embedding_cache:
                        embedding_cache[reference_key] = _process_image_condition(
                            store=store,
                            image_path=reference_key[0],
                            condition=reference_key[1],
                            image_role="reference",
                            mtcnn=mtcnn,
                            model=model,
                            device=device,
                            preprocess_order=preprocess_order,
                        )
                    if probe_key not in embedding_cache:
                        embedding_cache[probe_key] = _process_image_condition(
                            store=store,
                            image_path=probe_key[0],
                            condition=probe_key[1],
                            image_role="probe",
                            mtcnn=mtcnn,
                            model=model,
                            device=device,
                            preprocess_order=preprocess_order,
                        )
                    reference = embedding_cache[reference_key]
                    probe = embedding_cache[probe_key]
                    similarity = np.nan
                    reasons: List[str] = []
                    if reference["failure_reason"]:
                        reasons.append(str(reference["failure_reason"]))
                        skipped_rows.append(
                            {
                                "config_fingerprint": config_fingerprint,
                                "split": pair["split"],
                                "pair_id": pair["pair_id"],
                                "quality": condition,
                                "image_path": pair["reference_path"],
                                "image_role": "reference",
                                "subject_id": pair["reference_subject_id"],
                                "category_code": pair["reference_category"],
                                "reason": reference["failure_reason"],
                                "detail": reference["detail"],
                            }
                        )
                    if probe["failure_reason"]:
                        reasons.append(str(probe["failure_reason"]))
                        skipped_rows.append(
                            {
                                "config_fingerprint": config_fingerprint,
                                "split": pair["split"],
                                "pair_id": pair["pair_id"],
                                "quality": condition,
                                "image_path": pair["probe_path"],
                                "image_role": "probe",
                                "subject_id": pair["probe_subject_id"],
                                "category_code": pair["probe_category"],
                                "reason": probe["failure_reason"],
                                "detail": probe["detail"],
                            }
                        )
                    if reference["embedding"] is not None and probe["embedding"] is not None:
                        similarity = _cosine(reference["embedding"], probe["embedding"])
                        if not np.isfinite(similarity):
                            reasons.append("cosine_nan")
                            skipped_rows.append(
                                {
                                    "config_fingerprint": config_fingerprint,
                                    "split": pair["split"],
                                    "pair_id": pair["pair_id"],
                                    "quality": condition,
                                    "image_path": "comparison",
                                    "image_role": "comparison",
                                    "subject_id": pair["probe_subject_id"],
                                    "category_code": pair["probe_category"],
                                    "reason": "cosine_nan",
                                    "detail": "Cosine similarity was non-finite",
                                }
                            )
                    unique_reasons = list(dict.fromkeys(reasons))
                    result_rows.append(
                        {
                            "config_fingerprint": config_fingerprint,
                            "split": pair["split"],
                            "pair_id": pair["pair_id"],
                            "quality": condition,
                            "label": int(pair["label"]),
                            "label_name": pair["label_name"],
                            "reference_subject_id": str(pair["reference_subject_id"]),
                            "probe_subject_id": str(pair["probe_subject_id"]),
                            "reference_path": pair["reference_path"],
                            "probe_path": pair["probe_path"],
                            "reference_category": pair["reference_category"],
                            "probe_category": pair["probe_category"],
                            "cosine_similarity": similarity,
                            "probe_size_bytes": probe["size_bytes"],
                            "reference_size_bytes": reference["size_bytes"],
                            "reference_detected": bool(reference["detected"]),
                            "probe_detected": bool(probe["detected"]),
                            "comparison_success": bool(np.isfinite(similarity)),
                            "failure_reason": ";".join(unique_reasons),
                        }
                    )
                    completed.add(key)
                    new_since_checkpoint += 1
                    if progress is not None:
                        progress.update(1)
                    if new_since_checkpoint >= max(1, int(checkpoint_every)):
                        _checkpoint_results(
                            result_rows,
                            skipped_rows,
                            partial_path,
                            skipped_path,
                        )
                        new_since_checkpoint = 0
    except KeyboardInterrupt:
        _checkpoint_results(result_rows, skipped_rows, partial_path, skipped_path)
        raise
    except Exception:
        _checkpoint_results(result_rows, skipped_rows, partial_path, skipped_path)
        raise
    finally:
        if progress is not None:
            progress.close()

    _checkpoint_results(result_rows, skipped_rows, partial_path, skipped_path)
    final = pd.DataFrame(result_rows, columns=FORMAL_RESULT_COLUMNS)
    order = _condition_order_map(conditions)
    final["_quality_order"] = final["quality"].astype(str).map(order).fillna(999)
    final = final.sort_values(
        ["split", "pair_id", "_quality_order"], kind="stable"
    ).drop(columns="_quality_order")
    skipped = pd.DataFrame(skipped_rows, columns=FORMAL_SKIPPED_COLUMNS)
    _atomic_csv(final, final_path)
    _atomic_csv(final, partial_path)
    _atomic_csv(skipped, skipped_path)
    return final, skipped, str(device), resumed_count


def _successful_scores(results: pd.DataFrame) -> pd.DataFrame:
    copy = results.copy()
    copy["cosine_similarity"] = pd.to_numeric(copy["cosine_similarity"], errors="coerce")
    return copy[copy["cosine_similarity"].notna()]


def _roc_points(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires both genuine and impostor scores")
    thresholds = np.concatenate(([np.inf], np.unique(scores)[::-1]))
    fpr: List[float] = []
    tpr: List[float] = []
    for threshold in thresholds:
        predicted = scores >= threshold
        fpr.append(float(predicted[labels == 0].mean()))
        tpr.append(float(predicted[labels == 1].mean()))
    return np.asarray(fpr), np.asarray(tpr), thresholds


def _eer_from_roc(
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
        left_threshold = thresholds[left]
        right_threshold = thresholds[right]
        if np.isfinite(left_threshold) and np.isfinite(right_threshold):
            threshold = float(
                left_threshold + weight * (right_threshold - left_threshold)
            )
        else:
            threshold = float(right_threshold)
        return eer, threshold
    index = int(np.argmin(np.abs(difference)))
    return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])


def select_threshold_from_dev(
    results: pd.DataFrame,
    baseline_condition: str,
) -> Dict[str, Any]:
    """Select one EER-style threshold from development baseline scores only."""
    successful = _successful_scores(results)
    development = successful[
        (successful["split"] == "development")
        & (successful["quality"].astype(str) == str(baseline_condition))
    ]
    genuine = development[development["label"].astype(int) == 1]
    impostor = development[development["label"].astype(int) == 0]
    if genuine.empty or impostor.empty:
        raise RuntimeError("Development baseline lacks both genuine and impostor scores")
    labels = development["label"].astype(int).to_numpy()
    scores = development["cosine_similarity"].astype(float).to_numpy()
    fpr, tpr, thresholds = _roc_points(labels, scores)
    eer, threshold = _eer_from_roc(fpr, tpr, thresholds)
    return {
        "threshold": float(threshold),
        "threshold_source_condition": str(baseline_condition),
        "num_development_genuine_pairs": int(len(genuine)),
        "num_development_impostor_pairs": int(len(impostor)),
        "method": "development_baseline_empirical_EER_intersection",
        "development_eer": float(eer),
    }


def compute_roc_det_eer(
    eval_results: pd.DataFrame,
    conditions: Sequence[str],
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Compute evaluation ROC/DET points and EER for every quality condition."""
    successful = _successful_scores(eval_results)
    rows: List[Dict[str, Any]] = []
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for condition in conditions:
        group = successful[successful["quality"].astype(str) == str(condition)]
        if set(group["label"].astype(int).unique()) != {0, 1}:
            rows.append(
                {
                    "quality": condition,
                    "eer": np.nan,
                    "threshold_at_eer": np.nan,
                    "num_genuine": int((group["label"] == 1).sum()),
                    "num_impostor": int((group["label"] == 0).sum()),
                }
            )
            continue
        fpr, tpr, thresholds = _roc_points(
            group["label"].astype(int).to_numpy(),
            group["cosine_similarity"].astype(float).to_numpy(),
        )
        eer, eer_threshold = _eer_from_roc(fpr, tpr, thresholds)
        curves[str(condition)] = (fpr, tpr, thresholds)
        rows.append(
            {
                "quality": condition,
                "eer": eer,
                "threshold_at_eer": eer_threshold,
                "num_genuine": int((group["label"] == 1).sum()),
                "num_impostor": int((group["label"] == 0).sum()),
            }
        )
    eer_table = pd.DataFrame(rows)
    eer_table.to_csv(output_dir / "eer_by_quality.csv", index=False)
    return eer_table, curves


def compute_tar_at_fmr(
    curves: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    eval_results: pd.DataFrame,
    conditions: Sequence[str],
    output_path: Path,
    operating_points: Sequence[float] = (0.01, 0.001),
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for condition in conditions:
        group = _successful_scores(
            eval_results[eval_results["quality"].astype(str) == str(condition)]
        )
        num_impostor = int((group["label"].astype(int) == 0).sum())
        resolution = 1.0 / num_impostor if num_impostor else np.nan
        curve = curves.get(str(condition))
        for target in operating_points:
            if curve is None:
                tar = achieved = threshold = np.nan
            else:
                fpr, tpr, thresholds = curve
                candidates = np.where(fpr <= float(target) + 1e-12)[0]
                if len(candidates):
                    best_tpr = np.max(tpr[candidates])
                    best_candidates = candidates[tpr[candidates] == best_tpr]
                    index = int(best_candidates[-1])
                    tar = float(tpr[index])
                    achieved = float(fpr[index])
                    threshold = float(thresholds[index])
                else:
                    tar = achieved = threshold = np.nan
            rows.append(
                {
                    "quality": condition,
                    "target_fmr": float(target),
                    "TAR": tar,
                    "achieved_fmr": achieved,
                    "threshold": threshold,
                    "num_eval_impostor": num_impostor,
                    "empirical_fmr_resolution": resolution,
                    "target_empirically_resolved": bool(
                        num_impostor and float(target) >= resolution
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return table


def evaluate_on_eval(
    results: pd.DataFrame,
    threshold: float,
    conditions: Sequence[str],
    eer_table: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate all final performance/storage metrics on evaluation rows only."""
    evaluation = results[results["split"] == "evaluation"].copy()
    successful = _successful_scores(evaluation)
    eer_lookup = {
        str(row["quality"]): float(row["eer"])
        for row in eer_table.to_dict(orient="records")
    }
    original_sizes = evaluation[
        evaluation["quality"].astype(str) == "Original"
    ]["probe_size_bytes"]
    original_average = float(pd.to_numeric(original_sizes, errors="coerce").mean())
    rows: List[Dict[str, Any]] = []
    for condition in conditions:
        attempted = evaluation[evaluation["quality"].astype(str) == str(condition)].copy()
        valid = successful[successful["quality"].astype(str) == str(condition)].copy()
        genuine = valid[valid["label"].astype(int) == 1]
        impostor = valid[valid["label"].astype(int) == 0]
        fmr = (
            float((impostor["cosine_similarity"].astype(float) >= threshold).mean())
            if not impostor.empty
            else np.nan
        )
        fnmr = (
            float((genuine["cosine_similarity"].astype(float) < threshold).mean())
            if not genuine.empty
            else np.nan
        )
        reference_detected = attempted["reference_detected"].map(_as_bool)
        probe_detected = attempted["probe_detected"].map(_as_bool)
        both_detected = reference_detected & probe_detected
        average_size = float(
            pd.to_numeric(attempted["probe_size_bytes"], errors="coerce").mean()
        )
        saving = (
            float(1.0 - average_size / original_average)
            if np.isfinite(original_average) and original_average > 0 and np.isfinite(average_size)
            else np.nan
        )
        rows.append(
            {
                "quality": condition,
                "num_eval_genuine": int(len(genuine)),
                "num_eval_impostor": int(len(impostor)),
                "threshold": float(threshold),
                "FMR": fmr,
                "FNMR": fnmr,
                "TAR": float(1.0 - fnmr) if np.isfinite(fnmr) else np.nan,
                "TNR": float(1.0 - fmr) if np.isfinite(fmr) else np.nan,
                "EER": eer_lookup.get(str(condition), np.nan),
                "mean_similarity_genuine": (
                    float(genuine["cosine_similarity"].mean()) if not genuine.empty else np.nan
                ),
                "mean_similarity_impostor": (
                    float(impostor["cosine_similarity"].mean()) if not impostor.empty else np.nan
                ),
                "median_similarity_genuine": (
                    float(genuine["cosine_similarity"].median()) if not genuine.empty else np.nan
                ),
                "median_similarity_impostor": (
                    float(impostor["cosine_similarity"].median()) if not impostor.empty else np.nan
                ),
                "average_probe_size_bytes": average_size,
                "storage_saving_vs_original": saving,
                "mtcnn_failure_rate": (
                    float(1.0 - both_detected.mean()) if len(attempted) else np.nan
                ),
                "reference_detection_failure_rate": (
                    float(1.0 - reference_detected.mean()) if len(attempted) else np.nan
                ),
                "probe_detection_failure_rate": (
                    float(1.0 - probe_detected.mean()) if len(attempted) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=FINAL_METRIC_COLUMNS)


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_similarity_distributions(
    eval_results: pd.DataFrame,
    conditions: Sequence[str],
    output_dir: Path,
) -> None:
    """Create boxplots, faceted histograms, and requested Q95/Q30 distributions."""
    plt = _matplotlib()
    valid = _successful_scores(eval_results)
    colors = {"genuine": "#2478B5", "impostor": "#E87520"}

    figure, axis = plt.subplots(figsize=(max(8, len(conditions) * 1.35), 5.5))
    data: List[np.ndarray] = []
    positions: List[float] = []
    box_colors: List[str] = []
    for index, condition in enumerate(conditions):
        for offset, label_name in ((-0.18, "genuine"), (0.18, "impostor")):
            values = valid[
                (valid["quality"].astype(str) == str(condition))
                & (valid["label_name"] == label_name)
            ]["cosine_similarity"].astype(float).to_numpy()
            data.append(values if len(values) else np.asarray([np.nan]))
            positions.append(index + offset)
            box_colors.append(colors[label_name])
    boxes = axis.boxplot(
        data,
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
    axis.set_xticks(range(len(conditions)))
    axis.set_xticklabels(conditions)
    axis.set_xlabel("Condition")
    axis.set_ylabel("Cosine similarity")
    axis.set_title("Evaluation score distributions by condition")
    axis.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch

    axis.legend(
        handles=[Patch(color=colors[name], label=name) for name in ("genuine", "impostor")]
    )
    figure.tight_layout()
    figure.savefig(output_dir / "similarity_boxplot_by_quality.png", dpi=160)
    plt.close(figure)

    columns = 2
    rows = int(math.ceil(len(conditions) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11, 3.5 * rows), squeeze=False)
    score_min = float(valid["cosine_similarity"].min()) if not valid.empty else -1.0
    score_max = float(valid["cosine_similarity"].max()) if not valid.empty else 1.0
    bins = np.linspace(score_min, score_max, 31)
    for index, condition in enumerate(conditions):
        axis = axes.flat[index]
        group = valid[valid["quality"].astype(str) == str(condition)]
        for label_name in ("genuine", "impostor"):
            values = group[group["label_name"] == label_name]["cosine_similarity"]
            axis.hist(
                values,
                bins=bins,
                alpha=0.55,
                density=True,
                label=label_name,
                color=colors[label_name],
            )
        axis.set_title(str(condition))
        axis.set_xlabel("Cosine similarity")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.20)
        axis.legend()
    for index in range(len(conditions), rows * columns):
        axes.flat[index].axis("off")
    figure.suptitle("Evaluation genuine/impostor distributions", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output_dir / "similarity_histogram_by_quality.png", dpi=160)
    plt.close(figure)

    for condition, filename in (
        ("95", "genuine_impostor_distribution_q95.png"),
        ("30", "genuine_impostor_distribution_q30.png"),
    ):
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        group = valid[valid["quality"].astype(str) == condition]
        if group.empty:
            axis.text(0.5, 0.5, "Condition {} was not run".format(condition), ha="center")
        else:
            for label_name in ("genuine", "impostor"):
                axis.hist(
                    group[group["label_name"] == label_name]["cosine_similarity"],
                    bins=bins,
                    alpha=0.55,
                    density=True,
                    label=label_name,
                    color=colors[label_name],
                )
            axis.legend()
        axis.set_title("Evaluation distribution at JPEG quality {}".format(condition))
        axis.set_xlabel("Cosine similarity")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.20)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)


def plot_roc_det(
    curves: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    conditions: Sequence[str],
    output_dir: Path,
) -> None:
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for condition in conditions:
        curve = curves.get(str(condition))
        if curve is None:
            continue
        fpr, tpr, _thresholds = curve
        axis.plot(np.maximum(fpr, 1e-5), tpr, label=str(condition))
    axis.set_xscale("log")
    axis.set_xlim(1e-4, 1.0)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("False Match Rate")
    axis.set_ylabel("True Accept Rate")
    axis.set_title("Evaluation ROC by condition")
    axis.grid(alpha=0.25, which="both")
    axis.legend(title="Condition")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_by_quality.png", dpi=160)
    plt.close(figure)

    normal = NormalDist()
    probability_ticks = np.asarray([0.001, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95])
    normal_ticks = np.asarray([normal.inv_cdf(float(value)) for value in probability_ticks])
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for condition in conditions:
        curve = curves.get(str(condition))
        if curve is None:
            continue
        fpr, tpr, _thresholds = curve
        fnr = 1.0 - tpr
        clipped_fpr = np.clip(fpr, 1e-5, 1.0 - 1e-5)
        clipped_fnr = np.clip(fnr, 1e-5, 1.0 - 1e-5)
        axis.plot(
            [normal.inv_cdf(float(value)) for value in clipped_fpr],
            [normal.inv_cdf(float(value)) for value in clipped_fnr],
            label=str(condition),
        )
    axis.set_xticks(normal_ticks)
    axis.set_xticklabels(["{:.1%}".format(value) for value in probability_ticks], rotation=35)
    axis.set_yticks(normal_ticks)
    axis.set_yticklabels(["{:.1%}".format(value) for value in probability_ticks])
    axis.set_xlabel("False Match Rate (normal deviate scale)")
    axis.set_ylabel("False Non-Match Rate (normal deviate scale)")
    axis.set_title("Evaluation DET by condition")
    axis.grid(alpha=0.25)
    axis.legend(title="Condition")
    figure.tight_layout()
    figure.savefig(output_dir / "det_by_quality.png", dpi=160)
    plt.close(figure)


def write_failure_summaries(
    skipped: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if skipped.empty:
        by_quality = pd.DataFrame(columns=["quality", "reason", "failure_records", "unique_images"])
        by_category = pd.DataFrame(
            columns=["category_code", "reason", "failure_records", "unique_images"]
        )
    else:
        by_quality = (
            skipped.groupby(["quality", "reason"], dropna=False)
            .agg(failure_records=("pair_id", "size"), unique_images=("image_path", "nunique"))
            .reset_index()
        )
        by_category = (
            skipped.groupby(["category_code", "reason"], dropna=False)
            .agg(failure_records=("pair_id", "size"), unique_images=("image_path", "nunique"))
            .reset_index()
        )
    by_quality.to_csv(output_dir / "failure_summary_by_quality.csv", index=False)
    by_category.to_csv(output_dir / "failure_summary_by_category.csv", index=False)
    return by_quality, by_category


def bootstrap_confidence_intervals(
    eval_results: pd.DataFrame,
    threshold: float,
    conditions: Sequence[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Subject-cluster bootstrap evaluation metrics with percentile 95% intervals."""
    if iterations <= 0:
        return pd.DataFrame(columns=["metric", "quality", "value", "ci_low", "ci_high"])
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []
    for condition in conditions:
        group = eval_results[
            eval_results["quality"].astype(str) == str(condition)
        ].copy()
        group["cosine_similarity"] = pd.to_numeric(
            group["cosine_similarity"], errors="coerce"
        )
        subjects = sorted(group["probe_subject_id"].astype(str).unique().tolist())
        if not subjects:
            continue

        def calculate(sample: pd.DataFrame) -> Dict[str, float]:
            valid = sample[sample["cosine_similarity"].notna()]
            genuine = valid[valid["label"].astype(int) == 1]
            impostor = valid[valid["label"].astype(int) == 0]
            fmr = (
                float((impostor["cosine_similarity"] >= threshold).mean())
                if not impostor.empty
                else np.nan
            )
            fnmr = (
                float((genuine["cosine_similarity"] < threshold).mean())
                if not genuine.empty
                else np.nan
            )
            return {
                "FMR": fmr,
                "FNMR": fnmr,
                "TAR": float(1.0 - fnmr) if np.isfinite(fnmr) else np.nan,
                "average_probe_size_bytes": float(
                    pd.to_numeric(sample["probe_size_bytes"], errors="coerce").mean()
                ),
            }

        point = calculate(group)
        samples: Dict[str, List[float]] = {metric: [] for metric in point}
        subject_groups = {
            subject: group[group["probe_subject_id"].astype(str) == subject]
            for subject in subjects
        }
        for _iteration in range(int(iterations)):
            selected = rng.choice(subjects, size=len(subjects), replace=True)
            bootstrap_sample = pd.concat(
                [subject_groups[str(subject)] for subject in selected], ignore_index=True
            )
            values = calculate(bootstrap_sample)
            for metric, value in values.items():
                if np.isfinite(value):
                    samples[metric].append(float(value))
        for metric, value in point.items():
            distribution = np.asarray(samples[metric], dtype=float)
            rows.append(
                {
                    "metric": metric,
                    "quality": condition,
                    "value": value,
                    "ci_low": (
                        float(np.percentile(distribution, 2.5)) if len(distribution) else np.nan
                    ),
                    "ci_high": (
                        float(np.percentile(distribution, 97.5)) if len(distribution) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows, columns=["metric", "quality", "value", "ci_low", "ci_high"])


def _format_metric(value: Any, percent: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return "{:.2%}".format(numeric) if percent else "{:.4f}".format(numeric)


def write_final_report(
    output_path: Path,
    inventory: pd.DataFrame,
    assignment: pd.DataFrame,
    dev_pairs: pd.DataFrame,
    eval_pairs: pd.DataFrame,
    conditions: Sequence[str],
    threshold_info: Optional[Mapping[str, Any]],
    final_metrics: Optional[pd.DataFrame],
    results: Optional[pd.DataFrame],
    skipped: Optional[pd.DataFrame],
    run_mode: str,
    device: Optional[str],
    compression_mode: str,
    preprocess_order: str,
    requested_impostors_per_probe: int,
    effective_impostors_per_probe: Mapping[str, int],
    error: Optional[str] = None,
) -> Path:
    primary_count = (
        int(inventory["is_primary_dataset_image"].map(_as_bool).sum())
        if "is_primary_dataset_image" in inventory.columns and not inventory.empty
        else len(inventory)
    )
    development_subjects = int((assignment["split"] == "development").sum())
    evaluation_subjects = int((assignment["split"] == "evaluation").sum())
    dev_genuine = int((dev_pairs["label"] == 1).sum())
    dev_impostor = int((dev_pairs["label"] == 0).sum())
    eval_genuine = int((eval_pairs["label"] == 1).sum())
    eval_impostor = int((eval_pairs["label"] == 0).sum())
    attempted = len(results) if results is not None else 0
    successful = (
        int(pd.to_numeric(results["cosine_similarity"], errors="coerce").notna().sum())
        if results is not None and not results.empty
        else 0
    )
    failure_count = len(skipped) if skipped is not None else 0
    mtcnn_failure = np.nan
    if results is not None and not results.empty:
        detected = results["reference_detected"].map(_as_bool) & results[
            "probe_detected"
        ].map(_as_bool)
        mtcnn_failure = float(1.0 - detected.mean())
    succeeded = bool(
        error is None
        and threshold_info is not None
        and final_metrics is not None
        and not final_metrics.empty
        and successful > 0
    )
    lines = [
        "# Formal Color FERET JPEG experiment report",
        "",
        "This report contains metadata and aggregate results only. No Color FERET face image was saved, displayed, or exported.",
        "",
        "## Outcome",
        "",
        "- Formal experiment completed successfully: **{}**".format("yes" if succeeded else "no"),
        "- Run mode: {}".format(run_mode),
        "- Embedding device: {}".format(device or "not run"),
        "- Compression mode: {}".format(compression_mode),
        "- Preprocessing order: {}".format(preprocess_order),
        "",
        "## Dataset and frozen protocol",
        "",
        "- Image-like tar members indexed: {:,}".format(len(inventory)),
        "- Primary Color FERET images: {:,}".format(primary_count),
        "- Eligible/selected identities: {:,}".format(len(assignment)),
        "- Development identities: {:,}".format(development_subjects),
        "- Evaluation identities: {:,}".format(evaluation_subjects),
        "- Requested impostors per genuine probe: {}".format(
            int(requested_impostors_per_probe)
        ),
        "- Effective impostors per genuine probe: development={}, evaluation={}".format(
            int(effective_impostors_per_probe["development"]),
            int(effective_impostors_per_probe["evaluation"]),
        ),
        "- Development pairs: {:,} genuine, {:,} impostor".format(
            dev_genuine, dev_impostor
        ),
        "- Development realised impostor-to-genuine ratio: {:.4f}".format(
            float(dev_impostor / dev_genuine) if dev_genuine else np.nan
        ),
        "- Evaluation pairs: {:,} genuine, {:,} impostor".format(
            eval_genuine, eval_impostor
        ),
        "- Evaluation realised impostor-to-genuine ratio: {:.4f}".format(
            float(eval_impostor / eval_genuine) if eval_genuine else np.nan
        ),
        "- Impostor sampling rule: {}".format(IMPOSTOR_SAMPLING_RULE),
        "- Conditions: {}".format(", ".join(conditions)),
        "",
        "## Threshold and embedding execution",
        "",
    ]
    if threshold_info is not None:
        lines.extend(
            [
                "- Fixed threshold: {:.6f}".format(float(threshold_info["threshold"])),
                "- Source condition: {} development scores".format(
                    threshold_info["threshold_source_condition"]
                ),
                "- Selection method: {}".format(threshold_info["method"]),
            ]
        )
    else:
        lines.append("- Threshold was not selected.")
    lines.extend(
        [
            "- Pair-condition rows attempted: {:,}".format(attempted),
            "- Rows with valid cosine similarity: {:,}".format(successful),
            "- MTCNN comparison failure rate: {}".format(
                _format_metric(mtcnn_failure, percent=True)
            ),
            "- Detailed failure records: {:,}".format(failure_count),
        ]
    )
    if error:
        lines.append("- Error: `{}`".format(error.replace("`", "'")[:1000]))

    if final_metrics is not None and not final_metrics.empty:
        lines.extend(
            [
                "",
                "## Evaluation-only main metrics",
                "",
                "| Condition | FMR | FNMR | TAR | EER | Mean genuine | Mean impostor | Storage saving |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in final_metrics.to_dict(orient="records"):
            lines.append(
                "| {quality} | {fmr} | {fnmr} | {tar} | {eer} | {genuine} | {impostor} | {saving} |".format(
                    quality=row["quality"],
                    fmr=_format_metric(row["FMR"], percent=True),
                    fnmr=_format_metric(row["FNMR"], percent=True),
                    tar=_format_metric(row["TAR"], percent=True),
                    eer=_format_metric(row["EER"], percent=True),
                    genuine=_format_metric(row["mean_similarity_genuine"]),
                    impostor=_format_metric(row["mean_similarity_impostor"]),
                    saving=_format_metric(row["storage_saving_vs_original"], percent=True),
                )
            )
    lines.extend(
        [
            "",
            "## Caveats and next steps",
            "",
            "The fixed threshold is selected only from development identities at the baseline condition and is then applied unchanged to evaluation identities. EER and TAR-at-FMR are empirical evaluation summaries; very small debug runs cannot resolve low FMR targets reliably. Bootstrap intervals, when requested, resample evaluation probe subjects as clusters. Freeze this configuration and pair list before using the run as dissertation evidence.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a formal, local-only Color FERET JPEG verification experiment."
    )
    parser.add_argument("--tar-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("formal_run"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full-experiment", action="store_true")
    mode.add_argument("--max-identities", type=int)
    parser.add_argument(
        "--qualities",
        nargs="+",
        default=["Original", "95", "90", "80", "70", "50", "30"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.20)
    parser.add_argument(
        "--impostors-per-probe",
        type=_positive_int,
        default=1,
        help="Requested unique different-subject references per genuine probe (default: 1)",
    )
    parser.add_argument("--run-embeddings", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument(
        "--compression-mode",
        choices=["probe-only", "both"],
        default="probe-only",
    )
    parser.add_argument(
        "--preprocess-order",
        choices=["compress-before-detect", "detect-before-compress"],
        default="compress-before-detect",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        conditions = normalize_quality_conditions(args.qualities)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
        return 2
    if args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if sys.version_info < (3, 10):
        print(
            "Warning: Python {}.{} is outside the supported formal environment; use Python 3.10/3.11.".format(
                sys.version_info.major, sys.version_info.minor
            ),
            file=sys.stderr,
        )

    tar_path = args.tar_path.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    output_dir = work_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Formal Color FERET JPEG experiment")
    print("Outputs: {}".format(output_dir))
    print("Images will be streamed from the tar and never saved to outputs.")

    inventory = inspect_tar(tar_path, output_dir)
    partitions = read_feret_partitions(tar_path)
    manifest = build_manifest(inventory, output_dir, partitions)
    eligible_subjects = select_eligible_subjects(manifest, output_dir)

    assignment, all_eligible_ids = build_reference_probe_assignment(
        eligible_subjects=eligible_subjects,
        seed=args.seed,
        max_identities=args.max_identities,
        full_experiment=bool(args.full_experiment),
    )
    assignment = split_subjects_dev_eval(
        assignment, seed=args.seed + 1009, dev_ratio=args.dev_ratio
    )
    assignment.to_csv(output_dir / "reference_probe_assignment.csv", index=False)
    dev_subjects = assignment[assignment["split"] == "development"][
        ["subject_id", "num_probe_images"]
    ]
    eval_subjects = assignment[assignment["split"] == "evaluation"][
        ["subject_id", "num_probe_images"]
    ]
    dev_subjects.to_csv(output_dir / "dev_subjects.csv", index=False)
    eval_subjects.to_csv(output_dir / "eval_subjects.csv", index=False)

    path_categories = _path_category_lookup(manifest)
    dev_pairs, dev_effective_k = build_impostor_sampled_pairs(
        assignment,
        "development",
        args.seed + 2003,
        path_categories,
        args.impostors_per_probe,
    )
    eval_pairs, eval_effective_k = build_impostor_sampled_pairs(
        assignment,
        "evaluation",
        args.seed + 3001,
        path_categories,
        args.impostors_per_probe,
    )
    dev_pairs.to_csv(output_dir / "dev_pairs.csv", index=False)
    eval_pairs.to_csv(output_dir / "eval_pairs.csv", index=False)
    pairs = pd.concat([dev_pairs, eval_pairs], ignore_index=True)
    write_pair_generation_summary(
        assignment,
        dev_pairs,
        eval_pairs,
        args.seed,
        args.impostors_per_probe,
        {
            "development": dev_effective_k,
            "evaluation": eval_effective_k,
        },
        output_dir / "pair_generation_summary.csv",
    )
    excluded = build_excluded_images(manifest, assignment, all_eligible_ids)
    excluded.to_csv(output_dir / "excluded_images.csv", index=False)

    baseline_condition = "Original" if "Original" in conditions else max(
        conditions, key=lambda value: int(value)
    )
    tar_stat = tar_path.stat()
    run_mode = "full-experiment" if args.full_experiment else "debug-max-{}".format(
        50 if args.max_identities is None else args.max_identities
    )
    stable_config: Dict[str, Any] = {
        "tar_filename": tar_path.name,
        "tar_size_bytes": int(tar_stat.st_size),
        "tar_mtime_ns": int(tar_stat.st_mtime_ns),
        "run_mode": run_mode,
        "random_seed": int(args.seed),
        "selected_quality_conditions": list(conditions),
        "frontal_codes": sorted(FRONTAL_CODES),
        "non_frontal_codes": sorted(NON_FRONTAL_CODES),
        "reference_selection_rule": "one fa if available, otherwise first conservative frontal candidate",
        "probe_selection_rule": "all remaining conservative frontal candidates",
        "requested_impostors_per_probe": int(args.impostors_per_probe),
        "effective_impostors_per_probe": {
            "development": int(dev_effective_k),
            "evaluation": int(eval_effective_k),
        },
        "impostor_sampling_rule": IMPOSTOR_SAMPLING_RULE,
        "development_evaluation_split_ratio": {
            "development": float(args.dev_ratio),
            "evaluation": float(1.0 - args.dev_ratio),
            "subject_disjoint": True,
        },
        "model_name": "InceptionResnetV1",
        "mtcnn_settings": {
            "image_size": 160,
            "margin": 0,
            "keep_all": False,
            "select_largest": True,
            "post_process": True,
        },
        "facenet_checkpoint": "vggface2",
        "threshold_strategy": {
            "source_split": "development",
            "source_condition": baseline_condition,
            "method": "empirical EER intersection",
            "applied_to": "all evaluation conditions",
        },
        "compression_mode": args.compression_mode,
        "preprocessing_order": args.preprocess_order,
        "jpeg_settings": {
            "subsampling": "4:2:0",
            "optimize": False,
            "progressive": False,
            "color_mode": "RGB",
        },
        "assignment_digest_sha256": _dataframe_digest(
            assignment,
            ["subject_id", "reference_path", "probe_paths", "split"],
        ),
        "dev_pair_digest_sha256": _dataframe_digest(dev_pairs, FORMAL_PAIR_COLUMNS),
        "eval_pair_digest_sha256": _dataframe_digest(eval_pairs, FORMAL_PAIR_COLUMNS),
        "bootstrap_iterations": int(args.bootstrap),
    }
    config = write_experiment_config(
        output_dir / "experiment_config.json",
        stable_config,
        output_dir / "results_partial.csv",
    )

    results: Optional[pd.DataFrame] = None
    skipped: Optional[pd.DataFrame] = None
    threshold_info: Optional[Dict[str, Any]] = None
    final_metrics: Optional[pd.DataFrame] = None
    device: Optional[str] = None
    error: Optional[str] = None
    exit_code = 0

    if args.run_embeddings:
        try:
            results, skipped, device, resumed_count = run_formal_embedding_experiment(
                pairs=pairs,
                conditions=conditions,
                tar_path=tar_path,
                output_dir=output_dir,
                compression_mode=args.compression_mode,
                preprocess_order=args.preprocess_order,
                config_fingerprint=str(config["config_fingerprint"]),
                checkpoint_every=args.checkpoint_every,
            )
            print("  Completed rows: {:,}; resumed rows: {:,}.".format(len(results), resumed_count))
            threshold_info = select_threshold_from_dev(results, baseline_condition)
            threshold_info["config_fingerprint"] = config["config_fingerprint"]
            _atomic_json(threshold_info, output_dir / "thresholds.json")
            evaluation_results = results[results["split"] == "evaluation"].copy()
            eer_table, curves = compute_roc_det_eer(
                evaluation_results, conditions, output_dir
            )
            compute_tar_at_fmr(
                curves,
                evaluation_results,
                conditions,
                output_dir / "tar_at_fmr.csv",
            )
            final_metrics = evaluate_on_eval(
                results,
                float(threshold_info["threshold"]),
                conditions,
                eer_table,
            )
            final_metrics.to_csv(output_dir / "final_metrics_by_quality.csv", index=False)
            plot_similarity_distributions(evaluation_results, conditions, output_dir)
            plot_roc_det(curves, conditions, output_dir)
            write_failure_summaries(skipped, output_dir)
            if args.bootstrap > 0:
                intervals = bootstrap_confidence_intervals(
                    evaluation_results,
                    float(threshold_info["threshold"]),
                    conditions,
                    args.bootstrap,
                    args.seed + 4001,
                )
                intervals.to_csv(output_dir / "final_metrics_with_ci.csv", index=False)
        except Exception as exc:
            error = "{}: {}".format(type(exc).__name__, str(exc))
            print("Formal embedding experiment failed: {}".format(error), file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            exit_code = 2
            if (output_dir / "results_partial.csv").exists():
                results = _load_resume_dataframe(
                    output_dir / "results_partial.csv", FORMAL_RESULT_COLUMNS
                )
            if (output_dir / "skipped_images.csv").exists():
                skipped = _load_resume_dataframe(
                    output_dir / "skipped_images.csv", FORMAL_SKIPPED_COLUMNS
                )
    else:
        print("Embedding execution was not requested; protocol/config outputs are complete.")

    report = write_final_report(
        output_path=output_dir / "formal_experiment_report.md",
        inventory=inventory,
        assignment=assignment,
        dev_pairs=dev_pairs,
        eval_pairs=eval_pairs,
        conditions=conditions,
        threshold_info=threshold_info,
        final_metrics=final_metrics,
        results=results,
        skipped=skipped,
        run_mode=run_mode,
        device=device,
        compression_mode=args.compression_mode,
        preprocess_order=args.preprocess_order,
        requested_impostors_per_probe=args.impostors_per_probe,
        effective_impostors_per_probe={
            "development": dev_effective_k,
            "evaluation": eval_effective_k,
        },
        error=error,
    )
    print("Formal experiment report: {}".format(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
