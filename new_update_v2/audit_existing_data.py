#!/usr/bin/env python3
"""Read-only audit of assets relevant to the hardened-impostor experiment.

All artifacts written by this script stay below new_update_v2/outputs/audit.
The audit deliberately ignores new_update_v2 itself when looking for pre-existing
embedding/crop caches, so rerunning it cannot turn the newly produced cache into
evidence that a cache existed before this update.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


EMBEDDING_EXTENSIONS = {
    ".npy", ".npz", ".pt", ".pth", ".pkl", ".pickle", ".joblib",
    ".parquet", ".feather", ".h5", ".hdf5", ".bin", ".mat", ".lmdb",
}
NAME_HINTS = ("embed", "embedding", "feature_vector", "face_crop", "aligned_face")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        try:
            if path.is_file():
                yield path
        except OSError:
            continue


def looks_like_embedding_or_crop(path: Path) -> bool:
    lower = path.name.lower()
    return path.suffix.lower() in EMBEDDING_EXTENSIONS or any(hint in lower for hint in NAME_HINTS)


def load_assignment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    required = {"subject_id", "reference_path", "probe_paths", "num_probe_images", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Assignment is missing columns: {missing}: {path}")
    return frame


def mapping_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    columns = ["subject_id", "reference_path", "probe_paths", "num_probe_images"]
    records = frame.loc[:, columns].sort_values("subject_id", kind="stable").to_dict("records")
    for record in records:
        record["subject_id"] = str(record["subject_id"])
        record["num_probe_images"] = int(record["num_probe_images"])
    return records


def canonical_digest(records: List[Dict[str, Any]]) -> str:
    raw = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def probe_count(frame: pd.DataFrame, split: str) -> int:
    values = pd.to_numeric(frame.loc[frame["split"] == split, "num_probe_images"], errors="raise")
    return int(values.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    root = analysis_dir.parent
    output_dir = analysis_dir / "outputs" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = analysis_dir / "analysis_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    candidates: List[Dict[str, Any]] = []
    for path in safe_files(root):
        try:
            path.relative_to(analysis_dir)
            continue
        except ValueError:
            pass
        if looks_like_embedding_or_crop(path):
            stat = path.stat()
            candidates.append({
                "relative_path": path.relative_to(root).as_posix(),
                "extension": path.suffix.lower(),
                "size_bytes": int(stat.st_size),
                "name_match": any(hint in path.name.lower() for hint in NAME_HINTS),
            })
    candidates.sort(key=lambda row: row["relative_path"])
    with (output_dir / "preexisting_embedding_crop_candidates.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "extension", "size_bytes", "name_match"]
        )
        writer.writeheader()
        writer.writerows(candidates)

    seed_rows: List[Dict[str, Any]] = []
    baseline_mapping = None
    baseline_digest = None
    all_mapping_equal = True
    for seed_text, seed_config in config["seeds"].items():
        assignment_path = (analysis_dir / seed_config["assignment_relative_to_analysis_dir"]).resolve()
        threshold_path = (analysis_dir / seed_config["threshold_relative_to_analysis_dir"]).resolve()
        assignment = load_assignment(assignment_path)
        mapping = mapping_records(assignment)
        digest = canonical_digest(mapping)
        if baseline_mapping is None:
            baseline_mapping = mapping
            baseline_digest = digest
        else:
            all_mapping_equal = all_mapping_equal and mapping == baseline_mapping
        dev_probes = probe_count(assignment, "development")
        eval_probes = probe_count(assignment, "evaluation")
        dev_refs = int((assignment["split"] == "development").sum())
        eval_refs = int((assignment["split"] == "evaluation").sum())
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        seed_rows.append({
            "seed": int(seed_text),
            "assignment_path": assignment_path.relative_to(root).as_posix(),
            "assignment_sha256": sha256_file(assignment_path),
            "identity_mapping_sha256": digest,
            "subjects": int(len(assignment)),
            "development_references": dev_refs,
            "development_probes": dev_probes,
            "development_all_nonmate_pairs": int(dev_probes * (dev_refs - 1)),
            "evaluation_references": eval_refs,
            "evaluation_probes": eval_probes,
            "evaluation_all_nonmate_pairs": int(eval_probes * (eval_refs - 1)),
            "old_sampled_development_impostors": int(threshold["num_development_impostor_pairs"]),
            "old_eer_threshold": float(threshold["threshold"]),
        })
    pd.DataFrame(seed_rows).sort_values("seed").to_csv(
        output_dir / "seed_protocol_inventory.csv", index=False
    )

    tar_path = (analysis_dir / config["raw_tar_relative_to_analysis_dir"]).resolve()
    score_files = sorted({
        (analysis_dir / relative).resolve()
        for relative in config["legacy_score_sources"].values()
    })
    source_files = [config_path, tar_path]
    for seed_config in config["seeds"].values():
        source_files.append((analysis_dir / seed_config["assignment_relative_to_analysis_dir"]).resolve())
        source_files.append((analysis_dir / seed_config["threshold_relative_to_analysis_dir"]).resolve())
    source_files.extend(score_files)
    manifest = []
    for path in sorted(set(source_files)):
        manifest.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    write_json(output_dir / "source_manifest_sha256.json", manifest)

    result = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(root),
        "preexisting_embedding_or_crop_candidate_count": len(candidates),
        "preexisting_embedding_or_crop_candidates": candidates,
        "embedding_cache_found": any(row["extension"] in EMBEDDING_EXTENSIONS for row in candidates),
        "saved_face_crop_found": any("crop" in row["relative_path"].lower() for row in candidates),
        "legacy_pairwise_score_files": [path.relative_to(root).as_posix() for path in score_files],
        "raw_tar": {
            "relative_path": tar_path.relative_to(root).as_posix(),
            "size_bytes": int(tar_path.stat().st_size),
            "sha256": sha256_file(tar_path),
        },
        "seed_count": len(seed_rows),
        "identity_reference_probe_mapping_equal_across_seeds": bool(all_mapping_equal),
        "canonical_identity_mapping_sha256": baseline_digest,
        "seed_protocols": seed_rows,
        "conclusion": (
            "No reusable pre-existing embedding/crop cache was found; full and hard-impostor "
            "scores cannot be recovered from the sampled pairwise score CSVs. The raw tar, five "
            "identity-disjoint splits, common reference/probe mapping, old thresholds, and legacy "
            "scores are available, so one embedding pass can be cached and reused across all seeds."
        ),
    }
    write_json(output_dir / "existing_data_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
