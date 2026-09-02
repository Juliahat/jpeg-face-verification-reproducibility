#!/usr/bin/env python3
"""Build a persistent, resumable FaceNet embedding cache for suggestion 3.

This reproduces the old experiment's image path exactly:
  Color FERET member -> in-memory PPM decode -> optional JPEG encode/decode
  -> MTCNN detect/align (160 px, zero margin, largest face) ->
  VGGFace2 InceptionResnetV1 -> explicit L2 normalisation.

References remain Original because the dissertation protocol compresses probes only.
No raw image or face crop is written to disk. One compressed NPZ is written per
completed image-role/condition, so interruption only loses the current condition.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # Passing an explicitly closed binary handle avoids a Windows race in
        # which numpy/NamedTemporaryFile can retain a sharing lock at replace.
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows Defender/indexers may briefly open a newly completed NPZ.
        # Retry the atomic promotion without ever accepting the temp name as a
        # completed cache.
        last_error: Optional[PermissionError] = None
        for _attempt in range(8):
            try:
                os.replace(temporary, path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.25)
        if last_error is not None:
            # Some managed Windows installations keep a persistent read/share
            # lock on new archive files. Copying the already-fsynced temp file
            # is the compatibility fallback. A crash during this copy cannot
            # create a false checkpoint because validate_cache opens the NPZ
            # and verifies every key, shape, role, condition, ID and path.
            shutil.copyfile(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                # Never hide the original write/replace exception. A stale
                # temp file is recognisable and is not treated as a cache.
                pass


class ArchiveImageStore:
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
    image: Image.Image, condition: str, original_size_bytes: int
) -> Tuple[Image.Image, int]:
    if condition == "Original":
        return image.copy(), int(original_size_bytes)
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=int(condition),
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


def extract_embedding(face: Any, model: Any, device: Any) -> np.ndarray:
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


def process_image(
    store: ArchiveImageStore,
    image_path: str,
    condition: str,
    role: str,
    mtcnn: Any,
    model: Any,
    device: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "embedding": None,
        "detected": False,
        "size_bytes": -1,
        "failure_reason": "",
        "detail": "",
    }
    try:
        original, original_size = store.load(image_path)
    except Exception as exc:
        result["failure_reason"] = "image_read_failed"
        result["detail"] = f"{type(exc).__name__}: {exc}"[:500]
        return result
    try:
        conditioned, size_bytes = apply_jpeg_condition(original, condition, original_size)
        result["size_bytes"] = size_bytes
    except Exception as exc:
        result["failure_reason"] = "jpeg_compression_failed"
        result["detail"] = f"{type(exc).__name__}: {exc}"[:500]
        return result
    try:
        face = mtcnn(conditioned)
    except Exception as exc:
        result["failure_reason"] = f"{role}_detection_failed"
        result["detail"] = f"{type(exc).__name__}: {exc}"[:500]
        return result
    if face is None or (hasattr(face, "numel") and face.numel() == 0):
        result["failure_reason"] = f"{role}_detection_failed"
        return result
    result["detected"] = True
    try:
        result["embedding"] = extract_embedding(face, model, device)
    except Exception as exc:
        result["failure_reason"] = "embedding_failed"
        result["detail"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def load_indices(assignment_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assignment = pd.read_csv(assignment_path, dtype={"subject_id": str})
    assignment = assignment.sort_values("subject_id", kind="stable").reset_index(drop=True)
    refs = assignment[["subject_id", "reference_path"]].copy()
    refs = refs.rename(columns={"reference_path": "image_path"})
    probe_rows: List[Dict[str, str]] = []
    for record in assignment.to_dict("records"):
        paths = json.loads(record["probe_paths"])
        if len(paths) != int(record["num_probe_images"]):
            raise RuntimeError(f"Probe path count mismatch for subject {record['subject_id']}")
        for path in paths:
            probe_rows.append({"subject_id": str(record["subject_id"]), "image_path": str(path)})
    probes = pd.DataFrame(probe_rows)
    if refs["subject_id"].duplicated().any() or probes["image_path"].duplicated().any():
        raise RuntimeError("Reference subjects and probe image paths must be unique")
    return refs, probes


def cache_path(cache_dir: Path, role: str, condition: str) -> Path:
    safe_condition = "original" if condition == "Original" else f"q{condition}"
    return cache_dir / f"{role}_embeddings_{safe_condition}.npz"


def validate_cache(path: Path, index: pd.DataFrame, role: str, condition: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return bool(
                data["embeddings"].shape == (len(index), 512)
                and data["valid"].shape == (len(index),)
                and data["subject_id"].astype(str).tolist() == index["subject_id"].astype(str).tolist()
                and data["image_path"].astype(str).tolist() == index["image_path"].astype(str).tolist()
                and str(data["role"].item()) == role
                and str(data["condition"].item()) == condition
            )
    except Exception:
        return False


def build_one_cache(
    path: Path,
    index: pd.DataFrame,
    role: str,
    condition: str,
    store: ArchiveImageStore,
    mtcnn: Any,
    model: Any,
    device: Any,
) -> Dict[str, Any]:
    embeddings = np.full((len(index), 512), np.nan, dtype=np.float32)
    valid = np.zeros(len(index), dtype=bool)
    detected = np.zeros(len(index), dtype=bool)
    sizes = np.full(len(index), -1, dtype=np.int64)
    reasons = np.full(len(index), "", dtype="<U64")
    details = np.full(len(index), "", dtype="<U500")
    try:
        from tqdm import tqdm

        iterator = tqdm(index.itertuples(index=False), total=len(index), desc=f"{role} {condition}")
    except Exception:
        iterator = index.itertuples(index=False)
    for row_index, row in enumerate(iterator):
        result = process_image(
            store, str(row.image_path), condition, role, mtcnn, model, device
        )
        if result["embedding"] is not None:
            embeddings[row_index] = result["embedding"]
            valid[row_index] = True
        detected[row_index] = bool(result["detected"])
        sizes[row_index] = int(result["size_bytes"])
        reasons[row_index] = str(result["failure_reason"])
        details[row_index] = str(result["detail"])
    atomic_npz(
        path,
        embeddings=embeddings,
        valid=valid,
        detected=detected,
        size_bytes=sizes,
        failure_reason=reasons,
        detail=details,
        subject_id=index["subject_id"].astype(str).to_numpy(dtype="<U32"),
        image_path=index["image_path"].astype(str).to_numpy(dtype="<U256"),
        role=np.asarray(role),
        condition=np.asarray(condition),
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "rows": int(len(index)),
        "valid_embeddings": int(valid.sum()),
        "failures": int((~valid).sum()),
        "condition": condition,
        "role": role,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--conditions", nargs="*")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    config = json.loads((analysis_dir / "analysis_config.json").read_text(encoding="utf-8"))
    conditions: Sequence[str] = args.conditions or config["conditions"]
    invalid = sorted(set(conditions) - set(config["conditions"]))
    if invalid:
        raise ValueError(f"Conditions not declared in analysis_config.json: {invalid}")
    tar_path = (analysis_dir / config["raw_tar_relative_to_analysis_dir"]).resolve()
    assignment_path = (
        analysis_dir / config["canonical_assignment_relative_to_analysis_dir"]
    ).resolve()
    cache_dir = analysis_dir / "embedding_cache"
    output_dir = analysis_dir / "outputs" / "embedding_run"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs, probes = load_indices(assignment_path)
    refs.to_csv(cache_dir / "reference_index.csv", index_label="embedding_index")
    probes.to_csv(cache_dir / "probe_index.csv", index_label="embedding_index")

    import torch
    import torchvision
    import facenet_pytorch
    from facenet_pytorch import InceptionResnetV1, MTCNN

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Embedding device: {device}", flush=True)
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        keep_all=False,
        select_largest=True,
        post_process=True,
        device=device,
    )
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    tasks: List[Tuple[str, str, pd.DataFrame]] = [("reference", "Original", refs)]
    tasks.extend(("probe", condition, probes) for condition in conditions)
    summaries: List[Dict[str, Any]] = []
    with ArchiveImageStore(tar_path) as store:
        for role, condition, index in tasks:
            path = cache_path(cache_dir, role, condition)
            if validate_cache(path, index, role, condition):
                print(f"Cache valid; skipping {path.name}", flush=True)
                with np.load(path, allow_pickle=False) as data:
                    summaries.append({
                        "path": path.name,
                        "sha256": sha256_file(path),
                        "rows": int(len(index)),
                        "valid_embeddings": int(data["valid"].sum()),
                        "failures": int((~data["valid"]).sum()),
                        "condition": condition,
                        "role": role,
                        "resumed": True,
                    })
                continue
            print(f"Building {path.name} ({len(index):,} images)", flush=True)
            summary = build_one_cache(
                path, index, role, condition, store, mtcnn, model, device
            )
            summary["resumed"] = False
            summaries.append(summary)
            atomic_json(output_dir / "embedding_cache_manifest.json", summaries)

    versions = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "Pillow": Image.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "facenet_pytorch": getattr(facenet_pytorch, "__version__", "2.6.0 package has no __version__"),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mtcnn": {
            "image_size": 160,
            "margin": 0,
            "keep_all": False,
            "select_largest": True,
            "post_process": True,
        },
        "embedding_model": "InceptionResnetV1(pretrained='vggface2')",
        "compression_mode": "probe-only",
        "preprocess_order": "compress-before-detect",
        "jpeg": {
            "subsampling": "4:2:0 (Pillow subsampling=2)",
            "optimize": False,
            "progressive": False,
            "color_mode": "RGB",
        },
        "raw_images_or_face_crops_saved": False,
    }
    atomic_json(output_dir / "software_and_protocol.json", versions)
    atomic_json(output_dir / "embedding_cache_manifest.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
