# Probe-Side JPEG Compression in Face Verification

Data-generation and audit artifact for the working manuscript **“JPEG Compression in Face Verification: Evaluating Performance–Storage Trade-offs and Security Implications in Digital Identity Systems.”**

This public directory is deliberately limited to experimental data generation, numerical aggregation, audit code, frozen configuration, validation records and aggregate CSV results. Scripts whose purpose includes generating manuscript-ready, thesis-ready or supervisor-report prose are excluded.

Color FERET images, identity-linked assignments and pair tables, per-comparison scores, hard-impostor identity lists and persisted face embeddings are not distributed. See `DATA_AVAILABILITY.md`.

## Included code

| Path | Function |
| --- | --- |
| `feret_formal_experiment_10imp.py` | Seed 42 formal JPEG/MTCNN/InceptionResnetV1 experiment runner. |
| `feret_formal_experiment_10imp_checkpoint_retry.py` | Restartable formal runner used in checkpoint validation. |
| `feret_formal_experiment_10imp_chunk_checkpoint.py` | Immutable chunk-checkpoint formal runner used for later seeded experiments. |
| `supplementary_transition_merge_audit.py` | Seed 42 refined-grid merge and paired numerical audit. |
| `audit_seed123_chunk_run.py` | Independent audit of each Seed 123 chunk run. |
| `merge_audit_seed123_chunk_split.py` | Audited merge of Seed 123 Parts A and B and generation of aggregate transition CSV files. |
| `audit_seed456_transition.py` | Seed 456 integrity, metric and bootstrap audit. |
| `audit_seed3_transition.py` | Seed 3 integrity, metric and bootstrap audit. |
| `audit_seed99_transition.py` | Seed 99 integrity, metric and bootstrap audit. |
| `prepare_seed456_report_update.py` | Generation of the three-seed aggregate CSV tables; it does not generate manuscript prose. |
| `new_update_v2/audit_existing_data.py` | Read-only audit of available sources, assignments and prior cache state. |
| `new_update_v2/build_embedding_cache.py` | Controlled reconstruction of normalised embeddings from an authorised local Color FERET archive. |
| `new_update_v2/analyze_hardened_impostors.py` | Exhaustive non-mate thresholds, FMR/FNMR statistics and hard-impostor numerical analysis. |
| `new_update_v2/validate_supplemental_transition.py` | Independent structural and numerical audit of the published supplemental CSV results. |

## Experimental flow

```text
Authorised local Color FERET archive
  -> formal seeded JPEG/face-verification runs
  -> seed-specific integrity and metric audits
  -> audited transition merges
  -> aggregate cross-seed CSV tables
  -> controlled embedding reconstruction
  -> exhaustive non-mate and hard-impostor numerical analysis
  -> independent validation records
  -> public aggregate CSV result snapshot
```

The five seeds are alternative Development/Evaluation allocations of one overlapping Color FERET cohort. They are sensitivity analyses, not five independent datasets.

## Environment

The recorded environment is provided in `environment-feret-gpu-v1.yml` and `requirements-feret-gpu-v1.txt`. The principal environment used Python 3.11, PyTorch 2.2.2 with CUDA 12.1, facenet-pytorch 2.6.0, Pillow 10.2.0, NumPy 1.26.4 and Pandas 2.2.3.

## Aggregate data

The retained CSV files contain condition-level, seed-level, bootstrap or validation aggregates. CSV files containing FERET identifiers, archive image paths, identity assignments, pair identifiers or per-comparison scores are excluded.

The principal final result groups are:

- `new_update_v2/outputs/analysis/`
- `new_update_v2/outputs/supplemental_transition/`
- `new_update_v2/outputs/degradation_point_search/`
- `new_update_v2/storage_baseline_reanalysis_v1/outputs/`

Some aggregate result snapshots were produced by historical scripts that combined numerical analysis with manuscript-ready prose generation. Those mixed-purpose scripts are intentionally not included in this public subset; the resulting aggregate CSV files and independent validation records are retained unchanged.

## Validation snapshot

The retained validation records report:

- core exhaustive-analysis validation: 16/16 checks passed;
- supplemental-transition validation: 34/34 checks passed;
- degradation-point validation: 11/11 checks passed; and
- storage-baseline validation: 13/13 checks passed.

## Reproducibility boundary

Full image-level reproduction requires an independently authorised Color FERET copy and locally regenerated identity-linked intermediate files. Restricted inputs are not supplied merely to make historical scripts executable. The public aggregate results support numerical inspection and audit without redistributing biometric source material.

## Citation and licence

Citation metadata is provided in `CITATION.cff`. Add the repository DOI after one has been issued. Original repository code is released under the MIT License; Color FERET and pretrained model weights retain their own terms.
