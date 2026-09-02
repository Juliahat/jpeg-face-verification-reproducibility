# Data Availability and Governance

## Color FERET source data

This study uses the Color FERET face-image database obtained through the NIST account-request process. The Color FERET archive is not included in this repository and must not be copied, published or redistributed through this project.

Researchers wishing to reproduce the image-level experiment must request their own authorised copy from NIST and comply with the applicable Color FERET Release Agreement. The expected local archive name is `colorferet.tar`; this location can be changed in `new_update_v2/analysis_config.json`.

The required acknowledgement for publications using Color FERET must be included in the associated manuscript together with the FERET database references specified by NIST.

## Publicly shareable outputs

The publication artifact may include code, frozen configuration, aggregate numerical results, validation reports, source hashes and aggregate plots. The principal public result tables are:

- `new_update_v2/outputs/analysis/development_thresholds_all_nonmates.csv`;
- `new_update_v2/outputs/analysis/protocol_pair_counts.csv`;
- `new_update_v2/outputs/analysis/evaluation_fnmr_same_thresholds.csv`;
- `new_update_v2/outputs/analysis/evaluation_fmr_all_and_hard.csv`;
- `new_update_v2/outputs/analysis/paired_fmr_delta_vs_original.csv`;
- `new_update_v2/outputs/supplemental_transition/genuine_metrics_subject_bootstrap.csv`;
- `new_update_v2/outputs/supplemental_transition/adjacent_paired_fnmr_tar_subject_bootstrap.csv`;
- `new_update_v2/outputs/supplemental_transition/cross_seed_transition_summary.csv`;
- `new_update_v2/outputs/supplemental_transition/fmr_subject_bootstrap_topk_sensitivity.csv`;
- `new_update_v2/outputs/supplemental_transition/adjacent_paired_fmr_subject_bootstrap.csv`;
- `new_update_v2/outputs/degradation_point_search/simultaneous_paired_fnmr_results.csv`; and
- the aggregate CSV files in `new_update_v2/storage_baseline_reanalysis_v1/outputs/`.

These files report derived statistics and do not contain face images.

## Restricted derived artifacts

The following artifacts are intentionally excluded from the public GitHub repository even though they were generated during the study:

- raw Color FERET files and extracted images;
- detected or aligned face crops;
- FaceNet embedding arrays in `new_update_v2/embedding_cache/`;
- embedding-to-subject index files;
- identity-linked reference/probe assignments;
- identity-linked genuine, impostor and hard-impostor pair lists;
- per-comparison score tables containing FERET subject identifiers or internal archive paths; and
- checkpoint chunks, partial results and diagnostic logs.

Face embeddings are derived biometric data. They are retained only in the controlled local research environment for the period authorised by the applicable institutional governance and retention arrangements. They are not released under the repository software licence.

## Reproduction without redistributed data

An authorised researcher can reproduce the restricted intermediate artifacts by:

1. obtaining Color FERET independently from NIST;
2. installing the environment recorded in `environment-feret-gpu-v1.yml`;
3. running the retained formal experiment runner for the documented seeds and conditions; and
4. running the retained data-generation and audit scripts for the required numerical stage.

This public subset intentionally excludes mixed-purpose scripts that also generated manuscript-ready or supervisor-report prose. Their unchanged aggregate CSV outputs and independent validation records are retained for numerical inspection.

## Third-party resources

The VGGFace2-pretrained InceptionResnetV1 weights used through `facenet-pytorch` are third-party resources and are governed by their own licences and source-dataset terms. They are downloaded through the normal model cache and are not committed to this repository.

The storage sensitivity analysis cites Commission Implementing Decision (EU) 2019/329. The 1:20 value is used only to derive an illustrative boundary-equivalent size for this experiment's raster dimensions; it is not a redistributed dataset or a measured operational EES file-size sample.

## Ethics and institutional records

No new participants were recruited and no new biometric samples were collected by this experiment. The final manuscript and archived artifact must state the confirmed institutional ethics approval or waiver identifier. That identifier is not inferred or fabricated by the repository.
