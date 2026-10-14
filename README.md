# 面向流式部署的轻量无参考语音质量评估 — scientific data and code

This repository contains the numerical evidence and research code accompanying the manuscript. The main model is the frozen two-layer GRU96 (`v2_pure`); GRU64 is the lower-cost variant. Model identities, prediction column names and all retained scientific values match the completed source results. No speech waveform, third-party weight, trained-model payload or private operational log is distributed in this package.

## Access map

| Evidence | Relative location |
|---|---|
| Primary quality results, 5855 rows | `code/R4/exp_C2/results/C_primary_accuracy_v3.csv` |
| Paired comparisons, 72 rows / 24 three-comparison Holm families | `code/R4/exp_C2/results/C5_Holm72_v3.csv` |
| Original registered prediction tables | `code/R4/exp_B/preds/` and `code/R4/exp_C2/preds/` |
| VAL / TEST evaluation-registration facts | `code/R4/shared/VAL_LEDGER.csv`, `TEST_LEDGER.csv` |
| Final cost results, 16 settings | `code/R4/exp_D/cost_v3/results/cost_final.csv` |
| All 185 formal runs and bounded repeats | `code/R4/exp_D/cost_v3/results/run_index.csv`; linked run, prediction and calibration JSONs in the same directory |
| Acceptance outcomes and measured environment | `code/R4/exp_D/cost_v3/results/COST_ACCEPTANCE.json` |
| Independent process memory, 16 measurements | `code/R4/exp_D/cost_v3/results/memory/` |
| Streaming block timing and resident memory, 12 measurements | `code/R4/exp_D/cost_v3/results/streaming/`; six timing summaries in `streaming_kernel.csv` |
| Complete-path MAC estimate and sensitivity contrasts | `mac_table.csv`, `MAC_contrasts.csv`, `cost_contrasts.csv` in the cost results directory |
| Supported-input-rate state bounds | `stream_state_bounds_by_samplerate.csv`, `stream_state_bound_components.csv` in the cost results directory |
| Two complete-VAL representativeness checks | `code/R4/exp_D/cost_v3/results/representativeness/` |
| Fresh TRAIN frontend experiment, 120 fits | `frontend/`; four arms A/B/C/D, three seeds, ten splits per arm/seed |
| Original seed predictions and seed-metric-mean results | `frontend/predictions/`; `TRAIN_frontend_metrics.csv`, `TRAIN_frontend_paired_bootstrap.csv` |
| Current three-seed mean-prediction metrics and intervals | `frontend/rule_meanprediction_20261014/`; 12 mean-prediction tables in `mean_predictions/` |
| Result-after-request B-minus-D auxiliary comparison | `frontend/rule_meanprediction_20261014/aux_B_D_20261014/` |
| Requested postfreeze VAL raw-MOS MSE decomposition | `VAL_MSE/VAL_main_raw_MOS_MSE_decomposition.csv` |
| Scientific rules, exposure and timing of reporting amendments | `protocols/SCIENTIFIC_PROTOCOL.md`, `protocols/ANALYSIS_CHRONOLOGY.md` |
| File-level source mapping and sizes | `source_data_manifest.csv` |
| Scientific-value preservation checks | `scientific_value_preservation.csv`, `PUBLIC_PACKAGE_VALIDATION.json` |

The 16 final cost settings are **completed**, with **8 accepted** and **8 environment-terminal** after the permitted bounded repeats. Failed calibration/frequency/dispersion gates and historical cross-device numerical differences are retained. Completion does not imply environment acceptance. No third repeat was added. S3 speed contrasts against unaccepted references use an explicitly labeled calibration-drift sensitivity assumption, not a confidence interval or an accepted speedup claim.

## Data and external assets

Obtain original datasets and official models through the providers listed in `DATA_ACCESS.md` and comply with their separate terms. Set `DATA_ROOT` to a user-chosen asset directory; `PYTHONPATH` should include `code/R4`, `code/R4/shared`, `code/R4/exp_A` and the cost directory when using the runtime adapters. Package-internal provenance is relative to the repository root. External provenance uses the `DATA_ROOT/` prefix; `code/R4/public_paths.py` and the adapter resolver expand it at execution time.

Saved-prediction statistical reproduction requires no waveform or model weights. Audio inference and retraining additionally require lawful datasets, caches and weights that are intentionally outside this small-file deposit. Their expected locations are documented, rather than claiming the deposit is a self-contained audio or pretrained-model release. The actual full-utterance candidate1 implementation and the lean streaming implementation are separate code paths; whole-process RSS must not be inferred from streaming RSS or state bytes.

## Analysis conventions

Quality identities are preserved: official NISQA five-dimensional quality and official MOS-only runtime cost are different models and must not be combined into a single quality-cost point. Primary raw correlations and frozen-TRAIN-mapped RMSE are separate outcomes. Reference-group bootstrap is used for TRAIN/VAL, condition bootstrap for NISQA TEST, and the originally specified source/domain weighting is preserved. Macro intervals that were not estimated remain unavailable. P563's eight unavailable VAL estimates remain missing.

The frontend reporting analysis averages three seed predictions before metrics and bootstrap. Its tier rule was received after the 120-fit outcomes; the request chronology is disclosed. The original B reproduction gate remains based on the original seed-metric-mean statistic, not the new ensemble statistic. Original and new aggregation tables have distinct names. Negative effects across all five heads are retained.

## Software and licensing

See `LICENSE_AND_ATTRIBUTION.md`. No repository-wide MIT claim is made. The authors have not selected an additional license for their own code and data. Authors remain blank in the manuscript, and no DOI is assigned or requested. The repository is public at https://github.com/chenyikai1207/nisqa-streaming-quality. Third-party code/weights retain their providers' separate licenses. Public readers can inspect all deposited numerical results while those publication metadata are being finalized.

This package has been read completely and its CSV/JSON payloads parsed. Source-to-derived scientific values were checked; machine paths and operational metadata were the only intended changes. No new model inference, fit, timing or hash calculation was performed in assembling it.

## Manuscript, figures and version

The Computer Engineering manuscript is available in `manuscript/` as an editable DOCX and a PDF: **11 pages including references, 8 editable tables, 4 main figures and 7 editable equations**. Chinese and English abstracts, captions and reference entries follow the journal's structure. The main text presents the evidence needed for its conclusions; detailed results are provided in this repository.

`manuscript_sources/` contains the manuscript text, eight-table definitions, number-filling scripts and their small scientific inputs. `figures/` contains the four final Chinese-labeled figures, plotting code, source tables, vector PDF/SVG and 600-dpi PNG files. The framework additionally has an editable Visio VSDX and EMF source.

Version: `nisqa-paper1-ecice-v32-20261014` at https://github.com/chenyikai1207/nisqa-streaming-quality. The documents were exported with **WPS Office**, as requested by the authors. Equations are native editable OMML objects; they have not been converted to MathType. Author and funding fields are intentionally blank pending author completion. No Zenodo deposit or DOI is assigned.
