# Official MOS-only TCD quality completion

The missing same-identity TCD point was completed after the original evaluations with the independent official `nisqa_mos_only.tar` model. This model has218402parameters and a907215B weight file, distinct in model identity from the five-dimensional252458parameter model. The quality producer's weight was directly read and found byte-for-byte equal to the copy used for the existing official MOS-only CPU cost; no hash was calculated, no weight is redistributed, and the five-dimensional MOS head was not substituted.

The actual completion used one CUDA inference pass over the fixed384files, preserving96reference groups and96conditions. The chop subset contains96files,24reference groups and24conditions. The original TSP/four-speaker source overlap and unknown exact NISQA concatenation crosswalk remain disclosed. This evaluation is not evidence of complete source/speaker disjointness.

Raw file-level pooled ALL metrics: Pearson0.8420599973613129, MAE0.6065984550347223, RMSE0.7633518508893653, Spearman0.8432514921308016. There is one TCD domain; the ALL support is the same384-file support as the earlier TCD/EQUAL_DOMAIN scope. No TRAIN mapping, paired hypothesis, Holm correction or bootstrap interval was added. These predictions do not change the original5855-row quality table,72comparison rows, original evaluation ledgers or measured cost matrix.

`TCD_MOS_predictions.csv` and `TCD_fixed_support_metadata.csv` retain exact file identities and MOS values. `TCD_MOS_raw_file_ALL_metrics.csv` contains the descriptive metrics. `TCD_MOS_FIGURE_JOIN_INTERFACE.json` connects this model to the already measured `NISQA_MOS/official` cost identity. No inference elapsed time is used as a cost result.

For optional reproduction, obtain the original TCD-VOIP corpus and author code/weight separately. The metadata's `archive_member` is relative to DATA_ROOT, for example `TCD-VOIP/Test Set/echo/...`. Author source: https://github.com/gabrielmittag/NISQA ; corpus source: https://www.mee.tcd.ie/~sigmedia/Resources/TCD-VoIP . Their terms apply; model/audio payloads are not bundled.

```
python postfreeze_TCD_MOS_completion/produce_official_MOS_TCD.py --data-root YOUR_DATA_ROOT
python postfreeze_TCD_MOS_completion/evaluate_raw_saved_predictions.py --predictions postfreeze_TCD_MOS_completion/reproduced_predictions/TCD_MOS_predictions.csv
```

The inference source uses the original author dataset/model, CUDA batch16, matmul TF32 disabled and cuDNN TF32 enabled, matching the actual completed implementation. The evaluation source reports only raw descriptive metrics. Both default to separate outputs and leave deposited results intact.
