# External data and assets

The deposit contains derived predictions/results, record identities and research code; it does not redistribute original audio or third-party trained models. Metadata labels and record identifiers in the result files remain subject to the originating dataset terms. Users should independently obtain the source data for reproduction.

| Source | Official access | Role and limitation |
|---|---|---|
| NISQA Corpus | https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus | TRAIN/VAL and FOR/P501/LIVETALK tests; provider dataset-specific licenses apply. Official NISQA training includes its TRAIN data, so teacher TRAIN results are not independent generalization evidence. |
| Official NISQA code / weights | https://github.com/gabrielmittag/NISQA | Five-dimensional and MOS-only models are distinguished. Code MIT; provided pretrained weights have the separately stated CC BY-NC-SA 4.0 terms. Neither weight payload is deposited here. |
| PLCMOS v2 | https://github.com/microsoft/PLC-Challenge/tree/main/PLCMOS | The official score uses 15 rater-network calls. External STFT is computed once per audio in the author implementation; seed23 embeddings are generated per audio. Official code license is MIT; this package does not redistribute the source model payload. |
| DNSMOS | https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS | Use the P835 implementation and its provider terms; no P808 substitution. Missing five-dimensional outputs remain missing. |
| SQUIM | https://pytorch.org/audio/stable/pipelines.html | Official objective pipeline, with its actual proxy and missing dimensions preserved; obtain assets through the provider. |
| TCD-VoIP | https://www.mee.tcd.ie/~sigmedia/Resources/TCD-VoIP | External test with the original dataset terms and source-overlap disclosure; preserve the published file and condition identities. |

## DATA_ROOT layout for inference/retraining

These are user-supplied assets, not bundled files:

```
DATA_ROOT/
  NISQA_Corpus/                    # original published per-dataset paths
  TCD_VOIP/                       # original external dataset
  cache/packed/peak16000_mel48/    # original packed causal cache
  cache/packed/power16000_mel48/   # linear mel power, energy/mask/side metadata
  frozen_models/v2_pure/          # spec + mapping + independently provided model.pt
  frozen_models/v2_64/
  frozen_models/v2_conv64/
  frozen_models/v1/
  deployment_arrays/             # arrays derived from the same frozen models
  third_party/NISQA/             # author source and separately obtained weights
  third_party/PLCMOS/            # author source and separately obtained ONNX
```

Some retained historical provenance fields use `DATA_ROOT/external/<asset-name>` where the original machine path was not a published corpus-relative identity. They denote an external asset, not an included file. Source record IDs and all prediction values remain unchanged. Package internal path references begin `code/R4/`, `frontend/` or `VAL_MSE/`; execute statistical scripts from the package root or resolve them with `public_paths.package_path`.

The public derivative includes the exact small FIR coefficient table used by the stream implementation, without trained network arrays. State upper bounds are computed for supported native 16/48k input rates; unsupported rates are marked unavailable instead of borrowing the 48k observed state. The maximum supported recursive state is 7072 B for GRU96 and 6560 B for GRU64. Temporary scratch, input blocks and whole-process RSS are separate quantities.
