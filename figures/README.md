# Four final v32 figures and source data

The final article figures are:

1. `fig1_framework_native_zh_v32`: native editable Visio framework (`.vsdx`, `.emf`, `.svg`, `.pdf`, and `_view.png`), generated from `fig1_nodes_edges_zh_v32.json` with `build_native_visio_framework_v32.ps1`. A Matplotlib version is separately identified as `fig1_framework_zh_v32`; no model/data difference is implied.
2. `fig2_quality_zh_v32`: raw Pearson and fixed TRAIN-mapped RMSE. Numeric input: `fig2_source_rows_zh_v32.csv`. The main GRU96 row is emphasized; model identities and missing evaluations remain distinct.
3. `fig3_mse_seven_methods_zh_v32`: raw MOS MSE components with the existing marginal intervals. Numeric input: `fig3_source_rows_zh_v32.csv`, including the explicitly requested postfreeze VAL decomposition. These are algebraic components, not causal sources or pure rank-order error.
4. `fig4_pareto_zh_v32`: existing quality versus complete-path MAC estimate or S3 RTF. Numeric input: `fig4_source_rows_zh_v32.csv`. Unaccepted environment points retain their markers; official MOS-only cost and official five-dimensional quality are not combined into a different model identity.

SVG/PDF/PNG files were copied from the final article without plotting or recalculating values during packaging. The native Visio file is the final editable version, with43shapes and20editable text objects. Source CSV scientific numerical fields were checked against their source files without recomputing quality.

`make_four_main_figures_zh_v32.py` retains the final rendering logic, using package-relative scientific input or the deposited point/component CSVs. Private Codex/SKILL imports and version-preservation dependencies have been removed; no audit API is required. It uses Matplotlib's installed-font resolver. For the original appearance, use STZhongsong and Times New Roman where legally installed. An open CJK serif alternative can be obtained from https://github.com/notofonts/noto-cjk/tree/main/Serif ; fonts remain external and keep their original licenses. No proprietary font or official journal template is uploaded.

Optional regeneration, from the repository root:

```
python figures/make_four_main_figures_zh_v32.py --only all
```

This renders figures from existing values; it does not run a model, bootstrap, select a candidate or time inference. Native Visio regeneration uses Microsoft Visio on Windows and explicit `-SourceJson` / `-OutputFolder` arguments. The deposited native file remains editable without regenerating it.
