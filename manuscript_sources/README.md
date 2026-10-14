# Manuscript and eight-table source files

This directory contains only the final v32 manuscript source, filled source, eight-table JSON and authoring helpers for the article. It contains no prior manuscript/SI versions, operational logs or document-build receipts. The scientific values are those already published in the accompanying data directories; they are not recalculated during deposit preparation.

`manuscript.md` is the source with number keys, `manuscript_filled.md` is the final filled text, and `owner_tables.json` holds the eight final tables. `make_tables.py`, `fill_numbers.py`, `submission_table_layout.py` and `cost_v3_publication.py` retain the authoring selection/layout logic, with package-relative paths. Small completed scientific inputs used by those selectors are in `scientific_inputs/R4/`. The postfreeze frontend source resolves to the sibling `frontend/` directory and the VAL decomposition to `VAL_MSE/`. The cost helper checks completed numerical evidence rather than relying on a private controller-state file.

No official journal template is bundled. Obtain current instructions and templates from the journal: https://www.ecice06.com/ . Word/PDF rendering and any author-selected template are outside these source files; final document artifacts are deposited separately by the authors.

The files can be inspected without invoking any authoring script. The authoring scripts generate their outputs in this directory when a reader explicitly executes them. The final deposited text/tables remain directly available, so regeneration is not required to access the results. No fitting, inference, new quality metrics or cost measurement is part of these authoring helpers.
