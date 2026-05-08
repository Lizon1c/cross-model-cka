Cross-Model CKA Analysis Pipeline
==================================

Requirements: Python 3.10+, PyTorch 2.0+, numpy, scipy, biopython, pandas

All scripts import from the project root via sys.path. Before running:
  export PROJECT_ROOT=/path/to/DMS_Project
  export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

Or modify scripts to use relative imports from their own directory.

Core Libraries:
  core/align_residues.py          — RF3/EVO2 422-aa position ID mapping
  core/evo2_to_rf3_align_v32.py   — CKA/RSA/codon_pool functions + sanity checks

Scan:
  scan/full_scan.py               — 160 EVO2 × 1216 RF3 full CKA matrix (GPU)

Controls:
  controls/max_null.py            — Max-null 200 permutations
  controls/max_null_2000.py       — Max-null 2,000 permutations
  controls/per_residue_pearson.py — Strict Pearson per-residue + domain enrichment
  controls/segment_analysis.py    — Segment-level CKA (Tables 2-5)
  controls/cross_mutant.py        — Cross-mutant robustness scan

Diagnostics:
  diagnostics/frame_shift_scan.py    — Codon frame-shift systematic scan
  diagnostics/shift_diagnostics.py   — Shift invariance controls
  diagnostics/position_controls.py   — Polynomial regression, block shuffle
  diagnostics/residual_scan.py       — Residualized full scan
  diagnostics/chain_ab_diag.py       — Chain A/B comparison

Report:
  report/generate_report.py       — Generate markdown report
  report/best_pairs_analysis.py   — Best-pair deep analysis

IMPORTANT: Runtime Dependencies
===============================
These scripts depend on run_attn_extraction.py (not included in this repo)
which defines CRISPR locus construction constants (WT_PROTEIN_SEQ, codon table,
flanking sequences, reverse_translate_ascas12f, build_ascas12f_minimal_crispr_locus).

The core library align_residues.py uses exec() to load constants from
run_attn_extraction.py. To run the pipeline, place run_attn_extraction.py
in the project root and set:

  export PROJECT_ROOT=/path/to/DMS_Project
  export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

Alternatively, extract the relevant constants from run_attn_extraction.py
(lines 60-239) into a standalone locus_constants.py file.
