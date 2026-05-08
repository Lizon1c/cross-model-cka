# Cross-Model CKA: Low-Frequency Smoothing Bias in EVO2/FAESM–RFdiffusion3 Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Cheng Ding**  
ShanghaiTech University  
dingcheng2024@shanghaitech.edu.cn

This repository contains the complete analysis pipeline for the paper:

**"Cross-Model CKA Alignment Between EVO2 and RFdiffusion3 Reflects Low-Frequency Smoothing Bias Rather Than Structural Convergence"**

## Repository Structure

```
├── paper/                     # Manuscript (PDF, DOCX, source)
├── scripts/                   # Analysis pipeline
│   ├── core/                  # Position mapping, CKA library
│   ├── scan/                  # Full head-pair CKA scan
│   ├── controls/              # Max-null, Pearson, segment analysis
│   ├── diagnostics/           # Frame-shift, position controls, residual scans
│   └── report/                # Report generation
├── data/                      # Processed results
│   ├── matrices/              # CKA matrices
│   ├── nulls/                 # Null distributions
│   └── figures/               # Publication figures
├── LICENSE
└── README.md
```

## Key Results

| Control | EVO2×RF3 | FAESM×RF3 | FAESM×EVO2 |
|---------|----------|-----------|------------|
| Raw CKA | 0.851 | 0.870 | 0.101 |
| Cubic residual | 0.004 | 0.272 | 0.001 |
| Fourier20 residual | 0.000 | 0.024 | 0.000 |
| Delta excess (obs−shuf) | −0.005 | −0.005 | — |
| Retrieval top-1 | 1.0% | 0.0% | — |

## Citation

```bibtex
@article{ding2025crossmodelcka,
  title={Cross-Model CKA Alignment Between EVO2 and RFdiffusion3 Reflects
         Low-Frequency Smoothing Bias Rather Than Structural Convergence},
  author={Ding, Cheng},
  year={2025},
  eprint={arXiv:XXXX.XXXXX}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
