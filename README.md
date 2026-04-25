# XBRL Error Localization Code Repository

## Overview

This repository contains the implementation code for the paper:

**"Constraint-Based Root-Cause Localization for XBRL Calculation Inconsistencies: Theory and Validation"**

Published in International Journal of Data Science and Analytics (JDSA), Special Issue on Data Science and AI in Finance.

## Project Structure

```
├── src/                    # Source code
│   ├── unified_ranker.py   # Main unified XGBoost ranker
│   ├── cross_validation.py # 5-fold cross-validation evaluation
│   ├── baseline_comparison.py # Baseline method comparisons
│   ├── statistical_analysis.py # McNemar test, bootstrap CI
│   ├── theorem_validation.py   # Theorem 5.1 numerical validation
│   ├── theorem5_noise_sweep.py # Noise threshold experiments
│   ├── theorem5_plots.py       # Figure generation
│   ├── sec_controllable_benchmark.py # SEC semisynthetic benchmark
│   ├── sec_evaluation_proper.py # SEC zero-shot evaluation
│   ├── baselines/               # Baseline implementations
│   │   ├── repair_baselines.py  # Leave-one-out repair scoring
│   │   └── real_data_baselines.py # Real XBRL baselines
│   └── ...
├── data/                   # Benchmark data
│   ├── synthetic_benchmark.json # Synthetic benchmark (120 cases)
│   ├── sec_semisynthetic_benchmark.json # SEC benchmark (117 cases)
│   ├── experiment_results.json  # Main experiment results
│   ├── cv_results.json          # Cross-validation results
│   └── ...
├── figures/                # Generated figures
│   ├── framework_architecture.png  # Two-stage framework (Figure 1)
│   ├── theorem5_combined.png       # Theorem validation (Figure 2)
│   └── ...
└── docs/                   # Documentation
```

## Key Results

| Benchmark | Top-1 | Top-3 | Top-5 | MRR |
|-----------|-------|-------|-------|-----|
| Synthetic (multi-constraint) | **87.28%** | 95.83% | 100% | 0.93 |
| SEC semisynthetic (single-constraint) | 33.3% | 100% | 100% | 0.56 |

## Requirements

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

## Quick Start

```python
# Run unified ranker evaluation
python src/unified_ranker.py

# Run theorem validation
python src/theorem5_noise_sweep.py

# Generate figures
python src/theorem5_plots.py
```

## Method

### Stage 1: Candidate Generation
- Violation Score: VS(j) = Σ|r_k|·|A_{k,j}|
- Top-K selection (K=5)
- Recall@5 = 100%

### Stage 2: Unified Ranking
- 38 features (role, residual, structural)
- XGBoost with 5-fold CV
- Role-aware features address total-component ambiguity

## Citation

```bibtex
@article{tian2026xbrl,
  title={Constraint-Based Root-Cause Localization for XBRL Calculation Inconsistencies: Theory and Validation},
  author={Tian, Zihan},
  journal={International Journal of Data Science and Analytics},
  year={2026}
}
```

## License

MIT License

## Author

Zihan Tian (田子晗)
- Xinjiang University, China
- Institut Polytechnique de Paris, France
- Email: 3208566786@qq.com