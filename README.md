# Enterprise Security Upsell Classifier

[![CI](https://github.com/cerenaaa/enterprise-upsell-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/enterprise-upsell-classifier/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Azure ML](https://img.shields.io/badge/cloud-Azure%20ML-0078D4.svg)](https://azure.microsoft.com/en-us/products/machine-learning)

Binary classification pipeline to identify enterprise accounts with high propensity for security product upsell. Deployed at scale on Azure ML; drove **$47M in incremental revenue** within six months of launch.

## Problem

Given firmographic, product usage, licensing, and behavioral signals across enterprise accounts, predict which accounts are most likely to expand into security products (e.g., Microsoft Defender for Endpoint, Sentinel).

## Approach

| Stage | Method |
|---|---|
| Feature engineering | Firmographic enrichment, usage velocity, license gap analysis |
| Modeling | XGBoost with Optuna + calibrated probabilities |
| Threshold selection | F-beta (β=0.5) optimized for precision-recall tradeoff |
| Explainability | SHAP values surfaced to sales reps via CRM integration |
| Deployment | Azure ML managed endpoints, batch scoring pipeline |
| Monitoring | PSI-based drift detection, weekly score refresh |

## Structure

```
enterprise-upsell-classifier/
├── data/
│   └── synthetic_accounts.py     # Synthetic enterprise account dataset
├── features/
│   ├── firmographic.py           # Company size, industry, region encoding
│   ├── usage_signals.py          # Product adoption velocity, engagement scores
│   └── license_gap.py            # License coverage gap features
├── models/
│   ├── xgboost_classifier.py     # XGBoost + Optuna + calibration
│   └── threshold_optimizer.py    # Precision-recall threshold selection
├── evaluation/
│   ├── metrics.py                # AUC, lift curves, decile analysis
│   └── drift_monitor.py          # PSI drift detection for production
├── pipeline/
│   └── azure_ml_pipeline.py      # Azure ML pipeline definition
├── train.py
└── score.py                      # Batch scoring entry point
```

## Quickstart

```bash
pip install -e ".[dev]"
python train.py --output results/
python score.py --input data/accounts.csv --output results/scores.csv
```

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.87 |
| Top-decile lift | 4.2x |
| Precision @ 30% recall | 0.71 |
| Revenue lift (6mo) | $47M |

## Key Design Decisions

- **Calibrated probabilities** allow sales teams to prioritize by expected revenue, not just rank
- **SHAP explanations** surfaced per account so reps understand *why* an account is flagged
- **PSI monitoring** detects account population drift between training and scoring windows
- **Azure ML pipelines** enable weekly retraining with zero manual intervention
