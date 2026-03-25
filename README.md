# Investigation of Variance Reduction in Ensemble Methods

An experiment conducted in Python comparing bias-variance tradeoffs across bagged ensemble methods, conducted as part of the Statistical Learning with High Dimensional data course at Umeå University.

## Overview

This project investigates the claim that ensemble methods reduce variance while keeping bias constant. Some criterion for the project had to be fulfilled. The claim had to be tested using real world data. Ideally such a claim would be tested on simulate data. Also three different methods had to be included when testing the claim. Lastly the report could be no longer than 6 pages including figures and tables.

Three base learners with different variance profiles are compared:

- **Decision Tree** (high variance)
- **K-Nearest Neighbors** (medium variance)
- **Logistic Regression** (low variance)

## Key Findings

- Decision tree ensembles saw the largest variance reduction
- KNN ensembles also showed meaningful variance reduction
- Logistic regression ensembles showed minimal improvement
- Bias remained approximately constant for all models

## Tools

Python, scikit-learn, NumPy, pandas, matplotlib

## Files

- `ensemble_variance_reduction.py` — full experiment code
- `ensemble_variance_reduction_report.pdf` — written report with methodology and results
