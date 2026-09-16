"""
Baseline classifier evaluation.

Trains XGBoost, Random Forest, and EFC on a leave-one-attack-out basis and reports
their classification performance on the known classes only. This does NOT run the
zero-day/unknown-attack detection approach itself (see `pipeline.py` for that) — it
just measures how well the classifiers backing that approach perform at their
ordinary, known-class classification job.
"""
import os
import warnings
from .config import ALL_ATTACKS
from .utils import read_data, evaluate_base_models

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    data = read_data()
    base_model_results = evaluate_base_models(data, ALL_ATTACKS)
    print("\n=== Base Model Performance Summary ===")
    print(base_model_results)
