# Detecting Zero-Day Attacks via Reconstruction of Feature Influence and Model Uncertainty

Code for:

> H. Fawaz, J. Talpini, M. Savi, S. Giordano, O. Ayoub, "Detecting Zero-Day Attacks via
> Reconstruction of Feature Influence and Model Uncertainty," *IEEE Transactions on
> Network and Service Management*, 2026. https://ieeexplore.ieee.org/abstract/document/11683510/

**Authors:** Hussein Fawaz, Jacopo Talpini, Marco Savi, Silvia Giordano, Omran Ayoub

**Affiliations:**
- H. Fawaz — University of Applied Sciences and Arts of Southern Switzerland (SUPSI), Switzerland; Università della Svizzera italiana (USI), Switzerland
- J. Talpini — University of Applied Sciences and Arts of Southern Switzerland (SUPSI), Switzerland; Department of Informatics, Systems and Communication (DISCo), University of Milano-Bicocca, Italy
- M. Savi — Department of Informatics, Systems and Communication (DISCo), University of Milano-Bicocca, Italy
- S. Giordano — University of Applied Sciences and Arts of Southern Switzerland (SUPSI), Switzerland
- O. Ayoub — University of Applied Sciences and Arts of Southern Switzerland (SUPSI), Switzerland

**Corresponding author:** Hussein Fawaz ([hussein.fawaz@usi.ch](mailto:hussein.fawaz@usi.ch))

## Overview

This repository implements a zero-day (unknown attack) detection pipeline for network
intrusion detection. A classifier (XGBoost or Random Forest) is trained on a set of
*known* attack types while one attack type is held out as the simulated unknown/zero-day
class. The pipeline then flags the unknown class as anomalous by combining:

1. **SHAP** feature-attribution vectors for each flow,
2. the classifier's **epistemic uncertainty** (via sub-ensemble sampling), and
3. an **autoencoder**'s reconstruction error in SHAP space,

and clustering the resulting 2D space `[reconstruction error, epistemic uncertainty]`
with a **Gaussian Mixture Model**; low-density regions are flagged as unknown attacks.
A raw-feature-space autoencoder and an Energy-based Flow Classifier (EFC) are included
as baselines for comparison.

## Structure

```
reliable_ids/
  config.py       # dataset paths, attack lists
  utils.py        # data loading/preprocessing, feature selection, metrics, GMM selection
  models.py       # XGBoost/RF training, SHAP explanations, epistemic uncertainty
  autoencoder.py  # autoencoder used for reconstruction-error anomaly detection
  pipeline.py     # ClusterZeroDayPipeline: the core approach described above
  baselines.py    # baseline classifier evaluation (known-class performance only)
requirements.txt
```

`reliable_ids/` is a namespace package: run modules from the repository root with `-m`
so relative imports and default output paths (`./Results`) resolve correctly.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The EFC baseline (`efc`) is installed directly from its GitHub repository and needs
`cython` and `numpy` available at build time. If the `pip install -r requirements.txt`
above fails while building `efc`, install those two first and retry:

```bash
pip install cython numpy
pip install -r requirements.txt
```

## Datasets

Supported datasets are registered in `reliable_ids/config.py` under `DATASET_INFO`:
`nf_ton_iot` (NF-ToN-IoT), `ciciot_23` (CICIoT-2023), and `cicddos_2019` (CIC-DDoS2019).
Each entry lists the expected file/folder name and its known attack classes.

Download the corresponding public dataset release and place it so the paths in
`DATASET_INFO` resolve, i.e. by default under `./data/<name>` — or point elsewhere with:

```bash
export IDS_DATA_DIR=/path/to/your/datasets
```

Select which dataset to run with `DATASET_B`:

```bash
export DATASET_B=ciciot_23   # one of the DATASET_INFO keys above
```

## Running

```bash
export DATASET_B=ciciot_23
python -m reliable_ids.pipeline
```

This reproduces the paper's results end to end: for `K=5` seeded repetitions
(seed = `42 + k`, re-seeding Python's `random`, NumPy, and TensorFlow each time), it
holds out every attack type in turn as the unknown class, runs both classifier
backends (XGBoost, RF) and the baselines, and writes per-repetition results under
`./Results/<Month><Day>/<dataset>/run_<k+1>/<attack>/` — per-attack metrics, ROC
points, threshold-sensitivity curves, and inference latency breakdowns — plus the
aggregated mean/std across all `K` repetitions under
`./Results/<Month><Day>/<dataset>/aggregated/`.

To change the number of repetitions, call `main(K=...)` from `reliable_ids.pipeline`
directly instead of running the module as a script.

To evaluate baseline classifiers' performance on known classes only (no zero-day
detection — see `reliable_ids/baselines.py`), run:

```bash
python -m reliable_ids.baselines
```

## Citation

```bibtex
@article{fawaz2026reliableids,
  title   = {Detecting Zero-Day Attacks via Reconstruction of Feature Influence and Model Uncertainty},
  author  = {Fawaz, Hussein and Talpini, Jacopo and Savi, Marco and Giordano, Silvia and Ayoub, Omran},
  journal = {IEEE Transactions on Network and Service Management},
  year    = {2026}
}
```
