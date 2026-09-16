import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
import shap


def train_xgboost(X_train, y_train, X_test, y_test, known_attacks, seed=42):
    """
    Train XGBoost on known-attack training data.
    Use 'multi:softprob' to ensure predict_proba works reliably for uncertainty and SHAP.
    """
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=seed,
    )
    model.fit(X_train, y_train)
    y_proba_test = model.predict_proba(X_test)
    y_pred_test = np.argmax(y_proba_test, axis=1)
    print("Test Performance (Known Attacks):")
    print(classification_report(y_test, y_pred_test, target_names=known_attacks, zero_division=0))
    return model, y_pred_test


def train_rf(X_train, y_train, X_test, y_test, known_attacks, seed=42):
    model = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    print("Test Performance (Known Attacks):")
    print(classification_report(y_test, y_pred_test, target_names=known_attacks, zero_division=0))
    return model, y_pred_test


def _stack_shap_values(shap_values):
    """
    Convert SHAP outputs to a consistent (n_samples, n_features, n_classes) shape.
    Handles cases:
      - list of arrays -> stack into (n, f, c)
      - (c, n, f) -> transpose to (n, f, c)
      - (n, f) -> reshape to (n, f, 1)
    """
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=2)
    arr = np.array(shap_values)
    if arr.ndim == 2:
        n, f = arr.shape
        return arr.reshape(n, f, 1)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
        # (C, n, f) -> (n, f, C)
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def get_shap_explanations(model, X):
    """
    Compute SHAP values for the predicted class for each sample.
    Supports XGBoost and RandomForest multiclass models.
    """
    if isinstance(model, xgb.XGBClassifier):
        approximate = False  # exact for gradient boosting
    elif isinstance(model, RandomForestClassifier):
        approximate = True  # approximate speeds up forest SHAP
    else:
        approximate = True

    try:
        pred_classes = np.argmax(model.predict_proba(X), axis=1)
    except Exception:
        pred_classes = model.predict(X)

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X, approximate=approximate, check_additivity=False)
    shap_values = _stack_shap_values(shap_values_raw)  # -> (n, f, C)

    n, f, c = shap_values.shape
    idx = np.arange(n)
    result = shap_values[idx, :, pred_classes]  # (n, f)

    return result


def select_shap_by_threshold(shap_values, X_train=None, threshold=0.90, feature_names=None, path="shap_selection"):
    """Select the smallest set of top features whose cumulative mean |SHAP| covers `threshold`."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    sorted_idx = np.argsort(mean_abs)[::-1]
    sorted_shap = mean_abs[sorted_idx]

    cumulative = np.cumsum(sorted_shap)
    total = np.sum(sorted_shap)
    cumulative_pct = cumulative / total

    k = np.searchsorted(cumulative_pct, threshold) + 1
    top_idx = sorted_idx[:k]

    shap_selected = shap_values[:, top_idx]

    print(f"\n=== SHAP Feature Selection by Threshold ({threshold*100:.0f}%) ===")
    print(f"Number of features selected: {k}")
    print(f"Contribution covered: {cumulative_pct[k-1]*100:.2f}%")
    print("=" * 60)

    if feature_names is not None:
        for rank, idx in enumerate(top_idx, start=1):
            print(f"{rank:2d}. {feature_names[idx]:<30} | Mean|SHAP| = {mean_abs[idx]:.6f}")
    else:
        for rank, idx in enumerate(top_idx, start=1):
            print(f"{rank:2d}. Feature index {idx:<3d} | Mean|SHAP| = {mean_abs[idx]:.6f}")

    print("=" * 60)

    txt_path = path + ".txt"
    with open(txt_path, "w") as f:
        f.write(f"Selected SHAP Features (threshold={threshold*100:.0f}%)\n")
        f.write(f"Total features selected: {k}\n")
        f.write(f"Total contribution: {cumulative_pct[k-1]*100:.2f}%\n")
        f.write("=" * 80 + "\n\n")

        if feature_names is not None:
            for rank, idx in enumerate(top_idx, start=1):
                f.write(f"{rank:2d}. {feature_names[idx]:<30} | Mean|SHAP| = {mean_abs[idx]:.6f}\n")
        else:
            for rank, idx in enumerate(top_idx, start=1):
                f.write(f"{rank:2d}. Feature index {idx:<3d} | Mean|SHAP| = {mean_abs[idx]:.6f}\n")

    return shap_selected, top_idx


def compute_xgb_uncertainty(model, X, num_ensembles=20):
    """
    Compute predictive uncertainty for XGBoost using sub-ensemble sampling.
    Partitions boosted trees into groups and aggregates probabilities to
    decompose total predictive uncertainty into epistemic and aleatoric parts.
    """
    booster = model.get_booster()
    try:
        num_trees = booster.best_iteration + 1
    except Exception:
        num_trees = booster.num_boosted_rounds()

    num_classes = model.n_classes_

    tree_indices = np.array_split(np.arange(num_trees), num_ensembles)
    probas = []

    for trees in tree_indices:
        # iteration_range is [start, end), so add 1 to end
        margin = model.predict(X, iteration_range=(int(trees[0]), int(trees[-1]) + 1), output_margin=True)
        margin = margin.reshape(-1, num_classes)
        probas.append(softmax(margin, axis=1))

    probas = np.array(probas)             # (E, n, C)
    mean_proba = np.mean(probas, axis=0)  # (n, C)

    total_uncertainty = entropy(mean_proba.T, axis=0)
    aleatoric = np.mean(entropy(probas, axis=-1), axis=0)
    epistemic = total_uncertainty - aleatoric

    return total_uncertainty, epistemic, aleatoric


def compute_rf_uncertainty(model, X, num_ensembles=20):
    """Compute predictive uncertainty for Random Forest using sub-ensemble sampling."""
    trees = model.estimators_
    num_trees = len(trees)
    num_classes = model.n_classes_

    tree_indices = np.array_split(np.arange(num_trees), num_ensembles)
    probas = []

    for indices in tree_indices:
        sub_trees = [trees[int(i)] for i in indices]
        sub_probas = np.array([t.predict_proba(X) for t in sub_trees])  # (k, n, C)
        avg_proba = np.mean(sub_probas, axis=0)  # (n, C)
        probas.append(avg_proba)

    probas = np.array(probas)             # (E, n, C)
    mean_proba = np.mean(probas, axis=0)  # (n, C)

    total_uncertainty = entropy(mean_proba.T, axis=0)
    aleatoric = np.mean(entropy(probas, axis=-1), axis=0)
    epistemic = total_uncertainty - aleatoric

    return total_uncertainty, epistemic, aleatoric
