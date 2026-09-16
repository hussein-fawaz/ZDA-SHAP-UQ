import os

import shap
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from datetime import date
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import random
import time
from .utils import read_data
from .config import ALL_ATTACKS, DATASET
from efc import EnergyBasedFlowClassifier
from .models import (
    select_shap_by_threshold,
    train_xgboost,
    train_rf,
    get_shap_explanations,
    compute_xgb_uncertainty,
    compute_rf_uncertainty
)
import tensorflow as tf
from .utils import (
    AE_anomaly_detection,
    compute_reconstruction_error,
    train_and_evaluate_model_with_val,
    compute_performance_stats,
    select_best_gmm
)
from .autoencoder import Autoencoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

today = date.today()


class ClusterZeroDayPipeline:
    """
    Zero-day detection pipeline for a single held-out ("unknown") attack class.

    Trains a classifier (XGBoost or RF) on the remaining ("known") attacks, then
    detects the unknown attack as anomalous by clustering (via a GMM) the joint
    space of [SHAP-space autoencoder reconstruction error, epistemic uncertainty].
    Several baselines (raw-feature autoencoder, EFC) are computed for comparison.
    """

    def __init__(self, unknown_attack, df, results_dir="./Results", seed=42, run_id=1):

        self.unknown_attack = unknown_attack
        self.known_attacks = [a for a in ALL_ATTACKS if a != unknown_attack]
        self.df = df
        self.seed = seed
        self.run_id = run_id

        self.results_dir = os.path.join(
            results_dir,
            today.strftime("%B%d"),
            DATASET,
            f"run_{self.run_id}",
            unknown_attack
        )
        os.makedirs(self.results_dir, exist_ok=True)

        self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_shap = StandardScaler()

    # ----------------------------------------------------------
    # DATA PREP (shared)
    # ----------------------------------------------------------
    def prepare_data(self):

        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            self.X_test_known,
            self.y_test_known
        ) = train_and_evaluate_model_with_val(self.known_attacks, self.df, random_state=self.seed)

        self.X_train_s = self.scaler_x.fit_transform(self.X_train)
        self.X_val_s = self.scaler_x.transform(self.X_val)
        self.X_test_s = self.scaler_x.transform(self.X_test)

    # ----------------------------------------------------------
    # SHARED TRAINING BASELINE (raw-feature autoencoder)
    # ----------------------------------------------------------
    def compute_training_baseline(self):

        ground_truth_locs = np.where(self.y_test == -1)[0]

        self.autoenc_normal = Autoencoder(self.X_train_s.shape[1])
        self.autoenc_normal.train(self.X_train_s, self.X_val_s)

        res_train = AE_anomaly_detection(
            self.autoenc_normal,
            self.X_train_s, self.X_val_s, self.X_test_s,
            ground_truth_locs,
            plt_title="Training"
        )

        res_train.insert(0, "Model", "Shared")
        if "Title" in res_train.columns:
            res_train["Title"] = "Training"
        else:
            res_train.insert(0, "Title", "Training")

        return res_train

    # ----------------------------------------------------------
    # EFC BASELINE (binary: known vs. unknown)
    # ----------------------------------------------------------
    def compute_efc_baseline(self):

        print("\n--- Running EFC Baseline ---")

        y_true = (self.y_test == -1).astype(int)

        # Train on known data only
        y_train_efc = np.zeros(len(self.y_train))

        efc = EnergyBasedFlowClassifier(n_bins=10, cutoff_quantile=0.99, pseudocounts=0.1)

        X_train_efc = self.X_train_s
        X_val_efc = self.X_val_s
        X_test_efc = self.X_test_s

        try:
            efc.fit(X_train_efc, y_train_efc, categorical_columns=[])
        except TypeError:
            efc.fit(X_train_efc, y_train_efc)

        _, val_energy = efc.predict(X_val_efc, return_energies=True)
        _, test_energy = efc.predict(X_test_efc, return_energies=True)

        # Threshold from validation only, to avoid leakage
        threshold = np.quantile(val_energy, 0.95)
        y_pred = (test_energy > threshold).astype(int)

        # Normalize scores for the Brier score
        lo, hi = val_energy.min(), val_energy.max()
        probs = np.clip((test_energy - lo) / (hi - lo + 1e-12), 0, 1)

        stats = compute_performance_stats(
            y_true,
            y_pred,
            best_percentile=95,
            y_prob=probs
        )

        stats["AUROC"] = roc_auc_score(y_true, test_energy)
        stats["AUPRC"] = average_precision_score(y_true, test_energy)

        stats.insert(0, "Model", "efc")
        stats.insert(0, "Title", "EFC")
        return stats

    # ----------------------------------------------------------
    # CORE APPROACH: SHAP + epistemic uncertainty + AE + GMM
    # ----------------------------------------------------------
    def run_model_branch(self, model_type):

        print(f"\n--- Running {model_type.upper()} ---")

        ground_truth_locs = np.where(self.y_test == -1)[0]

        # ---------------- TRAIN MODEL ----------------
        if model_type == "xgb":
            model, _ = train_xgboost(
                self.X_train, self.y_train,
                self.X_test_known, self.y_test_known,
                self.known_attacks,
                seed=self.seed
            )
        else:
            model, _ = train_rf(
                self.X_train, self.y_train,
                self.X_test_known, self.y_test_known,
                self.known_attacks,
                seed=self.seed
            )

        # ---------------- SHAP ----------------
        shap_train = get_shap_explanations(model, self.X_train)
        shap_val = get_shap_explanations(model, self.X_val)
        shap_test = get_shap_explanations(model, self.X_test)

        shap_train, top_idx = select_shap_by_threshold(
            shap_train,
            self.X_train,
            threshold=0.90,
            feature_names=None,
            path=os.path.join(self.results_dir, f"{model_type}_shap_selection")
        )

        shap_val = shap_val[:, top_idx]
        shap_test = shap_test[:, top_idx]

        shap_train = self.scaler_shap.fit_transform(shap_train)
        shap_val = self.scaler_shap.transform(shap_val)
        shap_test = self.scaler_shap.transform(shap_test)

        # ---------------- UNCERTAINTY ----------------
        if model_type == "xgb":
            _, epi_train, _ = compute_xgb_uncertainty(model, self.X_train)
            _, epi_val, _ = compute_xgb_uncertainty(model, self.X_val)
            _, epi_test, _ = compute_xgb_uncertainty(model, self.X_test)
        else:
            _, epi_train, _ = compute_rf_uncertainty(model, self.X_train)
            _, epi_val, _ = compute_rf_uncertainty(model, self.X_val)
            _, epi_test, _ = compute_rf_uncertainty(model, self.X_test)

        epi_train = epi_train.reshape(-1, 1)
        epi_val = epi_val.reshape(-1, 1)
        epi_test = epi_test.reshape(-1, 1)

        # ---------------- AUTOENCODERS ----------------
        autoenc_shap = Autoencoder(shap_train.shape[1])
        autoenc_shap.train(shap_train, shap_val)

        shap_ep_train = np.concatenate((shap_train, epi_train), axis=1)
        shap_ep_val = np.concatenate((shap_val, epi_val), axis=1)
        shap_ep_test = np.concatenate((shap_test, epi_test), axis=1)

        autoenc_shap_epi = Autoencoder(shap_ep_train.shape[1])
        autoenc_shap_epi.train(shap_ep_train, shap_ep_val)

        results = []

        def record(df, title):
            df = df.copy()
            df.insert(0, "Model", model_type)
            if "Title" in df.columns:
                df["Title"] = title
            else:
                df.insert(0, "Title", title)
            return df

        # ---------------- AE SHAP ----------------
        res_shap = AE_anomaly_detection(
            autoenc_shap,
            shap_train, shap_val, shap_test,
            ground_truth_locs,
            plt_title="SHAP"
        )
        results.append(record(res_shap, "SHAP"))

        # ---------------- Epistemic ----------------
        res_epi = AE_anomaly_detection(
            None,
            epi_train, epi_val, epi_test,
            ground_truth_locs,
            plt_title="Epistemic",
            use_autoencoder=False
        )
        results.append(record(res_epi, "Epistemic"))

        # ---------------- SHAP + Epistemic ----------------
        res_shap_epi = AE_anomaly_detection(
            autoenc_shap_epi,
            shap_ep_train, shap_ep_val, shap_ep_test,
            ground_truth_locs,
            plt_title="SHAP+Epi"
        )
        results.append(record(res_shap_epi, "SHAP+Epi"))

        # ---------------- GMM (main approach) ----------------
        mse_val = compute_reconstruction_error(autoenc_shap, shap_val)
        mse_test = compute_reconstruction_error(autoenc_shap, shap_test)

        X_val_2d = np.hstack([mse_val.reshape(-1, 1), epi_val])
        X_test_2d = np.hstack([mse_test.reshape(-1, 1), epi_test])

        scaler_gmm = StandardScaler()
        X_val_s = scaler_gmm.fit_transform(X_val_2d)
        X_test_s = scaler_gmm.transform(X_test_2d)

        gmm, _ = select_best_gmm(X_val_s, random_state=self.seed)

        dens_val = gmm.score_samples(X_val_s)
        dens_test = gmm.score_samples(X_test_s)

        threshold = np.percentile(dens_val, 5)

        y_true = (self.y_test == -1).astype(int)
        y_pred = (dens_test < threshold).astype(int)

        # ---------------- FULL ROC ----------------
        fpr, tpr, thresholds = roc_curve(y_true, -dens_test)

        roc_df = pd.DataFrame({
            "FPR": fpr,
            "TPR": tpr,
            "Threshold": thresholds
        })

        roc_df.to_csv(os.path.join(
            self.results_dir,
            f"{self.unknown_attack}_{model_type}_gmm_roc.csv"
        ), index=False)

        # ---------------- GMM RESULT ----------------
        stats = compute_performance_stats(y_true, y_pred)
        stats["AUROC"] = roc_auc_score(y_true, -dens_test)
        stats.insert(0, "Model", model_type)
        stats.insert(0, "Title", "GMM-5")
        results.append(stats)

        # ---------------- SENSITIVITY (threshold percentile 1-15) ----------------
        sens_results = []
        for p in range(1, 16):
            thr = np.percentile(dens_val, p)
            y_pred = (dens_test < thr).astype(int)

            stats = compute_performance_stats(y_true, y_pred)
            stats["Percentile"] = p
            stats["Model"] = model_type
            sens_results.append(stats)

        sens_df = pd.concat(sens_results, ignore_index=True)

        sens_df.to_csv(os.path.join(
            self.results_dir,
            f"{self.unknown_attack}_{model_type}_gmm_sensitivity.csv"
        ), index=False)

        # ==========================================================
        # DEPLOYMENT LATENCY (full pipeline + per-component breakdown)
        # ==========================================================
        n_samples = min(2000, len(self.X_test))
        X_bench = self.X_test.iloc[:n_samples] if hasattr(self.X_test, "iloc") else self.X_test[:n_samples]

        lat_model = []
        lat_epi = []
        lat_shap = []
        lat_ae = []
        lat_gmm = []
        lat_total = []
        explainer = shap.TreeExplainer(model)

        for i in range(len(X_bench)):

            x = X_bench.iloc[i:i + 1] if hasattr(X_bench, "iloc") else X_bench[i:i + 1]

            start_total = time.perf_counter()

            # ---------------- MODEL ----------------
            t0 = time.perf_counter()
            proba = model.predict_proba(x)
            t1 = time.perf_counter()
            lat_model.append(t1 - t0)
            pred_class = np.argmax(proba, axis=1)[0]

            # ---------------- EPISTEMIC ----------------
            t0 = time.perf_counter()
            if model_type == "xgb":
                _, epi, _ = compute_xgb_uncertainty(model, x, num_ensembles=1)
            else:
                _, epi, _ = compute_rf_uncertainty(model, x, num_ensembles=1)
            t1 = time.perf_counter()
            lat_epi.append(t1 - t0)

            epi = epi.reshape(-1, 1)

            # ---------------- SHAP ----------------
            t0 = time.perf_counter()

            try:
                shap_raw = explainer.shap_values(x, check_additivity=False)
            except TypeError:
                shap_raw = explainer.shap_values(x)

            # Handle all SHAP output formats
            if isinstance(shap_raw, list):
                # list of (1, f) -> pick predicted class
                shap_vals = shap_raw[pred_class]
            elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
                # (1, f, C) -> pick predicted class
                shap_vals = shap_raw[:, :, pred_class]
            else:
                # already (1, f)
                shap_vals = shap_raw

            shap_vals = shap_vals.reshape(1, -1)

            shap_vals = shap_vals[:, top_idx]
            shap_vals = self.scaler_shap.transform(shap_vals)

            t1 = time.perf_counter()
            lat_shap.append(t1 - t0)

            # ---------------- AE ----------------
            t0 = time.perf_counter()
            mse = compute_reconstruction_error(autoenc_shap, shap_vals)
            t1 = time.perf_counter()
            lat_ae.append(t1 - t0)

            # ---------------- GMM ----------------
            t0 = time.perf_counter()
            X_gmm = np.hstack([mse.reshape(-1, 1), epi])
            X_gmm = scaler_gmm.transform(X_gmm)
            _ = gmm.score_samples(X_gmm)
            t1 = time.perf_counter()
            lat_gmm.append(t1 - t0)

            end_total = time.perf_counter()
            lat_total.append(end_total - start_total)

        mean_latency = np.mean(lat_total)
        latency_df = pd.DataFrame({
            "Model": [model_type],

            "Total_mean_ms": [mean_latency * 1000],
            "Total_std_ms": [np.std(lat_total) * 1000],
            "Total_p95_ms": [np.percentile(lat_total, 95) * 1000],

            "Model_ms": [np.mean(lat_model) * 1000],
            "Epistemic_ms": [np.mean(lat_epi) * 1000],
            "SHAP_ms": [np.mean(lat_shap) * 1000],
            "AE_ms": [np.mean(lat_ae) * 1000],
            "GMM_ms": [np.mean(lat_gmm) * 1000],

            "Throughput_flows_per_sec": [1 / mean_latency if mean_latency > 0 else 0]
        })

        latency_path = os.path.join(
            self.results_dir,
            f"{self.unknown_attack}_{model_type}_latency.csv"
        )
        latency_df.to_csv(latency_path, index=False)

        return pd.concat(results, ignore_index=True)

    # ----------------------------------------------------------
    def run_pipeline(self):
        final_path = os.path.join(self.results_dir, f"{self.unknown_attack}_results.csv")
        if os.path.exists(final_path):
            print(f"Skipping {self.unknown_attack} (already completed)")
            return

        print(f"\n==== Running for {self.unknown_attack} ====")

        self.prepare_data()

        all_results = []

        # Shared baselines
        all_results.append(self.compute_training_baseline())
        all_results.append(self.compute_efc_baseline())

        # Core approach, for both classifier backends
        for model_type in ["xgb", "rf"]:
            all_results.append(self.run_model_branch(model_type))

        final_df = pd.concat(all_results, ignore_index=True)

        final_df.to_csv(
            os.path.join(self.results_dir, f"{self.unknown_attack}_results.csv"),
            index=False
        )

        print("Done.")


def main(K=5):
    """
    Reproduce the paper's results: repeat the full leave-one-out sweep (every
    attack type held out once, both XGBoost and RF, all baselines) K times with
    different seeds, then aggregate mean/std across runs.

    Each repetition re-seeds Python's `random`, NumPy, and TensorFlow with
    seed = 42 + k so results are reproducible run-to-run.
    """
    start_time = time.time()

    print("\n==============================")
    print(f"Starting {K} repeated experiments")
    print("==============================\n")

    print("Loading dataset...")
    df = read_data()
    print("Dataset loaded.\n")

    all_runs = []

    for k in range(K):

        print("\n====================================")
        print(f"RUN {k+1}/{K}")
        print("====================================\n")

        seed = 42 + k
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        run_results = []

        for unknown_attack in ALL_ATTACKS:

            print("\n----------------------------------------")
            print(f"Running for unknown attack: {unknown_attack}")
            print("----------------------------------------")

            pipeline = ClusterZeroDayPipeline(
                unknown_attack=unknown_attack,
                df=df,
                seed=seed,
                run_id=k + 1,
                results_dir="./Results"
            )

            pipeline.run_pipeline()

            result_path = os.path.join(
                pipeline.results_dir,
                f"{unknown_attack}_results.csv"
            )

            df_res = pd.read_csv(result_path)
            df_res["Run"] = k + 1
            df_res["Unknown_Attack"] = unknown_attack

            run_results.append(df_res)

        run_df = pd.concat(run_results, ignore_index=True)
        all_runs.append(run_df)

    full_df = pd.concat(all_runs, ignore_index=True)

    group_cols = ["Unknown_Attack", "Model", "Title"]
    metric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in metric_cols if c != "Run"]

    agg_df = (
        full_df
        .groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg_df.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]

    agg_dir = os.path.join(
        "./Results",
        today.strftime("%B%d"),
        DATASET,
        "aggregated"
    )

    os.makedirs(agg_dir, exist_ok=True)

    full_df.to_csv(os.path.join(agg_dir, "all_runs.csv"), index=False)
    agg_df.to_csv(os.path.join(agg_dir, "mean_std_results.csv"), index=False)

    total_time = (time.time() - start_time) / 60

    print("\n==============================")
    print("All runs completed")
    print("==============================")
    print(f"Total time: {total_time:.2f} minutes\n")


if __name__ == "__main__":
    main()
