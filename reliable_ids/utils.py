from .config import DATASET, DATASET_PATH, NF_DATASET
from .models import train_rf, train_xgboost
import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, brier_score_loss, precision_recall_fscore_support
)
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
from sklearn.mixture import GaussianMixture
from efc import EnergyBasedFlowClassifier

pd.set_option("display.max_columns", None)


def process_CICIOT_23_data():
    """Read and preprocess the CICIoT_23 dataset."""
    path = DATASET_PATH
    csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    print(csv_files)
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"Read {csv_file} with shape {df.shape}!")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()
    df = change_label(df)
    print(f"CICIOT_23 combined dataset shape: {df.shape}")
    return df


def process_CICDDOS_2019_data():
    """
    Process CICDDOS-2019 parquet data into a clean dataframe that
    matches the structure and preprocessing of the other CIC datasets.
    """
    data_dir = DATASET_PATH  # Path from config (folder containing parquet files)
    files = os.listdir(data_dir)
    print(f"Found {len(files)} parquet files in {DATASET_PATH}")

    dfs = []
    for csv_file in files:
        df = pd.read_parquet(data_dir + csv_file)
        dfs.append(df)
        print(f"Read {csv_file} with shape {df.shape}!")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    print(f"CICDDOS_2019 combined dataset shape: {df.shape}")

    drop_columns = [
        "Flow ID",
        "Fwd Header Length.1",
        "Source IP", "Src IP",
        "Source Port", "Src Port",
        "Destination IP", "Dst IP",
        "Destination Port", "Dst Port",
        "Timestamp",
        "Unnamed: 0",
        "Inbound",
        "SimillarHTTP"
    ]

    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")

    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)

    label_mapping = {
        'DrDoS_UDP': 'UDP',
        'DrDoS_MSSQL': 'MSSQL',
        'DrDoS_LDAP': 'LDAP',
        'DrDoS_NetBIOS': 'NetBIOS',
        'UDP-lag': 'UDPLag',
        'Syn': 'Syn',
        'Benign': 'Benign',
        'DrDoS_DNS': 'DNS',
        'TFTP': 'TFTP',
        'Portmap': 'Portmap',
        'DrDoS_NTP': 'NTP',
        'WebDDoS': 'WebDDoS',
        'DrDoS_SNMP': 'SNMP'
    }
    df["label"] = df["label"].map(label_mapping).fillna(df["label"])
    return df


def preprocess_data(df):
    """Preprocess the dataset."""
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if NF_DATASET:
        df = df.drop([
            'IPV4_SRC_ADDR',
            'IPV4_DST_ADDR',
            'L4_SRC_PORT',
            'L4_DST_PORT',
            'Label'
        ], axis=1)
        df = df.rename(columns={'Attack': 'label'})

    # Drop benign to focus on attack detection as designed
    print(f"Original Label distribution:\n{df['label'].value_counts()}")
    df = df[df['label'] != 'Benign']

    # Remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def feature_selection(df, target_col='label'):
    """Perform simple feature filtering (variance + correlation)."""
    df_filtered = df.drop(target_col, axis=1)
    label = df[target_col]

    # Variance thresholding
    var_thr = VarianceThreshold(threshold=0.025)
    var_thr.fit(df_filtered)
    keep_cols = df_filtered.columns[var_thr.get_support()]
    df_var = df_filtered[keep_cols]

    # Winsorize extremely skewed continuous columns to reduce outliers' impact
    skewness = df_var.skew(numeric_only=True)
    hs_cols = skewness.index[abs(skewness) > 50].tolist()
    if len(hs_cols) > 0:
        df_wins = df_var[hs_cols].apply(lambda x: mstats.winsorize(x, limits=[0.05, 0.05]))
        df_var[hs_cols] = np.array(df_wins)

    # Remove highly correlated features
    df_reduced = CorrelationFilter(threshold=0.93).fit_transform(df_var)
    df_reduced[target_col] = label.values

    return df_reduced


def read_data():
    """Read and preprocess the dataset selected by config.DATASET."""
    if DATASET == "cicddos_2019":
        df = process_CICDDOS_2019_data()
    elif NF_DATASET:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = process_CICIOT_23_data()
    df = preprocess_data(df)
    df = feature_selection(df)
    print(f"Final dataset shape after preprocessing and feature selection: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def change_label(df):
    """Map fine-grained CICIoT-23 labels to coarser attack categories."""
    category_dict = {
        'DDoS-ACK_Fragmentation': 'DDoS',
        'DDoS-HTTP_Flood': 'DDoS',
        'DDoS-ICMP_Flood': 'DDoS',
        'DDoS-PSHACK_Flood': 'DDoS',
        'DDoS-RSTFINFlood': 'DDoS',
        'DDoS-SYN_Flood': 'DDoS',
        'DDoS-SlowLoris': 'DDoS',
        'DDoS-SynonymousIP_Flood': 'DDoS',
        'DDoS-TCP_Flood': 'DDoS',
        'DDoS-UDP_Flood': 'DDoS',
        'DDoS-UDP_Fragmentation': 'DDoS',
        'DDoS-ICMP_Fragmentation': 'DDoS',

        'DoS-HTTP_Flood': 'DoS',
        'DoS-SYN_Flood': 'DoS',
        'DoS-TCP_Flood': 'DoS',
        'DoS-UDP_Flood': 'DoS',

        'DictionaryBruteForce': 'BruteForce',

        'MITM-ArpSpoofing': 'Spoofing',
        'DNS_Spoofing': 'Spoofing',

        'Recon-HostDiscovery': 'Recon',
        'Recon-OSScan': 'Recon',
        'Recon-PingSweep': 'Recon',
        'Recon-PortScan': 'Recon',
        'VulnerabilityScan': 'Recon',

        'SqlInjection': 'Web',
        'CommandInjection': 'Web',
        'Backdoor_Malware': 'Web',
        'Uploading_Attack': 'Web',
        'XSS': 'Web',
        'BrowserHijacking': 'Web',

        'Mirai-greeth_flood': 'Mirai',
        'Mirai-greip_flood': 'Mirai',
        'Mirai-udpplain': 'Mirai',

        'BenignTraffic': 'Benign'
    }

    df = df.copy()
    df['label'] = df['label'].map(category_dict).fillna(df['label'])
    return df


def train_and_evaluate_model_with_val(known_attacks, dataset,
                                       val_size=0.1, test_size=0.2, random_state=None):
    """Split known classes into train / val / known-test, then add unknowns into the full test set."""
    data = dataset.copy()
    X = data.drop(columns=["label"])
    y = data["label"]

    mapping = {att: i for i, att in enumerate(known_attacks)}
    y_encoded = y.apply(lambda x: mapping.get(x, -1))

    known_mask = y.isin(known_attacks)
    X_known = X[known_mask].reset_index(drop=True)
    y_known = y_encoded[known_mask].reset_index(drop=True)
    X_unknown = X[~known_mask].reset_index(drop=True)
    y_unknown = y_encoded[~known_mask].reset_index(drop=True)

    hold_fraction = val_size + test_size
    X_train, X_hold, y_train, y_hold = train_test_split(
        X_known, y_known,
        test_size=hold_fraction,
        random_state=random_state,
        stratify=y_known
    )

    prop_val = val_size / (val_size + test_size)
    X_val, X_test_known, y_val, y_test_known = train_test_split(
        X_hold, y_hold,
        test_size=(1 - prop_val),
        random_state=random_state,
        stratify=y_hold
    )

    X_test = pd.concat([X_test_known, X_unknown], axis=0).reset_index(drop=True)
    y_test = pd.concat([y_test_known, y_unknown], axis=0).reset_index(drop=True)

    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_test_known, y_test_known


class CorrelationFilter:
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr_matrix = X.corr(numeric_only=True).abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.threshold)]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.to_drop_, errors='ignore')

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def compute_performance_stats(y_true, y_pred, best_percentile=95, y_prob=None):
    """Compute performance metrics for binary (known-vs-unknown) classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    ACC = (tp + tn) / max((tp + fp + tn + fn), 1)
    TPR_M = recall_score(y_true, y_pred, average='macro', zero_division=0)
    PREC_M = precision_score(y_true, y_pred, average='macro', zero_division=0)
    F1_M = f1_score(y_true, y_pred, average='macro', zero_division=0)
    FPR = fp / max((fp + tn), 1)

    metrics = {
        'ACCURACY': ACC,
        'RECALL_Macro': TPR_M,
        'PRECISION_Macro': PREC_M,
        'F1_Macro': F1_M,
        'FPR': FPR,
        'Percentile': best_percentile
    }

    if y_prob is not None:
        metrics['Brier'] = brier_score_loss(y_true, y_prob)

    print(classification_report(y_true, y_pred, zero_division=0))
    return pd.DataFrame(metrics, index=[0])


def compute_reconstruction_error(autoencoder, X, batch_size=128):
    """Compute per-sample autoencoder reconstruction error (MSE)."""
    errors = []
    num_samples = X.shape[0]
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)

    for i in range(0, num_samples, batch_size):
        X_batch = X_tf[i:i + batch_size]
        encoded = autoencoder.encoder(X_batch, training=False)
        decoded = autoencoder.decoder(encoded, training=False)
        mse_batch = tf.reduce_mean(tf.square(X_batch - decoded), axis=1)
        errors.append(mse_batch.numpy())

    return np.concatenate(errors)


def AE_anomaly_detection(autoencoder, train_data, val_data, shap_test, ground_truth_locs,
                          plt_title=None, use_autoencoder=True):
    """
    Quantile-based anomaly detection using a fixed percentile threshold.
    If use_autoencoder=False, treat values as direct scores (e.g., epistemic uncertainty)
    instead of computing an autoencoder reconstruction error.
    """
    if use_autoencoder:
        mse_val = compute_reconstruction_error(autoencoder, val_data)
        mse_test = compute_reconstruction_error(autoencoder, shap_test)
    else:
        mse_val = val_data
        mse_test = shap_test

    y_true = np.zeros(len(mse_test))
    y_true[ground_truth_locs] = 1

    auc_full = roc_auc_score(y_true, mse_test)
    ap = average_precision_score(y_true, mse_test)

    lo, hi = mse_val.min(), mse_val.max()
    probs = np.clip((mse_test - lo) / (hi - lo + 1e-12), 0, 1)

    percentiles = [95]
    all_stats = []

    for p in percentiles:
        threshold = np.quantile(mse_val, p / 100)
        y_pred = (mse_test > threshold).astype(int)
        stats = compute_performance_stats(y_true, y_pred, best_percentile=p, y_prob=probs)
        stats.insert(0, 'Title', f"{plt_title} - {p}th Percentile")
        all_stats.append(stats)

    results_df = pd.concat(all_stats, ignore_index=True)
    results_df["AUROC"] = auc_full
    results_df["AUPRC"] = ap

    return results_df


def select_best_gmm(X_val_s, max_components=6, cov_types=('full', 'diag', 'tied', 'spherical'),
                     reg_covar=1e-6, random_state=42, n_init=3):
    """
    Fit a small grid of GMMs and pick the one with the lowest BIC on validation-known data.
    Returns the best model and a small info dict.
    """
    best_model, best_bic, best_cfg = None, np.inf, None
    for cov in cov_types:
        for k in range(1, max_components + 1):
            try:
                gm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    reg_covar=reg_covar,
                    random_state=random_state,
                    n_init=n_init
                ).fit(X_val_s)
                bic = gm.bic(X_val_s)
                if bic < best_bic:
                    best_bic, best_model, best_cfg = bic, gm, (k, cov)
            except Exception:
                # Skip numerically unstable configs
                continue
    return best_model, {"bic": best_bic, "config": best_cfg}


def evaluate_base_models(data, attack_types):
    """
    Evaluate base classifiers (XGBoost, Random Forest, EFC) across all attack
    types (leave-one-out), evaluating performance on known classes only.
    """
    results = []

    for model_type in ['XGBoost', 'RandomForest', 'EFC']:
        print(f"\n=== Evaluating {model_type} Model ===")

        model_metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': []
        }

        for unknown_attack in attack_types:
            known_attacks = [a for a in attack_types if a != unknown_attack]

            X_train, y_train, X_val, y_val, X_test, y_test, X_test_known, y_test_known = \
                train_and_evaluate_model_with_val(known_attacks, data)

            if model_type == 'XGBoost':
                model, y_pred = train_xgboost(
                    X_train, y_train,
                    X_test_known, y_test_known,
                    known_attacks
                )

            elif model_type == 'RandomForest':
                model, y_pred = train_rf(
                    X_train, y_train,
                    X_test_known, y_test_known,
                    known_attacks
                )

            elif model_type == 'EFC':
                print("\n--- Running EFC Multiclass ---")

                efc = EnergyBasedFlowClassifier(cutoff_quantile=0.95)

                try:
                    efc.fit(X_train, y_train, categorical_columns=[])
                except TypeError:
                    efc.fit(X_train, y_train)

                y_pred = efc.predict(X_test_known)

                print("Test Performance (Known Attacks):")
                print(classification_report(
                    y_test_known,
                    y_pred,
                    target_names=known_attacks,
                    zero_division=0
                ))

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_known,
                y_pred,
                average='macro',
                labels=range(len(known_attacks))
            )
            accuracy = accuracy_score(y_test_known, y_pred)

            model_metrics['precision'].append(precision)
            model_metrics['recall'].append(recall)
            model_metrics['f1'].append(f1)
            model_metrics['accuracy'].append(accuracy)

            print(f"Unknown Attack (excluded): {unknown_attack}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  Accuracy:  {accuracy:.4f}\n")

        results.append({
            'model': model_type,
            'avg_precision': np.mean(model_metrics['precision']),
            'std_precision': np.std(model_metrics['precision']),
            'avg_recall': np.mean(model_metrics['recall']),
            'std_recall': np.std(model_metrics['recall']),
            'avg_f1': np.mean(model_metrics['f1']),
            'std_f1': np.std(model_metrics['f1']),
            'avg_accuracy': np.mean(model_metrics['accuracy']),
            'std_accuracy': np.std(model_metrics['accuracy']),
            'n_runs': len(attack_types)
        })

    results_df = pd.DataFrame(results)

    os.makedirs('./Results/Base', exist_ok=True)
    results_df.to_csv(f'./Results/Base/base_model_performance_{DATASET}.csv', index=False)

    return results_df
