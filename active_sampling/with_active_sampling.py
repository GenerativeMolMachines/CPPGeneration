import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import joblib
import json
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
os.chdir(PROJECT_ROOT)

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2  # fraction of the full dataset reserved for validation
    target_precision: float = 0.95
    min_positive_preds_on_val: int = 5  # to avoid an "empty" threshold
    # Batch sampling quotas (fractions of B)
    core_set_ratio: float = 0.7
    hard_neg_ratio: float = 0.2
    n_clusters_per_class: int = 10
    hard_neg_top_quantile: float = 0.25  # among negatives, take top-q by proba as hard-neg candidates
    # Budgets 
    budgets: List[int] = None 
    # OOF
    oof_cv: int = 5
    oof_n_estimators: int = 400  # lightweight model for OOF estimates
    # Calibration
    calibr_cv: int = 5
    calibr_method: str = 'isotonic'

    def __post_init__(self):
        if self.budgets is None:
            self.budgets = [100, 200, 300, 400, 500, 600, 800, 1000]

# -----------------------------
# Helper functions
# -----------------------------
def load_rf_params(config_path: str = "rf_model_config.json") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "rf_params_final" not in cfg:
        raise KeyError(f"'rf_params_final' not found in {config_path}")

    return cfg["rf_params_final"]

def compute_oof_probas(X, y, base_rf_params: Dict, cv=5, n_estimators_override: Optional[int]=None, random_state=42) -> np.ndarray:
    """OOF probability estimates for the positive class on the training pool."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        rf_params = base_rf_params.copy()
        if n_estimators_override is not None:
            rf_params['n_estimators'] = n_estimators_override
        clf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=random_state + fold)
        clf.fit(X[tr_idx], y[tr_idx])
        oof_proba[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    return oof_proba

def compute_embedding_for_diversity(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Standardized embedding for diversification (Euclidean)."""
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    return Z, scaler

def k_center_greedy(embedding: np.ndarray, candidate_idx: np.ndarray, k: int) -> List[int]:
    """Farthest-first k-center selection of indices from candidate_idx using embedding (Euclidean)."""
    if k <= 0 or len(candidate_idx) == 0:
        return []
    if len(candidate_idx) <= k:
        return candidate_idx.tolist()

    C = candidate_idx
    E = embedding[C]

    # First center: farthest from the centroid (more stable than random)
    centroid = E.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(E - centroid, axis=1)
    first = np.argmax(dists)
    selected = [C[first]]

    # Initialize distances to the nearest center
    min_dist = pairwise_distances(E, E[[first]], metric='euclidean').reshape(-1)

    while len(selected) < k:
        next_idx = np.argmax(min_dist)
        selected.append(C[next_idx])
        # update min_dist
        new_d = pairwise_distances(E, E[[next_idx]], metric='euclidean').reshape(-1)
        min_dist = np.minimum(min_dist, new_d)

    return selected

def find_threshold_for_precision_max_f1(y_true, proba, target_precision, min_pos=1):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # thresholds has length len(precision)-1; align indices accordingly
    valid = np.where(precision[:-1] >= target_precision)[0]
    if len(valid) == 0:
        return None, {'precision': None, 'recall': None, 'f1': None, 'ap': average_precision_score(y_true, proba)}
    # select threshold with maximum F1 among valid ones
    best_idx, best_f1 = None, -1.0
    for i in valid:
        thr = thresholds[i]
        y_pred = (proba >= thr).astype(int)
        if y_pred.sum() < min_pos:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_idx = f1, i
    if best_idx is None:
        return None, {'precision': None, 'recall': None, 'f1': None, 'ap': average_precision_score(y_true, proba)}
    thr = thresholds[best_idx]
    y_pred = (proba >= thr).astype(int)
    return float(thr), {
        'precision': precision[best_idx],
        'recall': recall[best_idx],
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'ap': average_precision_score(y_true, proba)
    }

def select_subset_indices_v3(
    X_pool, y_pool, oof_proba, B, embedding,
    core_set_ratio=0.7,       
    hard_neg_ratio=0.2,       
    n_clusters_per_class=10,   # number of prototypes per class
    hard_neg_top_quantile=0.25,
    random_state=42
):
    n = len(y_pool)
    B_core = int(round(B * core_set_ratio))
    B_hneg = int(round(B * hard_neg_ratio))
    B_margin = max(0, B - B_core - B_hneg)

    selected = []
    
    # --- STAGE 1: FOUNDATION (Core-Set) ---
    # Split the core budget proportionally to class balance
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]
    
    pos_ratio = len(pos_idx) / n
    B_core_pos = int(round(B_core * pos_ratio))
    B_core_neg = max(0, B_core - B_core_pos)

    # Find prototypes for the positive class
    if B_core_pos > 0 and len(pos_idx) > n_clusters_per_class:
        kmeans_pos = KMeans(n_clusters=n_clusters_per_class, random_state=random_state, n_init='auto')
        clusters = kmeans_pos.fit_predict(embedding[pos_idx])
        distances = kmeans_pos.transform(embedding[pos_idx])
        
        # Find 1 closest point to each cluster center
        core_pos_indices = []
        for i in range(n_clusters_per_class):
            cluster_members = np.where(clusters == i)[0]
            if len(cluster_members) > 0:
                closest_point_idx = cluster_members[np.argmin(distances[cluster_members, i])]
                core_pos_indices.append(pos_idx[closest_point_idx])
        
        sel_core_pos = k_center_greedy(embedding, np.array(list(set(core_pos_indices))), B_core_pos)
        selected.extend(sel_core_pos)

    # Find prototypes for the negative class (same approach)
    if B_core_neg > 0 and len(neg_idx) > n_clusters_per_class:
        kmeans_neg = KMeans(n_clusters=n_clusters_per_class, random_state=random_state+1, n_init='auto')
        clusters = kmeans_neg.fit_predict(embedding[neg_idx])
        distances = kmeans_neg.transform(embedding[neg_idx])
        
        core_neg_indices = []
        for i in range(n_clusters_per_class):
            cluster_members = np.where(clusters == i)[0]
            if len(cluster_members) > 0:
                closest_point_idx = cluster_members[np.argmin(distances[cluster_members, i])]
                core_neg_indices.append(neg_idx[closest_point_idx])

        sel_core_neg = k_center_greedy(embedding, np.array(list(set(core_neg_indices))), B_core_neg)
        selected.extend(sel_core_neg)

    # --- STAGE 2: REFINEMENT ---
    already_selected = np.array(list(set(selected)), dtype=int)
    
    # Hard negatives
    if B_hneg > 0:
        neg_sorted_desc = neg_idx[np.argsort(-oof_proba[neg_idx])]
        k_hn = max(1, int(len(neg_idx) * hard_neg_top_quantile))
        hn_candidates = np.setdiff1d(neg_sorted_desc[:k_hn], already_selected, assume_unique=False)
        sel_hneg = k_center_greedy(embedding, hn_candidates, min(B_hneg, len(hn_candidates)))
        selected.extend(sel_hneg)
    
    # Margin samples
    already_selected = np.array(list(set(selected)), dtype=int)
    if B_margin > 0:
        margin = np.abs(oof_proba - 0.5)
        order = np.argsort(margin)
        k_cand = min(len(order), max(B_margin * 5, B_margin))  # take more candidates for diversification
        marg_candidates = np.setdiff1d(order[:k_cand], already_selected, assume_unique=False)
        sel_marg = k_center_greedy(embedding, marg_candidates, min(B_margin, len(marg_candidates)))
        selected.extend(sel_marg)

    # Final deduplication and fill-up if needed
    final_selected = list(dict.fromkeys(selected))
    if len(final_selected) < B:
        remaining = np.setdiff1d(np.arange(n), np.array(final_selected, dtype=int))
        extra = k_center_greedy(embedding, remaining, B - len(final_selected))
        final_selected.extend(extra)
        
    return np.array(final_selected[:B], dtype=int)

def fit_calibrated_rf(X_train: np.ndarray, y_train: np.ndarray, rf_params: Dict, cv: int = 5, method: str = 'isotonic', random_state: int = 42):
    rf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=random_state)
    clf = CalibratedClassifierCV(estimator=rf, method=method, cv=cv, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def save_classifier_artifacts(model, threshold, feature_names, path, meta=None):
    """
    Save classifier artifacts:
      - model: CalibratedClassifierCV (or any sklearn classifier)
      - threshold: float (decision threshold for probability binarization)
      - feature_names: list/np.array of feature names so that inference can select the right columns
      - meta: optional metadata (for traceability)
    """
    # Ensure feature_names_in_ is set (used by inference)
    if not hasattr(model, "feature_names_in_"):
        model.feature_names_in_ = np.array(feature_names, dtype=object)

    artifacts = {
        "model": model,
        "threshold": float(threshold),
    }
    if meta is not None:
        artifacts["meta"] = meta

    joblib.dump(artifacts, path)
    print(f"Artifacts saved to: {path}")

def stratified_train_val_test_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return pool/validation/test indices from a stratified split.
    Provides "global" indices relative to the original df_ready.
    """
    idx_all = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_size, stratify=y, random_state=random_state
    )
    y_trainval = y[idx_trainval]
    val_rel_size = val_size / (1.0 - test_size)
    idx_pool, idx_val = train_test_split(
        idx_trainval, test_size=val_rel_size, stratify=y_trainval, random_state=random_state
    )
    return idx_pool, idx_val, idx_test

# -----------------------------
# Main pipeline procedure
# -----------------------------
def run_active_sampling_pipeline_2(
    X: np.ndarray,
    y: np.ndarray,
    rf_params_final: Dict,
        config: Optional[Config] = None,
    feature_names: Optional[List[str]] = None,
    artifacts_path: Optional[str] = None,
):
    if config is None:
        config = Config()

    rs = config.random_state
    # 0) Index-based splits → then extract arrays
    idx_pool, idx_val, idx_test = stratified_train_val_test_indices(
        y, test_size=config.test_size, val_size=config.val_size, random_state=rs
    )
    X_pool, y_pool = X[idx_pool], y[idx_pool]
    X_val,  y_val  = X[idx_val],  y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    print(f"Shapes: pool={X_pool.shape}, val={X_val.shape}, test={X_test.shape}")

    # 1) OOF predictions on pool (teacher RF with a reduced number of trees)
    base_rf_params_for_oof = rf_params_final.copy()
    oof_proba = compute_oof_probas(
        X_pool, y_pool,
        base_rf_params_for_oof,
        cv=config.oof_cv,
        n_estimators_override=config.oof_n_estimators,
        random_state=rs
    )
    # 2) Embedding for diversification (Euclidean on standardized features)
    embedding, scaler = compute_embedding_for_diversity(X_pool)

    results = []
    best_solution = None

    for B in config.budgets:
        # 2a) Select subset indices of size B
        sel_idx = select_subset_indices_v3(
            X_pool, y_pool, oof_proba, B, embedding,
            core_set_ratio=config.core_set_ratio, 
            hard_neg_ratio=config.hard_neg_ratio,
            n_clusters_per_class=config.n_clusters_per_class,
            hard_neg_top_quantile=config.hard_neg_top_quantile,
            random_state=rs
        )
        X_sub, y_sub = X_pool[sel_idx], y_pool[sel_idx]

        # 3) Train calibrated final model on the subset
        clf = fit_calibrated_rf(
            X_sub, y_sub, rf_params_final, cv=config.calibr_cv,
            method=config.calibr_method, random_state=rs
        )

        # 4) Find threshold for target precision on validation set
        proba_val = clf.predict_proba(X_val)[:, 1]
        thr, val_metrics = find_threshold_for_precision_max_f1(
            y_val, proba_val, config.target_precision, min_pos=config.min_positive_preds_on_val
        )

        if thr is not None:
            # evaluate on test (not used for selection, logged only)
            proba_test = clf.predict_proba(X_test)[:, 1]
            y_pred_val = (proba_val >= thr).astype(int)
            y_pred_test = (proba_test >= thr).astype(int)
            test_prec = precision_score(y_test, y_pred_test, zero_division=0)
            test_rec = recall_score(y_test, y_pred_test, zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

            result = {
                'B': B,
                'threshold': thr,
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_AP': val_metrics['ap'],
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_f1': test_f1,
                'selected_indices': sel_idx,
                'selected_global_indices': idx_pool[sel_idx].tolist()
            }
            results.append(result)

            print(f"[B={B}] VAL: P={val_metrics['precision']:.3f}, R={val_metrics['recall']:.3f}, F1={val_metrics['f1']:.3f}, thr={thr:.4f} | "
                  f"TEST: P={test_prec:.3f}, R={test_rec:.3f}, F1={test_f1:.3f}")

            if artifacts_path is not None:
                # Note: clf was trained on np.ndarray → attach feature_names explicitly
                if feature_names is None:
                    # If feature names were not provided, try to get them from the model;
                    # if not available there either, raise a descriptive error
                    if not hasattr(clf, "feature_names_in_"):
                        raise ValueError("feature_names for saving are not set and are absent in the model. "
                                        "Pass feature_names to run_active_sampling_pipeline_2.")
                    fnames = list(clf.feature_names_in_)
                else:
                    fnames = list(feature_names)

                meta = {
                    "precision_target": config.target_precision,
                    "calibration": config.calibr_method,
                    "B": B,
                    "selector": "v3 core-set (kMeans)",
                    "val": {"precision": result['val_precision'], "recall": result['val_recall'],
                            "f1": result['val_f1'], "AP": result['val_AP']},
                    "test": {"precision": result['test_precision'], "recall": result['test_recall'],
                            "f1": result['test_f1']},
                    "selected_count": len(sel_idx),
                }
                # Save artifacts
                save_classifier_artifacts(
                    model=clf,
                    threshold=thr,
                    feature_names=fnames,
                    path=artifacts_path,
                    meta=meta
                )

            # Early stopping: target precision achieved on validation set
            if best_solution is None:
                best_solution = result
                print(f"--> Stop criterion met on validation at B={B}. Fixing the minimal subset.")
                break
        else:
            print(f"[B={B}] No threshold found for precision >= {config.target_precision:.2f} "
                  f"(or too few positive predictions). Increasing budget.")

    if best_solution is None and len(results) > 0:
        # if the criterion was never met, pick the best by validation precision
        best_solution = max(results, key=lambda r: r['val_precision'] if r['val_precision'] is not None else -1)
        print(f"Warning: target precision not reached on validation; "
              f"best by precision selected: B={best_solution['B']}.")

    return {
        'best': best_solution,
        'all_results': results
    }

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    rf_params_final = load_rf_params("config/rf_model_config.json")
    print("Loaded rf_params_final:", rf_params_final)

    cfg = Config(
        random_state=42,
        test_size=0.2,
        val_size=0.2,
        target_precision=0.90,
        min_positive_preds_on_val=5,
        core_set_ratio=0.7,  # 70% of budget for the "foundation"
        hard_neg_ratio=0.2,  # 20% for "hard negatives"
        # the remaining budget automatically goes to "margin samples" 
        n_clusters_per_class=10,
        hard_neg_top_quantile=0.25,
        budgets=[100, 200, 300, 400, 500, 600, 800, 1000],
        oof_cv=5,
        oof_n_estimators=400,   
        calibr_cv=5,
        calibr_method='isotonic',
    )

    # Run the pipeline
    df_ready = pd.read_csv("data/processed/peptides_for_classification_active_sampling.csv")
    columns_to_use = [
        'seq_length', 'molecular_weight', 'nh3_tail', 'po3_pos',
        'biotinylated', 'acylated_n_terminal', 'cyclic', 'amidated',
        'stearyl_uptake', 'hexahistidine_tagged', 'aromaticity',
        'instability_index', 'isoelectric_point', 'helix_fraction',
        'turn_fraction', 'sheet_fraction', 'molar_extinction_coefficient_reduced',
        'molar_extinction_coefficient_oxidized', 'gravy'
    ]
    artifacts_path = "models/classifier_artifacts_active_learning.joblib"
    X = df_ready[columns_to_use].to_numpy()
    y = df_ready['is_cpp'].to_numpy()
    results = run_active_sampling_pipeline_2(
        X, y,
        rf_params_final=rf_params_final,
        config=cfg,
        feature_names=columns_to_use,
        artifacts_path=artifacts_path
    )
    print("Best solution:", results['best'])

    active_config_path = Path("config/active_learning_config.json")

    with active_config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print(f"Active learning config saved to: {active_config_path.resolve()}")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    best = results['best']
    if best and 'selected_global_indices' in best:
        # 1) Get row ids from df_ready using global indices
        sel_global = np.unique(np.array(best['selected_global_indices'], dtype=int))
        # Verify that 'id' column is present and unique
        if 'id' not in df_ready.columns:
            raise KeyError("Column 'id' not found in df_ready. Cannot match rows.")
        assert df_ready['id'].is_unique, "Expected a unique 'id' column in df_ready."
        train_ids = df_ready.iloc[sel_global]['id'].values

        # 2) Load the original (unprocessed) dataset all.csv
        all_path = "data/raw/all_peptides.csv" 
        all_df = pd.read_csv(all_path)
        if 'id' not in all_df.columns:
            raise KeyError(f"Column 'id' not found in {all_path}. Cannot remove rows by id.")
        assert all_df['id'].is_unique, f"Expected a unique 'id' column in {all_path}."

        # 3) Align id types and build removal mask
        #    (important: types must match — strings vs. integers)
        train_ids = train_ids.astype(all_df['id'].dtype, copy=False)

        before = len(all_df)
        cleaned_df = all_df.loc[~all_df['id'].isin(train_ids)].copy()
        removed = before - len(cleaned_df)
        print(f"Rows to be removed: {removed} out of {before} (by id).")

        # 4) Save the cleaned dataset
        out_path = "data/active_sampling/peptides_without_classifier_train_rows.csv"
        cleaned_df.to_csv(out_path, index=False)
        print(f"Cleaned dataset saved: {out_path} | shape={cleaned_df.shape}")

    else:
        print("Warning: cannot remove train rows from all_peptides.csv (no selected_global_indices in results['best']).")
    pass