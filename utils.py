# ============================================================
# 🛠️ FONCTIONS UTILITAIRES POUR L'ÉVALUATION DES MODÈLES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0):
    """
    Détecte les outliers pour toutes les colonnes numériques et fournit un rapport détaillé.

    Returns:
        outlier_rows : DataFrame des lignes avec au moins un outlier
        outlier_details : DataFrame avec colonnes numériques, booléen outlier et Z-score
        summary : dict avec nombre et % d'outliers par colonne
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Calcul Z-score
    z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    z_scores_abs = z_scores.abs()

    # Masque des outliers
    outlier_mask = z_scores_abs > threshold

    # Lignes avec au moins un outlier
    outlier_rows = df[outlier_mask.any(axis=1)].copy()

    # DataFrame avec booléen + Z-score
    outlier_details = pd.DataFrame(index=outlier_rows.index)
    for col in numeric_cols:
        outlier_details[col + "_outlier"] = outlier_mask.loc[outlier_rows.index, col]
        outlier_details[col + "_zscore"] = z_scores.loc[outlier_rows.index, col]

    # Résumé par colonne
    summary = {}
    for col in numeric_cols:
        count = outlier_mask[col].sum()
        pct = (count / len(df)) * 100
        summary[col] = {"count": count, "percent": pct}

    return outlier_rows, outlier_details, summary


def extract_cv_metrics(cv_results):
    """
    Extrait les métriques de Cross-Validation dans le format standard.

    Transforme les résultats bruts de cross_validate() en trois dictionnaires
    avec des noms normalisés :
    - CV_mean_* : moyenne des scores de test sur les folds
    - CV_std_* : écart-type des scores de test sur les folds
    - Train_mean_* : moyenne des scores de train sur les folds (diagnostic biais/variance)

    Parameters
    ----------
    cv_results : dict
        Résultats bruts de sklearn.model_selection.cross_validate()

    Returns
    -------
    cv_mean : dict
        Moyennes des scores de test (CV)
    cv_std : dict
        Écarts-types des scores de test (CV)
    train_mean : dict
        Moyennes des scores de train (diagnostic)

    Examples
    --------
    >>> from sklearn.model_selection import cross_validate, StratifiedKFold
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from config import GLOBAL_SCORING, CV_STRATEGY
    >>> X, y = make_classification(n_samples=100, random_state=42)
    >>> model = LogisticRegression()
    >>> cv_results = cross_validate(
    ...     model, X, y, cv=CV_STRATEGY, scoring=GLOBAL_SCORING, return_train_score=True
    ... )
    >>> cv_mean, cv_std, train_mean = extract_cv_metrics(cv_results)
    """
    cv_mean = {
        (
            f"CV_mean_{k.split('_')[-1]}_{k.split('_')[1]}"
            if len(k.split("_")) > 2
            else f"CV_mean_{k.split('_')[-1]}"
        ): v.mean()
        for k, v in cv_results.items()
        if k.startswith("test_")
    }
    cv_std = {
        (
            f"CV_std_{k.split('_')[-1]}_{k.split('_')[1]}"
            if len(k.split("_")) > 2
            else f"CV_std_{k.split('_')[-1]}"
        ): v.std()
        for k, v in cv_results.items()
        if k.startswith("test_")
    }
    train_mean = {
        (
            f"Train_mean_{k.split('_')[-1]}_{k.split('_')[1]}"
            if len(k.split("_")) > 2
            else f"Train_mean_{k.split('_')[-1]}"
        ): v.mean()
        for k, v in cv_results.items()
        if k.startswith("train_")
    }
    return cv_mean, cv_std, train_mean


def compute_train_test_metrics(
    y_train, y_pred_train, y_proba_train, y_test, y_pred_test, y_proba_test
):
    """
    Calcule les métriques sur train ET test, ainsi que l'overfitting gap.

    Cette fonction combine les métriques de train et test avec les gaps
    d'overfitting pour une vue complète de la généralisation du modèle.

    Parameters
    ----------
    y_train : array-like
        Valeurs vraies d'entraînement
    y_pred_train : array-like
        Prédictions sur le training set
    y_proba_train : array-like
        Probabilités prédites sur le training set
    y_test : array-like
        Valeurs vraies de test
    y_pred_test : array-like
        Prédictions sur le test set
    y_proba_test : array-like
        Probabilités prédites sur le test set

    Returns
    -------
    all_metrics : dict
        Dictionnaire contenant :
        - Métriques train : préfixe 'train_*'
        - Métriques test : préfixe 'final_test_*'
        - Gaps d'overfitting : préfixe 'overfit_gap_*'

    Examples
    --------
    >>> from utils import compute_train_test_metrics
    >>> all_metrics = compute_train_test_metrics(
    ...     y_train, y_pred_train, y_proba_train,
    ...     y_test, y_pred_test, y_proba_test
    ... )
    >>> print(f"Accuracy train: {all_metrics['train_accuracy']:.3f}")
    >>> print(f"Accuracy test: {all_metrics['final_test_accuracy']:.3f}")
    >>> print(f"Overfitting gap: {all_metrics['overfit_gap_accuracy']:.3f}")
    """
    # Métriques de train
    train_metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_auc": roc_auc_score(y_train, y_proba_train),
        "train_auprc": average_precision_score(y_train, y_proba_train),
        "train_recall_1": recall_score(y_train, y_pred_train, pos_label=1),
        "train_precision_1": precision_score(
            y_train, y_pred_train, pos_label=1, zero_division=0
        ),
        "train_f1_1": f1_score(y_train, y_pred_train, pos_label=1),
        "train_recall_0": recall_score(y_train, y_pred_train, pos_label=0),
        "train_precision_0": precision_score(
            y_train, y_pred_train, pos_label=0, zero_division=0
        ),
        "train_f1_0": f1_score(y_train, y_pred_train, pos_label=0),
    }

    # Métriques de test
    test_metrics = {
        "final_test_accuracy": accuracy_score(y_test, y_pred_test),
        "final_test_auc": roc_auc_score(y_test, y_proba_test),
        "final_test_auprc": average_precision_score(y_test, y_proba_test),
        "final_test_recall_1": recall_score(y_test, y_pred_test, pos_label=1),
        "final_test_precision_1": precision_score(
            y_test, y_pred_test, pos_label=1, zero_division=0
        ),
        "final_test_f1_1": f1_score(y_test, y_pred_test, pos_label=1),
        "final_test_recall_0": recall_score(y_test, y_pred_test, pos_label=0),
        "final_test_precision_0": precision_score(
            y_test, y_pred_test, pos_label=0, zero_division=0
        ),
        "final_test_f1_0": f1_score(y_test, y_pred_test, pos_label=0),
    }

    # Matrice de confusion et taux d'erreur (test)
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    test_metrics["final_test_false_positive_rate"] = float(fp_rate)
    test_metrics["final_test_false_negative_rate"] = float(fn_rate)

    # Gaps d'overfitting (train - test)
    overfitting_metrics = {
        "overfit_gap_accuracy": float(
            train_metrics["train_accuracy"] - test_metrics["final_test_accuracy"]
        ),
        "overfit_gap_auc": float(
            train_metrics["train_auc"] - test_metrics["final_test_auc"]
        ),
        "overfit_gap_recall_1": float(
            train_metrics["train_recall_1"] - test_metrics["final_test_recall_1"]
        ),
        "overfit_gap_precision_1": float(
            train_metrics["train_precision_1"] - test_metrics["final_test_precision_1"]
        ),
        "overfit_gap_f1_1": float(
            train_metrics["train_f1_1"] - test_metrics["final_test_f1_1"]
        ),
        "overfit_gap_recall_0": float(
            train_metrics["train_recall_0"] - test_metrics["final_test_recall_0"]
        ),
        "overfit_gap_precision_0": float(
            train_metrics["train_precision_0"] - test_metrics["final_test_precision_0"]
        ),
        "overfit_gap_f1_0": float(
            train_metrics["train_f1_0"] - test_metrics["final_test_f1_0"]
        ),
    }

    # Combiner tous les metrics
    all_metrics = {**train_metrics, **test_metrics, **overfitting_metrics}

    return all_metrics, cm, (fp_rate, fn_rate)
