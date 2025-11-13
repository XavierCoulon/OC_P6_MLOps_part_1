# ============================================================
# 🛠️ FONCTIONS UTILITAIRES POUR L'ÉVALUATION DES MODÈLES
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_overfitting_metrics(
    y_train, y_pred_train, y_proba_train, y_test, y_pred_test, y_proba_test
):
    """
    Calcule les métriques d'overfitting (différence train/test).

    Mesure le gap entre la performance sur le training set et le test set
    pour diagnostiquer l'overfitting ou l'underfitting.

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
    overfitting_metrics : dict
        Dictionnaire contenant les gaps train/test pour toutes les métriques
        Format : 'overfit_gap_*' (valeurs positives = overfitting)

    Notes
    -----
    - Valeur positive = overfitting (bon train, mauvais test)
    - Valeur négative = underfitting (mauvais train, bon test)
    - |gap| proche de 0 = bonne généralisation

    Examples
    --------
    >>> from utils import compute_overfitting_metrics
    >>> gap_metrics = compute_overfitting_metrics(
    ...     y_train, y_pred_train, y_proba_train,
    ...     y_test, y_pred_test, y_proba_test
    ... )
    """
    # Calculer les métriques pour train
    train_metrics = {
        "accuracy": accuracy_score(y_train, y_pred_train),
        "auc": roc_auc_score(y_train, y_proba_train),
        "recall_1": recall_score(y_train, y_pred_train, pos_label=1),
        "precision_1": precision_score(
            y_train, y_pred_train, pos_label=1, zero_division=0
        ),
        "f1_1": f1_score(y_train, y_pred_train, pos_label=1),
        "recall_0": recall_score(y_train, y_pred_train, pos_label=0),
        "precision_0": precision_score(
            y_train, y_pred_train, pos_label=0, zero_division=0
        ),
        "f1_0": f1_score(y_train, y_pred_train, pos_label=0),
    }

    # Calculer les métriques pour test
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "auc": roc_auc_score(y_test, y_proba_test),
        "recall_1": recall_score(y_test, y_pred_test, pos_label=1),
        "precision_1": precision_score(
            y_test, y_pred_test, pos_label=1, zero_division=0
        ),
        "f1_1": f1_score(y_test, y_pred_test, pos_label=1),
        "recall_0": recall_score(y_test, y_pred_test, pos_label=0),
        "precision_0": precision_score(
            y_test, y_pred_test, pos_label=0, zero_division=0
        ),
        "f1_0": f1_score(y_test, y_pred_test, pos_label=0),
    }

    # Calculer les gaps (train - test)
    # Valeur positive = overfitting, négative = underfitting
    overfitting_metrics = {
        f"overfit_gap_{k}": float(train_metrics[k] - test_metrics[k])
        for k in train_metrics.keys()
    }

    return overfitting_metrics


def compute_final_metrics(y_true, y_pred, y_proba):
    """
    Calcule les métriques finales standard pour tous les modèles.

    Parameters
    ----------
    y_true : array-like
        Valeurs vraies (vérité terrain)
    y_pred : array-like
        Prédictions du modèle
    y_proba : array-like
        Probabilités prédites (pour la classe positive)

    Returns
    -------
    metrics : dict
        Dictionnaire contenant toutes les métriques au format 'final_test_*'
    cm : ndarray
        Matrice de confusion
    rates : tuple
        Tuple contenant (false_positive_rate, false_negative_rate)

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=100, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> y_proba = model.predict_proba(X_test)[:, 1]
    >>> metrics, cm, (fpr, fnr) = compute_final_metrics(y_test, y_pred, y_proba)
    """
    metrics = {
        "final_test_accuracy": accuracy_score(y_true, y_pred),
        "final_test_auc": roc_auc_score(y_true, y_proba),
        # Classe 1 (Majorité)
        "final_test_recall_1": recall_score(y_true, y_pred, pos_label=1),
        "final_test_precision_1": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "final_test_f1_1": f1_score(y_true, y_pred, pos_label=1),
        # Classe 0 (Minorité)
        "final_test_recall_0": recall_score(y_true, y_pred, pos_label=0),
        "final_test_precision_0": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "final_test_f1_0": f1_score(y_true, y_pred, pos_label=0),
    }

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics["final_test_false_positive_rate"] = float(fp_rate)
    metrics["final_test_false_negative_rate"] = float(fn_rate)

    return metrics, cm, (fp_rate, fn_rate)


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
