# 🏉 Projet MLOps - Prédiction de Tirs au Rugby

## 📋 Vue d'ensemble

Ce projet implémente un pipeline MLOps complet pour prédire la réussite des tirs au rugby en utilisant différents modèles de machine learning. Le projet utilise **MLflow** pour le suivi des expériences et des artefacts, et **SHAP** pour l'interprétabilité des modèles.

### Objectifs

-   Comparer plusieurs modèles de classification
-   Optimiser les hyperparamètres avec GridSearchCV
-   Appliquer des techniques de rééquilibrage (SMOTE)
-   Fournir des explications via SHAP
-   Tracer tous les modèles et métriques dans MLflow

---

## 🚀 Installation

### Prérequis

-   Python 3.8+
-   `uv` (gestionnaire de paquets ultra-rapide)

### Installation avec `uv`

1. **Cloner le projet**

```bash
git clone <repo-url>
cd OC_P6_Rugby_MLOps
```

2. **Installer les dépendances avec uv**

```bash
uv pip install -r requirements.txt
```

Ou directement avec uv:

```bash
uv sync
```

3. **Vérifier l'installation**

```bash
python --version
uv pip list | grep -E "mlflow|shap|scikit-learn"
```

### Dépendances principales

-   **mlflow**: Suivi des expériences et versioning des modèles
-   **scikit-learn**: Modèles et métriques
-   **xgboost**: Gradient boosting
-   **shap**: Interprétabilité des modèles
-   **imbalanced-learn**: SMOTE pour rééquilibrage
-   **pandas, numpy**: Manipulation de données
-   **matplotlib, seaborn**: Visualisations
-   **rich**: Affichage formaté en terminal

---

## 📁 Structure du projet

```
OC_P6_Rugby_MLOps/
├── README.md                          # Ce fichier
├── config.py                          # Configuration globale (paths, constantes)
├── utils.py                           # Fonctions utilitaires (métriques, etc.)
├── main.py                            # Script principal (optionnel)
│
├── data/
│   ├── raw/                           # Données brutes
│   ├── interim/                       # Données nettoyées
│   └── processed/                     # Données prêtes pour le modèle
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_features.ipynb              # Feature Engineering
│   ├── 03_modeling.ipynb              # Benchmark de 5 modèles
│   ├── 04_xgboost_finetuning.ipynb    # Fine-tuning XGBoost + SHAP
│   ├── 05_lr_finetuning.ipynb         # Fine-tuning LogisticRegression + SHAP
│   └── 06_lr_final_model.ipynb        # Modèle final LR avec Feature Engineering + SHAP
│
├── outputs/
│   ├── figures/                       # Graphiques générés
│   └── reports/                       # Rapports d'analyse
│
└── mlruns/                            # Artefacts MLflow (généré automatiquement)
```

---

## 📓 Notebooks expliqués

### 1️⃣ **01_eda.ipynb** - Exploratory Data Analysis

**Objectif**: Comprendre les données brutes

-   Analyse statistique descriptive
-   Distribution des classes
-   Visualisation des features
-   Détection des valeurs manquantes
-   Corrélations entre features

**Résultat**: Dataset compris et prêt pour feature engineering

---

### 2️⃣ **02_features.ipynb** - Feature Engineering

**Objectif**: Créer et transformer les features

-   Création de `difficulty_score` = distance × angle
-   Création de `foot_side_match` = correspondance pied/côté
-   Analyse de corrélation
-   Sélection des features pertinentes

**Résultat**: Dataset enrichi `kicks_ready_for_model.csv`

---

### 3️⃣ **03_modeling.ipynb** - Benchmark de modèles

**Objectif**: Comparer 5 modèles baseline

Modèles testés:

1. **DummyClassifier** - Baseline
2. **LogisticRegression** - Modèle linéaire
3. **RandomForest** - Ensemble basé arbres
4. **SVM** - Support Vector Machine
5. **XGBoost** - Gradient boosting

Pour chaque modèle:

-   Entraînement avec cross-validation
-   Matrice de confusion
-   Courbes ROC et Precision-Recall
-   Feature importances
-   Logging dans MLflow

**Résultat**: Meilleur modèle identifié (généralement XGBoost)

---

### 4️⃣ **04_xgboost_finetuning.ipynb** - Fine-tuning XGBoost

**Objectif**: Optimiser les hyperparamètres d'XGBoost

Étapes:

1. Preprocessing (StandardScaler)
2. **SMOTE** pour rééquilibrer les classes
3. GridSearchCV avec 16 combinaisons (grille réduite)
4. Entraînement du meilleur modèle
5. **SHAP Analysis** complète:
    - Summary plots (Bar + Bee swarm)
    - Dependence plots
    - Force plots (explication par prédiction)
    - Waterfall plots

**Hyperparamètres optimisés**:

-   n_estimators, max_depth, learning_rate
-   subsample, colsample_bytree, min_child_weight

**Résultat**: Modèle XGBoost optimisé avec explications SHAP

---

### 5️⃣ **05_lr_finetuning.ipynb** - Fine-tuning LogisticRegression

**Objectif**: Optimiser LogisticRegression avec SMOTE

Étapes similaires à NB04:

1. Preprocessing
2. SMOTE
3. GridSearchCV (24 combinaisons: lbfgs+l2, liblinear+l1/l2)
4. Entraînement
5. **SHAP Analysis** avec LinearExplainer:
    - Summary plots
    - Dependence plots
    - Force plots
    - Waterfall plots

**Hyperparamètres optimisés**:

-   solver (lbfgs, liblinear)
-   C (régularisation)
-   penalty (l1, l2)
-   max_iter

**Résultat**: Modèle LogisticRegression optimisé avec SHAP

---

### 6️⃣ **06_lr_final_model.ipynb** - Modèle final avec Feature Engineering

**Objectif**: Produire le meilleur modèle avec Feature Engineering intégré

Spécificités:

-   **Feature Engineering appliquée** dès le départ:

    -   difficulty_score
    -   foot_side_match
    -   Analyse de corrélation

-   **Analyse des seuils optimaux**:

    -   Courbe Precision-Recall avec seuil optimal
    -   Seuil maximisant la précision avec recall ≥ 0.60
    -   Visualisation du point optimal

-   **SHAP Analysis complète**:

    -   6 visualisations (comme NB04/05)
    -   Explications par prédiction
    -   Impact des features

-   **Tag MLflow**: `feature_engineering: applied`

**Résultat**: Modèle de production avec explications complètes

---

## 🔐 Configuration des variables d'environnement

### Fichier `.env`

Le projet utilise des variables d'environnement pour les configurations sensibles (tokens, chemins, etc.).

#### Créer le fichier `.env`

```bash
cp .env.example .env
```

Puis remplir les variables selon vos besoins.

#### Variables disponibles

| Variable               | Description                             | Exemple                                        |
| ---------------------- | --------------------------------------- | ---------------------------------------------- |
| `PROJECT_ROOT`         | Chemin racine du projet                 | `/Users/xavier/Documents/OC/OC_P6_Rugby_MLOps` |
| `MLFLOW_TRACKING_URI`  | URI du tracking MLflow                  | `file:./mlruns`                                |
| `MLFLOW_MODEL_NAME`    | Nom du modèle dans le Registry          | `rugby-kicks-logistic-regression`              |
| `MLFLOW_MODEL_VERSION` | Version du modèle à publier (optionnel) | `1` ou laisser vide                            |
| `HF_REPO_ID`           | ID du repository Hugging Face           | `XavierCoulon/rugby-kicks-model`               |
| `HF_TOKEN`             | Token d'authentification Hugging Face   | `hf_xxxxxxxxxxxxx`                             |

#### Exemple de `.env.example`

Voir le fichier `.env.example` pour un template complet avec explications.

---

## 🚀 Publication sur Hugging Face

### Prérequis

1. **Créer un repository Hugging Face**:

    - Aller sur https://huggingface.co/new
    - Remplir le nom du repository
    - Choisir "Model"
    - Cliquer "Create repository"

2. **Générer un token Hugging Face**:

    - Aller sur https://huggingface.co/settings/tokens
    - Créer un nouveau token avec permissions "write"
    - Copier le token

3. **Remplir le `.env`**:
    ```env
    HF_REPO_ID=
    HF_TOKEN=
    MLFLOW_MODEL_VERSION=
    ```

### Lancer le script de publication

#### Option 1: Publier la dernière version en "Production"

```bash
# Laisser MLFLOW_MODEL_VERSION vide dans .env
python push_model_to_huggingface.py
```

#### Option 2: Publier une version spécifique

```bash
# Remplir MLFLOW_MODEL_VERSION=1 dans .env
python push_model_to_huggingface.py
```

### Étapes du script

1. ✅ Récupère le modèle depuis MLflow (version spécifiée ou dernière en Production)
2. ✅ Télécharge les artifacts (model, config, etc.)
3. ✅ Publie sur Hugging Face

### Résultat

Le modèle est disponible sur: `https://huggingface.co/<HF_REPO_ID>`

---

Fichier de constantes globales:

```python
PROCESSED_DATA_PATH = "data/processed/kicks_ready_for_model.csv"
TARGET_COL = "is_goal"  # Variable cible
SEED = 42  # Reproductibilité
FIG_DIR = "outputs/figures"  # Dossier des graphiques
CV_STRATEGY = StratifiedKFold(n_splits=5)  # Cross-validation
```

---

## 📊 MLflow - Suivi des expériences

### Démarrer MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Puis ouvrir: `http://localhost:5000`

### Structure MLflow

```
Expériences:
├── Rugby Kicks - Benchmark Models (NB03)
├── Rugby Kicks - XGBoost Finetuning (NB04)
├── Rugby Kicks - LogisticRegression Finetuning (NB05)
└── Rugby Kicks - LogisticRegression Final Model (NB06)

Chaque run contient:
├── Paramètres (hyperparamètres)
├── Métriques (accuracy, F1, AUC, etc.)
├── Artefacts:
│   ├── Modèle entraîné
│   ├── Matrices de confusion
│   ├── Courbes ROC/PR
│   └── SHAP visualizations
└── Tags (metadata)
```

---

## 📈 Exécution des notebooks

### Option 1: Notebooks Jupyter

```bash
jupyter notebook
```

Puis naviguer vers le notebook souhaité et exécuter les cellules.

### Option 2: Scripts Python

```bash
python -m jupyter nbconvert --to script notebooks/03_modeling.ipynb
python 03_modeling.py
```

### Ordre recommandé d'exécution

1. **01_eda.ipynb** - Exploration
2. **02_features.ipynb** - Feature engineering
3. **03_modeling.ipynb** - Benchmark
4. **04_xgboost_finetuning.ipynb** - Fine-tuning XGBoost
5. **05_lr_finetuning.ipynb** - Fine-tuning LR
6. **06_lr_final_model.ipynb** - Modèle final

---

## 🛠️ Utilitaires (utils.py)

### Fonctions principales

**`compute_train_test_metrics(y_train, y_pred_train, y_proba_train, y_test, y_pred, y_proba)`**

-   Calcule les métriques train et test
-   Retourne: metriques, confusion_matrix, (fpr, fnr)

**`extract_cv_metrics(cv_results)`**

-   Extrait les résultats de cross-validation

---

## 🎯 Résultats typiques

### Comparaison des modèles (NB03)

| Modèle             | Accuracy | F1-score | ROC-AUC  |
| ------------------ | -------- | -------- | -------- |
| Dummy              | ~50%     | -        | 0.50     |
| LogisticRegression | ~75%     | 0.72     | 0.82     |
| RandomForest       | ~77%     | 0.75     | 0.84     |
| SVM                | ~76%     | 0.74     | 0.83     |
| **XGBoost**        | **~80%** | **0.78** | **0.87** |

### Après fine-tuning + SMOTE

-   **XGBoost**: ~82-85% accuracy
-   **LogisticRegression**: ~80-82% accuracy

---

## 📌 Tags MLflow

Chaque run est tagué avec:

-   `author`: Xavier
-   `model_type`: xgboost_finetuned, logistic_regression_final, etc.
-   `optimization_method`: gridsearchcv
-   `resampling`: SMOTE
-   `feature_engineering`: applied (NB06 uniquement)
-   `search_strategy`: reduced_grid_option1 (NB04)

---

## ⚙️ Troubleshooting

### SHAP Analysis échoue

**Problème**: `Exception in SHAP Analysis`

**Solutions**:

-   Vérifier que `shap` est installé: `uv pip install shap`
-   Les modèles très complexes peuvent poser problème
-   Le code inclut une gestion d'erreur qui continue sans SHAP

### MLflow n'affiche pas les artifacts

**Solution**: Vérifier le `FIG_DIR` dans `config.py`

```bash
ls -la outputs/figures/
```

### Memory issue avec GridSearchCV

**Solution**: Réduire `n_jobs`:

```python
GridSearchCV(..., n_jobs=-1)  # Utilise tous les cores
# Ou
GridSearchCV(..., n_jobs=4)   # Limite à 4 cores
```

---

## 📚 Ressources

-   [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
-   [SHAP Documentation](https://shap.readthedocs.io/)
-   [Scikit-learn](https://scikit-learn.org/)
-   [XGBoost](https://xgboost.readthedocs.io/)
-   [uv - Package Manager](https://docs.astral.sh/uv/)

---

## 📝 Notes importantes

1. **Reproductibilité**: `SEED=42` est fixé partout
2. **Data Leakage**: Preprocessing fit uniquement sur train
3. **SMOTE**: Appliqué APRÈS split train/test
4. **Métriques**: F1-weighted pour données imbalancées
5. **SHAP**: LinearExplainer pour modèles linéaires, TreeExplainer pour arbres

---

## 🤝 Contributions

Pour modifier le pipeline:

1. Créer une nouvelle branche: `git checkout -b feature/xyz`
2. Faire les modifications
3. Tester les notebooks
4. Vérifier les runs MLflow
5. Commit et push

---

## 📄 Licence

Projet OpenClassrooms - Parcours AI Engineer

---

**Dernière mise à jour**: Novembre 2025
**Auteur**: Xavier
