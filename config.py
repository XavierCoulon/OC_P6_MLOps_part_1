import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = str(PROJECT_ROOT / "data")
RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
INTERIM_DIR = str(PROJECT_ROOT / "data" / "interim")
PROCESSED_DIR = str(PROJECT_ROOT / "data" / "processed")

OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
FIG_DIR = str(PROJECT_ROOT / "outputs" / "figures")
REPORT_DIR = str(PROJECT_ROOT / "outputs" / "reports")

RAW_DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "final_kicks_dataset.csv")
INTERIM_DATA_PATH = str(PROJECT_ROOT / "data" / "interim" / "kicks_clean.csv")
PROCESSED_DATA_PATH = str(
    PROJECT_ROOT / "data" / "processed" / "kicks_ready_for_model.csv"
)

TARGET_COL = "resultat"
SEED = 42

FINAL_MODEL_NAME = "rugby_kicks_model"

# ============================================================
# 📊 Configuration d'évaluation des modèles
# ============================================================
# Scores personnalisés pour la Cross-Validation
GLOBAL_SCORING = {
    "acc": "accuracy",
    "auc": "roc_auc",
    "log_loss": "neg_log_loss",
    # Métriques pour la Classe 1 (Majorité)
    "recall_1": make_scorer(recall_score, pos_label=1),
    "precision_1": make_scorer(precision_score, pos_label=1, zero_division=0),
    "f1_1": make_scorer(f1_score, pos_label=1),
    # Métriques pour la Classe 0 (Minorité - CRUCIALE)
    "recall_0": make_scorer(recall_score, pos_label=0),
    "precision_0": make_scorer(precision_score, pos_label=0, zero_division=0),
    "f1_0": make_scorer(f1_score, pos_label=0),
}

# Stratégie de Cross-Validation (Stratified pour l'imbalance)
CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for dir_path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIG_DIR, REPORT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print(f"✅ Config initialisée depuis : {PROJECT_ROOT}")
