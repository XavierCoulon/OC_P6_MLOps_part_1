"""
Script de publication du modèle MLflow vers Hugging Face
Envoie la version 'Production' du modèle depuis MLflow vers le Model Hub HF
"""

import os
import mlflow
import joblib
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi
from dotenv import load_dotenv

# --- IMPORTS ONNX ---
# Assure-toi d'avoir fait: pip install skl2onnx onnx
try:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print(
        "[bold yellow]⚠️ Attention: skl2onnx non installé. La conversion ONNX sera sautée.[/bold yellow]"
    )

# --- CHARGER LES VARIABLES D'ENVIRONNEMENT ---
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
MLFLOW_MODEL_VERSION = os.getenv(
    "MLFLOW_MODEL_VERSION", None
)  # None = dernière, ou "1", "2", etc.
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- VALIDATION DES VARIABLES ---
if not MLFLOW_MODEL_NAME:
    raise ValueError("❌ MLFLOW_MODEL_NAME non défini dans .env")
if not HF_REPO_ID:
    raise ValueError("❌ HF_REPO_ID non défini dans .env")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN non défini dans .env")

print("[bold cyan]🚀 Publication du modèle MLflow vers Hugging Face[/bold cyan]")
print(f"   MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"   Modèle MLflow: {MLFLOW_MODEL_NAME}")
print(f"   Repo HF: {HF_REPO_ID}")


# --- FONCTION DE CONVERSION ---
def convert_and_save_onnx(source_path, dest_folder):
    """Convertit un modèle PKL en ONNX en respectant les noms de colonnes."""
    print(f"   ⚙️ Conversion ONNX en cours depuis: {source_path}")

    # 1. Charger le modèle
    model = joblib.load(source_path)

    # 2. Reconstruire la signature exacte des features
    # C'est CRUCIAL : L'ordre et les noms doivent être identiques à ceux de l'entraînement
    feature_names = [
        "time_norm",
        "distance",
        "angle",
        "wind_speed",
        "precipitation_probability",
        "is_left_footed",
        "game_away",
        "is_endgame",
        "is_start",
        "is_left_side",
        "has_previous_attempts",
    ]

    # 3. Créer un DataFrame "Dummy" (Factice)
    # Une seule ligne, remplie de zéros, en float32
    # Cela permet à to_onnx de lire les noms de colonnes et les types automatiquement
    dummy_input = pd.DataFrame(
        np.zeros((1, len(feature_names)), dtype=np.float32), columns=feature_names
    )

    print("   ⚙️ Génération du graphe ONNX via signature Pandas...")

    # 4. Conversion
    # On passe X=dummy_input au lieu de initial_types
    onx_result = to_onnx(model, X=dummy_input, target_opset=12)

    # Gestion du type de retour (au cas où ce soit un tuple)
    if isinstance(onx_result, tuple):
        onx_model = onx_result[0]
    else:
        onx_model = onx_result

    # 5. Sauvegarde
    output_path = os.path.join(dest_folder, "model.onnx")
    with open(output_path, "wb") as f:
        f.write(onx_model.SerializeToString())  # type: ignore

    print(f"   ✅ Modèle ONNX généré : {output_path}")


# --- CONFIGURATION MLFLOW ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

try:
    # 1. Récupérer le modèle
    print("\n[cyan]1️⃣  Récupération du modèle...[/cyan]")

    if MLFLOW_MODEL_VERSION:
        # Récupérer une version SPÉCIFIQUE
        print(f"   Mode: Version spécifique #{MLFLOW_MODEL_VERSION}")
        latest_prod = client.get_model_version(
            name=MLFLOW_MODEL_NAME, version=MLFLOW_MODEL_VERSION
        )
    else:
        # Récupérer la DERNIÈRE version en Production
        print(f"   Mode: Dernière version en stage 'Production'")
        prod_models = client.get_latest_versions(
            MLFLOW_MODEL_NAME, stages=["Production"]
        )

        if not prod_models:
            raise Exception(
                f"❌ Aucun modèle '{MLFLOW_MODEL_NAME}' en stage 'Production' !\n"
                f"   Conseil: Promouvoir d'abord une version en 'Production' dans MLflow UI"
            )

        latest_prod = prod_models[0]

    print(f"   ✅ Modèle trouvé: Version {latest_prod.version}")
    print(f"   📦 Source: {latest_prod.source}")
    print(f"   📝 Description: {latest_prod.description or 'N/A'}")

    # 2. Télécharger les artifacts
    print("\n[cyan]2️⃣  Téléchargement des artifacts...[/cyan]")
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=latest_prod.source, dst_path="./model_export_prod"
    )
    print(f"   ✅ Téléchargé vers: {local_path}")

    # =================================================================
    # ✨ NOUVELLE ÉTAPE : CONVERSION ONNX
    # =================================================================
    if ONNX_AVAILABLE:
        print("\n[cyan]2️⃣.5️⃣  Optimisation ONNX...[/cyan]")

        # MLflow stocke souvent le modèle sous le nom 'model.pkl' DANS le dossier téléchargé.
        # Parfois c'est 'model/model.pkl'. Il faut trouver le fichier .pkl.
        pkl_path = os.path.join(local_path, "model.pkl")

        # Si le fichier n'est pas à la racine, on cherche dedans (cas fréquent MLflow)
        if not os.path.exists(pkl_path):
            # Tentative de recherche récursive simple ou chemin standard MLflow
            potential_paths = [
                os.path.join(local_path, "model.pkl"),
                os.path.join(
                    local_path, "model", "model.pkl"
                ),  # Structure standard MLflow
            ]
            for p in potential_paths:
                if os.path.exists(p):
                    pkl_path = p
                    break

        if os.path.exists(pkl_path):
            try:
                # On sauvegarde le .onnx à la racine du dossier d'export
                convert_and_save_onnx(pkl_path, local_path)
            except Exception as e:
                print(f"[bold red]⚠️ Echec conversion ONNX: {e}[/bold red]")
                print("   Le déploiement continuera avec le fichier .pkl uniquement.")
        else:
            print(
                f"[yellow]⚠️ Impossible de trouver le fichier .pkl dans {local_path}[/yellow]"
            )

    # 3. Upload vers Hugging Face
    print("\n[cyan]3️⃣  Publication sur Hugging Face...[/cyan]")
    api = HfApi(token=HF_TOKEN)

    api.upload_folder(
        folder_path=local_path,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"Deployment of MLflow Model v{latest_prod.version} (Stage: Production)",
    )

    print(f"   ✅ Publication réussie!")
    print(f"   🌐 Modèle disponible sur: https://huggingface.co/{HF_REPO_ID}")
    print("\n[bold green]✨ Terminé avec succès![/bold green]")

except Exception as e:
    print(f"\n[bold red]❌ Erreur: {str(e)}[/bold red]")
    exit(1)
