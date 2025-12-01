"""
Script de publication du modèle MLflow vers Hugging Face
Envoie la version 'Production' du modèle depuis MLflow vers le Model Hub HF
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi
from dotenv import load_dotenv

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
