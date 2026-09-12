"""DAG periódica de ingestão e treinamento do modelo."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from google.cloud import storage

import logging
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task


logger = logging.getLogger(__name__)


@dag(
    dag_id="train_medical_abstracts_model",
    description=(
        "Executa periodicamente a ingestão dos dados "
        "e o treinamento do modelo."
    ),
    schedule="0 2 1 * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="America/Sao_Paulo"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "ml-team",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=[
        "machine-learning",
        "monthly",
        "ingestion",
        "retraining",
    ],
)
def train_medical_abstracts_model() -> Any:
    """Define o fluxo periódico de ingestão e treinamento."""

    @task
    def ingest_data() -> str:
        """Executa o script de preparação do dataset.

        Returns:
            Caminho esperado do dataset preparado.
        """
        from scripts.prepare_dataset_medical_abstracts import (
            OUT_PATH,
            main as prepare_dataset,
        )

        logger.info("Iniciando ingestão do dataset.")

        prepare_dataset()

        if not OUT_PATH.exists():
            raise FileNotFoundError(
                f"Dataset não foi gerado: {OUT_PATH}"
            )

        logger.info(
            "Dataset preparado com sucesso: %s",
            OUT_PATH,
        )

        return str(OUT_PATH)

    @task
    def train_model(data_path: str) -> str:
        """Executa o script existente de treinamento.

        Args:
            data_path: Caminho do dataset preparado pela task anterior.

        Returns:
            Caminho esperado do modelo treinado.
        """
        from model.train import (
            DATA_PATH,
            MODEL_PATH,
            main as train_model_script,
        )

        if str(DATA_PATH) != data_path:
            raise ValueError(
                "O caminho retornado pela ingestão não corresponde "
                f"ao caminho esperado pelo treinamento: "
                f"{DATA_PATH}"
            )

        logger.info(
            "Iniciando treinamento com o dataset: %s",
            DATA_PATH,
        )

        train_model_script()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não foi gerado: {MODEL_PATH}"
            )

        logger.info(
            "Modelo treinado com sucesso: %s",
            MODEL_PATH,
        )

        return str(MODEL_PATH)

    @task
    def publish_model(model_path: str) -> str:
        """Publica o modelo no Cloud Storage e atualiza o Cloud Run.

        Args:
            model_path: Caminho local do modelo treinado.

        Returns:
            URI do modelo publicado no Cloud Storage.

        Raises:
            FileNotFoundError: Se o modelo não existir.
            RuntimeError: Se o Cloud Run não puder ser atualizado.
        """
        project_id = os.environ["GCP_PROJECT_ID"]
        region = os.environ["GCP_REGION"]
        bucket_name = os.environ["GCP_BUCKET_NAME"]
        service_name = os.environ["CLOUD_RUN_SERVICE"]

        local_model_path = Path(model_path)

        if not local_model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {local_model_path}"
            )

        model_version = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = (
            "models/"
            f"medical_abstracts_model_{model_version}.joblib"
        )
        model_uri = f"gs://{bucket_name}/{blob_name}"

        logger.info(
            "Publicando modelo em: %s",
            model_uri,
        )

        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(
            str(local_model_path),
            content_type="application/octet-stream",
        )

        logger.info(
            "Modelo publicado com sucesso: %s",
            model_uri,
        )

        subprocess.run(
            [
                "gcloud",
                "run",
                "services",
                "update",
                service_name,
                "--project",
                project_id,
                "--region",
                region,
                "--update-env-vars",
                f"MODEL_URI={model_uri},"
                f"MODEL_VERSION={model_version}",
            ],
            check=True,
        )

        logger.info(
            "Cloud Run atualizado: serviço=%s, modelo=%s",
            service_name,
            model_uri,
        )

        return model_uri

    dataset_path = ingest_data()
    model_path = train_model(dataset_path)
    publish_model(model_path)


train_medical_abstracts_model()
