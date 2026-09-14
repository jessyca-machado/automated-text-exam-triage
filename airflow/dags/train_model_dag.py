"""DAG mensal de ingestão, treinamento, exportação e deploy do modelo."""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pendulum
from airflow.decorators import dag, task
from google.cloud import storage


logger = logging.getLogger(__name__)


@dag(
    dag_id="train_medical_abstracts_model",
    description=(
        "Executa mensalmente a ingestão, o treinamento, "
        "a exportação para ONNX e o deploy do modelo."
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
        "onnx",
        "deployment",
    ],
)
def train_medical_abstracts_model() -> Any:
    """Define o fluxo mensal completo de atualização do modelo."""

    @task
    def ingest_data() -> str:
        """Executa o script de preparação do dataset.

        Returns:
            Caminho do dataset preparado.

        Raises:
            FileNotFoundError: Se o dataset não for gerado.
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
            data_path: Caminho do dataset preparado.

        Returns:
            Caminho do modelo Joblib treinado.

        Raises:
            FileNotFoundError: Se o modelo não for gerado.
            ValueError: Se o caminho do dataset for inesperado.
        """
        from model.train import (
            DATA_PATH,
            MODEL_PATH,
            main as train_model_script,
        )

        if str(DATA_PATH) != data_path:
            raise ValueError(
                "O caminho retornado pela ingestão não corresponde "
                f"ao caminho esperado pelo treinamento: {DATA_PATH}"
            )

        logger.info(
            "Iniciando treinamento com o dataset: %s",
            DATA_PATH,
        )

        train_model_script()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo Joblib não foi gerado: {MODEL_PATH}"
            )

        logger.info(
            "Modelo Joblib treinado com sucesso: %s",
            MODEL_PATH,
        )

        return str(MODEL_PATH)

    @task
    def export_model_onnx(model_path: str) -> str:
        """Exporta o modelo Joblib para ONNX.

        Args:
            model_path: Caminho do modelo Joblib treinado.

        Returns:
            Caminho do modelo ONNX exportado.

        Raises:
            FileNotFoundError: Se o modelo ONNX não for gerado.
            ValueError: Se o caminho informado for inesperado.
        """
        from scripts.export_model_onnx import (
            ONNX_MODEL_PATH,
            main as export_model_script,
            SKLEARN_MODEL_PATH,
        )

        if str(SKLEARN_MODEL_PATH) != model_path:
            raise ValueError(
                "O caminho retornado pelo treinamento não corresponde "
                f"ao caminho esperado para exportação: "
                f"{SKLEARN_MODEL_PATH}"
            )

        logger.info(
            "Exportando modelo Joblib para ONNX: %s",
            model_path,
        )

        export_model_script()

        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo ONNX não foi gerado: {ONNX_MODEL_PATH}"
            )

        logger.info(
            "Modelo ONNX exportado com sucesso: %s",
            ONNX_MODEL_PATH,
        )

        return str(ONNX_MODEL_PATH)

    @task
    def publish_model(model_path: str) -> str:
        """Publica o modelo ONNX e atualiza o Cloud Run.

        Args:
            model_path: Caminho local do modelo ONNX.

        Returns:
            URI do modelo publicado no Cloud Storage.

        Raises:
            FileNotFoundError: Se o modelo não existir.
            ValueError: Se o arquivo não tiver extensão ``.onnx``.
        """
        project_id = os.environ["GCP_PROJECT_ID"]
        region = os.environ["GCP_REGION"]
        bucket_name = os.environ["GCP_BUCKET_NAME"]
        service_name = os.environ["CLOUD_RUN_SERVICE"]

        local_model_path = Path(model_path)

        if not local_model_path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX não encontrado: {local_model_path}"
            )

        if local_model_path.suffix != ".onnx":
            raise ValueError(
                "A task publish_model esperava um arquivo ONNX: "
                f"{local_model_path}"
            )

        model_version = pendulum.now("America/Sao_Paulo").strftime(
            "%Y%m%d%H%M%S"
        )

        blob_name = (
            "models/"
            f"medical_abstracts_model_{model_version}.onnx"
        )
        model_uri = f"gs://{bucket_name}/{blob_name}"

        logger.info(
            "Publicando modelo ONNX em: %s",
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
            "Modelo ONNX publicado com sucesso: %s",
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
                (
                    f"MODEL_URI={model_uri},"
                    f"MODEL_FORMAT=onnx,"
                    f"MODEL_VERSION={model_version}"
                ),
            ],
            check=True,
        )

        logger.info(
            "Cloud Run atualizado com o modelo ONNX: "
            "serviço=%s, modelo=%s",
            service_name,
            model_uri,
        )

        return model_uri

    dataset_path = ingest_data()
    joblib_model_path = train_model(dataset_path)
    onnx_model_path = export_model_onnx(joblib_model_path)
    publish_model(onnx_model_path)


train_medical_abstracts_model()
