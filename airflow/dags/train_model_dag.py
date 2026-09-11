"""DAG periódica de ingestão e treinamento do modelo."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

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

    dataset_path = ingest_data()
    train_model(dataset_path)


train_medical_abstracts_model()
