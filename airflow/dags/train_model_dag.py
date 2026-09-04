"""DAG para simular o treinamento do modelo de classificação."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pendulum
from airflow.decorators import dag, task


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.train import (  # noqa: E402
    DATA_PATH,
    MODEL_PATH,
    create_pipeline,
    evaluate_model,
    load_data,
    save_model,
    split_data,
    train_model,
)


@dag(
    dag_id="train_medical_abstracts_model",
    description="Lê os dados, treina e salva o modelo de classificação.",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["machine-learning", "training"],
)
def train_medical_abstracts_model() -> Any:
    """Define a DAG de treinamento do modelo."""

    @task
    def read_training_data() -> dict[str, str | int]:
        """Lê e valida o dataset utilizado no treinamento.

        Returns:
            Metadados do dataset, incluindo caminho e quantidade de registros.

        Raises:
            FileNotFoundError: Se o dataset não existir.
            ValueError: Se o dataset não possuir dados válidos.
        """
        data = load_data(DATA_PATH)

        return {
            "data_path": str(DATA_PATH),
            "records": len(data),
        }

    @task
    def train_and_save_model(
        dataset_info: dict[str, str | int],
    ) -> str:
        """Treina o modelo e salva o artefato em formato Joblib.

        Args:
            dataset_info: Informações do dataset produzidas pela task
                de leitura.

        Returns:
            Caminho do modelo salvo.

        Raises:
            FileNotFoundError: Se o dataset não existir.
            ValueError: Se os dados forem inválidos.
        """
        data_path = Path(str(dataset_info["data_path"]))
        data = load_data(data_path)

        x_train, x_test, y_train, y_test = split_data(data)

        model = create_pipeline()
        train_model(model, x_train, y_train)
        evaluate_model(model, x_test, y_test)
        save_model(model, MODEL_PATH)

        return str(MODEL_PATH)

    dataset_info = read_training_data()
    train_and_save_model(dataset_info)


train_medical_abstracts_model()
