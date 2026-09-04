"""Testes unitários da pipeline de treinamento."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from model.train import (
    create_pipeline,
    load_data,
    save_model,
)


TEXT_COLUMN: Final[str] = "texto"
TARGET_COLUMN: Final[str] = "label"


def test_load_data_returns_valid_data(tmp_path: Path) -> None:
    """Verifica se o dataset é carregado e higienizado corretamente."""
    dataset_path = tmp_path / "laudos.csv"

    data = pd.DataFrame(
        {
            TEXT_COLUMN: [
                "Texto cardiovascular.",
                "Texto digestivo.",
                None,
                "   ",
            ],
            TARGET_COLUMN: [
                "cardiovascular",
                "digestive",
                "nervous",
                "general",
            ],
        }
    )
    data.to_csv(dataset_path, index=False)

    result = load_data(dataset_path)

    assert list(result.columns) == [TEXT_COLUMN, TARGET_COLUMN]
    assert len(result) == 2
    assert result[TEXT_COLUMN].tolist() == [
        "Texto cardiovascular.",
        "Texto digestivo.",
    ]


def test_load_data_raises_error_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    """Verifica o erro quando o dataset não existe."""
    dataset_path = tmp_path / "inexistente.csv"

    with pytest.raises(FileNotFoundError):
        load_data(dataset_path)


def test_load_data_raises_error_when_required_column_is_missing(
    tmp_path: Path,
) -> None:
    """Verifica o erro quando uma coluna obrigatória está ausente."""
    dataset_path = tmp_path / "laudos.csv"

    data = pd.DataFrame(
        {
            TEXT_COLUMN: ["Texto de teste."],
        }
    )
    data.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
        load_data(dataset_path)


def test_create_pipeline_returns_sklearn_pipeline() -> None:
    """Verifica se a pipeline possui os componentes esperados."""
    pipeline = create_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_save_model_creates_joblib_file(tmp_path: Path) -> None:
    """Verifica se o modelo é salvo corretamente em formato Joblib."""
    model_path = tmp_path / "artifacts" / "model.joblib"
    pipeline = create_pipeline()

    save_model(pipeline, model_path)

    assert model_path.exists()

    loaded_model = joblib.load(model_path)

    assert isinstance(loaded_model, Pipeline)
