"""Testes de integração da API FastAPI."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

import app.main as api
from model.train import create_pipeline


TEXT_COLUMN: Final[str] = "texto"
TARGET_COLUMN: Final[str] = "label"


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    """Cria e salva um modelo pequeno para os testes da API.

    Args:
        tmp_path: Diretório temporário fornecido pelo pytest.

    Returns:
        Caminho do modelo salvo.
    """
    data = pd.DataFrame(
        {
            TEXT_COLUMN: [
                "Paciente apresenta doença cardíaca.",
                "Alteração no sistema cardiovascular.",
                "Paciente apresenta dor abdominal.",
                "Alteração no sistema digestivo.",
                "Lesão no sistema nervoso.",
                "Comprometimento neurológico.",
            ],
            TARGET_COLUMN: [
                "cardiovascular",
                "cardiovascular",
                "digestive",
                "digestive",
                "nervous",
                "nervous",
            ],
        }
    )

    model: Pipeline = create_pipeline()
    model.fit(data[TEXT_COLUMN], data[TARGET_COLUMN])

    model_path = tmp_path / "medical_abstracts_model.joblib"
    joblib.dump(model, model_path)

    return model_path


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    trained_model_path: Path,
) -> Iterator[TestClient]:
    """Cria um cliente de teste com um modelo temporário carregado.

    Args:
        monkeypatch: Fixture para substituir o caminho do modelo.
        trained_model_path: Caminho do modelo temporário.

    Yields:
        Cliente HTTP de teste da API.
    """
    monkeypatch.setattr(api, "MODEL_PATH", trained_model_path)

    with TestClient(api.app) as test_client:
        yield test_client


def test_health_check(client: TestClient) -> None:
    """Verifica o endpoint de saúde da API."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_classification(client: TestClient) -> None:
    """Verifica se a API classifica um laudo válido."""
    response = client.post(
        "/predict",
        json={
            "texto": (
                "Paciente apresenta alterações no sistema "
                "cardiovascular."
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["classificacao"] in {
        "cardiovascular",
        "digestive",
        "nervous",
    }


def test_predict_rejects_empty_text(client: TestClient) -> None:
    """Verifica se a API rejeita um texto vazio."""
    response = client.post(
        "/predict",
        json={"texto": "   "},
    )

    assert response.status_code == 422
    assert "não pode estar vazio" in response.json()["detail"]


def test_predict_requires_text_field(client: TestClient) -> None:
    """Verifica se o campo texto é obrigatório."""
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_metrics_endpoint_exposes_request_metrics(
    client: TestClient,
) -> None:
    """Verifica se a API expõe métricas do Prometheus."""
    health_response = client.get("/health")

    assert health_response.status_code == 200

    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    assert "text/plain" in metrics_response.headers["content-type"]

    metrics = metrics_response.text

    assert "api_requests_total" in metrics
    assert "api_request_duration_seconds" in metrics
    assert 'endpoint="/health"' in metrics
    assert 'method="GET"' in metrics
    assert 'status="200"' in metrics


def test_metrics_endpoint_records_prediction_request(
    client: TestClient,
) -> None:
    """Verifica se as requisições de classificação são contabilizadas."""
    response = client.post(
        "/predict",
        json={
            "texto": (
                "Paciente apresenta alterações no sistema "
                "cardiovascular."
            )
        },
    )

    assert response.status_code == 200

    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200

    metrics = metrics_response.text

    assert 'endpoint="/predict"' in metrics
    assert 'method="POST"' in metrics
    assert 'status="200"' in metrics
    assert "api_request_duration_seconds_count" in metrics
