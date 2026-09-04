"""API para classificação automática de laudos médicos."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final, AsyncIterator, cast

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

import time
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import JSONResponse

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODEL_PATH: Final[Path] = (
    ROOT / "artifacts" / "medical_abstracts_model.joblib"
)

logger = logging.getLogger(__name__)
model: Pipeline | None = None


REQUEST_COUNT = Counter(
    "api_requests",
    "Quantidade total de requisições recebidas pela API.",
    labelnames=("method", "endpoint", "status"),
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Tempo de processamento das requisições da API em segundos.",
    labelnames=("method", "endpoint"),
)


class PredictionRequest(BaseModel):
    """Representa os dados recebidos para classificação."""

    texto: str = Field(
        ...,
        min_length=1,
        description="Texto do laudo médico.",
        examples=[
            "Paciente apresenta alterações no sistema cardiovascular."
        ],
    )


class PredictionResponse(BaseModel):
    """Representa o resultado da classificação."""

    classificacao: str = Field(
        ...,
        description="Categoria prevista pelo modelo.",
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo durante a inicialização da API.

    Args:
        application: Instância da aplicação FastAPI.

    Yields:
        Controle de execução durante o ciclo de vida da aplicação.

    Raises:
        FileNotFoundError: Se o arquivo do modelo não existir.
    """
    del application

    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}. "
            "Execute o treinamento antes de iniciar a API."
        )

    model = cast(Pipeline, joblib.load(MODEL_PATH))

    logger.info("Modelo carregado com sucesso: %s", MODEL_PATH)

    yield

    model = None
    logger.info("Modelo liberado.")


app = FastAPI(
    title="Automated Text Exam Triage",
    description="API para classificação automática de laudos médicos.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def instrument_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Registra quantidade e tempo de processamento das requisições.

    Args:
        request: Requisição HTTP recebida.
        call_next: Próximo middleware ou endpoint da aplicação.

    Returns:
        Resposta HTTP gerada pela aplicação.
    """
    if request.url.path == "/metrics":
        return await call_next(request)

    started_at = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - started_at
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(elapsed)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expõe as métricas no formato esperado pelo Prometheus.

    Returns:
        Resposta contendo as métricas da aplicação.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Verifica se a API está disponível.

    Returns:
        Dicionário contendo o status da aplicação.
    """
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Classifica o texto de um laudo médico.

    Args:
        request: Requisição contendo o texto do laudo.

    Returns:
        Categoria prevista pelo modelo.

    Raises:
        HTTPException: Se o modelo não estiver disponível ou ocorrer
            um erro durante a previsão.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está disponível.",
        )

    texto = request.texto.strip()

    if not texto:
        raise HTTPException(
            status_code=422,
            detail="O texto do laudo não pode estar vazio.",
        )

    try:
        prediction = model.predict([texto])[0]
    except Exception as error:
        logger.exception("Erro ao classificar o laudo.")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao classificar o laudo.",
        ) from error

    classification = str(prediction)

    logger.info(
        "Laudo classificado com sucesso: %s",
        classification,
    )

    return PredictionResponse(
        classificacao=classification,
    )
