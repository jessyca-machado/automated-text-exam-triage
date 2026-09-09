"""Compara a latência dos modelos Joblib e ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Final, Protocol

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.pipeline import Pipeline


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
JOBLIB_MODEL_PATH: Final[Path] = (
    ROOT / "artifacts" / "medical_abstracts_model.joblib"
)
ONNX_MODEL_PATH: Final[Path] = (
    ROOT / "artifacts" / "medical_abstracts_model.onnx"
)

DEFAULT_REQUESTS: Final[int] = 100
DEFAULT_WARMUP: Final[int] = 10
DEFAULT_TEXT: Final[str] = (
    "Paciente apresenta alterações no sistema cardiovascular."
)

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Define a interface comum para os modelos testados."""

    def predict(self, text: str) -> str:
        """Realiza uma previsão para um texto."""
        ...


class JoblibPredictor:
    """Adaptador para inferência usando scikit-learn."""

    def __init__(self, model: Pipeline) -> None:
        self.model = model

    def predict(self, text: str) -> str:
        """Realiza a previsão usando a pipeline original.

        Args:
            text: Texto do laudo.

        Returns:
            Classificação prevista.
        """
        prediction = self.model.predict([text])[0]
        return str(prediction)


class OnnxPredictor:
    """Adaptador para inferência usando ONNX Runtime."""

    def __init__(self, model_path: Path) -> None:
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, text: str) -> str:
        """Realiza a previsão usando ONNX Runtime.

        Args:
            text: Texto do laudo.

        Returns:
            Classificação prevista.
        """
        input_data = np.array([[text]], dtype=object)
        outputs = self.session.run(
            None,
            {self.input_name: input_data},
        )

        return str(outputs[0][0])


def measure_latency(
    predictor: Predictor,
    text: str,
    requests_count: int,
    warmup_count: int,
) -> dict[str, float | int]:
    """Mede a latência de um modelo.

    Args:
        predictor: Modelo adaptado para inferência.
        text: Texto utilizado no benchmark.
        requests_count: Quantidade de inferências medidas.
        warmup_count: Quantidade de inferências de aquecimento.

    Returns:
        Métricas de latência em milissegundos.
    """
    for _ in range(warmup_count):
        predictor.predict(text)

    latencies: list[float] = []

    for _ in range(requests_count):
        started_at = time.perf_counter()
        predictor.predict(text)
        elapsed = time.perf_counter() - started_at
        latencies.append(elapsed * 1000)

    sorted_latencies = sorted(latencies)
    p95_index = round(0.95 * (len(sorted_latencies) - 1))

    return {
        "requests": requests_count,
        "min_ms": round(min(latencies), 3),
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(sorted_latencies[p95_index], 3),
        "max_ms": round(max(latencies), 3),
    }


def parse_arguments() -> argparse.Namespace:
    """Processa os argumentos da linha de comando.

    Returns:
        Argumentos informados pelo usuário.
    """
    parser = argparse.ArgumentParser(
        description="Compara a latência dos modelos Joblib e ONNX."
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
    )

    return parser.parse_args()


def validate_predictions(
    joblib_model: Predictor,
    onnx_model: Predictor,
    texts: list[str],
) -> None:
    """Valida se os modelos retornam as mesmas classificações.

    Args:
        joblib_model: Modelo original carregado via Joblib.
        onnx_model: Modelo otimizado carregado via ONNX Runtime.
        texts: Lista de textos usados na validação.

    Raises:
        ValueError: Se a lista de textos estiver vazia.
        RuntimeError: Se os modelos retornarem classificações diferentes.
    """
    if not texts:
        raise ValueError("A lista de textos para validação não pode estar vazia.")

    for text in texts:
        joblib_prediction = joblib_model.predict(text)
        onnx_prediction = onnx_model.predict(text)

        if joblib_prediction != onnx_prediction:
            raise RuntimeError(
                "Os modelos retornaram classificações diferentes. "
                f"Texto: {text!r}; "
                f"Joblib: {joblib_prediction!r}; "
                f"ONNX: {onnx_prediction!r}"
            )

    logger.info(
        "Validação concluída: os modelos retornaram "
        "as mesmas classificações para %d texto(s).",
        len(texts),
    )


def main() -> None:
    """Executa a comparação de latência."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    arguments = parse_arguments()

    joblib_model = JoblibPredictor(
        joblib.load(JOBLIB_MODEL_PATH)
    )
    onnx_model = OnnxPredictor(ONNX_MODEL_PATH)

    validation_texts = [
        arguments.text,
        "Paciente apresenta dor abdominal e alterações digestivas.",
        "Paciente apresenta comprometimento neurológico.",
        "Exame indica presença de neoplasia.",
    ]

    validate_predictions(
        joblib_model=joblib_model,
        onnx_model=onnx_model,
        texts=validation_texts,
    )

    joblib_results = measure_latency(
        predictor=joblib_model,
        text=arguments.text,
        requests_count=arguments.requests,
        warmup_count=arguments.warmup,
    )

    onnx_results = measure_latency(
        predictor=onnx_model,
        text=arguments.text,
        requests_count=arguments.requests,
        warmup_count=arguments.warmup,
    )

    results = {
        "joblib": joblib_results,
        "onnx": onnx_results,
    }

    logger.info(
        "Resultados da comparação de latência:\n%s",
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    main()
