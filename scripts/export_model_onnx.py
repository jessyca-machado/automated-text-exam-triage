"""Exporta o modelo scikit-learn para ONNX."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SKLEARN_MODEL_PATH: Final[Path] = (
    ROOT / "artifacts" / "medical_abstracts_model.joblib"
)
ONNX_MODEL_PATH: Final[Path] = (
    ROOT / "artifacts" / "medical_abstracts_model.onnx"
)
INPUT_NAME: Final[str] = "texto"

logger = logging.getLogger(__name__)


def load_model(path: Path) -> Pipeline:
    """Carrega a pipeline treinada.

    Args:
        path: Caminho do modelo Joblib.

    Returns:
        Pipeline treinada.

    Raises:
        FileNotFoundError: Se o modelo não existir.
        TypeError: Se o arquivo não contiver uma Pipeline.
    """
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")

    model = joblib.load(path)

    if not isinstance(model, Pipeline):
        raise TypeError("O arquivo não contém uma Pipeline do scikit-learn.")

    return model


def export_model(model: Pipeline, output_path: Path) -> None:
    """Converte e salva a pipeline no formato ONNX.

    Args:
        model: Pipeline treinada do scikit-learn.
        output_path: Caminho do arquivo ONNX de saída.
    """
    initial_types = [
        (
            INPUT_NAME,
            StringTensorType([None, 1]),
        )
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=17,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())

    logger.info("Modelo ONNX salvo em: %s", output_path)


def main() -> None:
    """Executa a exportação do modelo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    model = load_model(SKLEARN_MODEL_PATH)
    export_model(model, ONNX_MODEL_PATH)


if __name__ == "__main__":
    main()
