"""Classes para inferência com Joblib e ONNX Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.pipeline import Pipeline


TEXT_INPUT_NAME: Final[str] = "texto"


class Predictor(Protocol):
    """Interface comum para os modelos de inferência."""

    def predict(self, text: str) -> str:
        """Classifica um texto.

        Args:
            text: Texto do laudo.

        Returns:
            Classificação prevista.
        """
        ...


class JoblibPredictor:
    """Realiza inferência usando uma Pipeline do scikit-learn."""

    def __init__(self, model_path: Path) -> None:
        """Inicializa o preditor Joblib.

        Args:
            model_path: Caminho do modelo Joblib.
        """
        self.model: Pipeline = joblib.load(model_path)

    def predict(self, text: str) -> str:
        """Classifica um texto usando scikit-learn.

        Args:
            text: Texto do laudo.

        Returns:
            Especialidade prevista.
        """
        prediction = self.model.predict([text])[0]
        return str(prediction)


class OnnxPredictor:
    """Realiza inferência utilizando ONNX Runtime."""

    def __init__(self, model_path: Path) -> None:
        """Inicializa a sessão ONNX.

        Args:
            model_path: Caminho do modelo ONNX.
        """
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, text: str) -> str:
        """Classifica o texto com o modelo ONNX.

        Args:
            text: Texto do laudo.

        Returns:
            Classificação prevista.
        """
        input_data = np.asarray(
            [[text]],
            dtype=object,
        )

        outputs = self.session.run(
            None,
            {self.input_name: input_data},
        )

        prediction = outputs[0][0]

        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()

        return str(prediction)
