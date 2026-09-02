"""Treina o modelo de classificação dos laudos médicos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = ROOT / "data" / "laudos.csv"
MODEL_PATH: Final[Path] = (
    ROOT / "artifacts" / "medical_abstracts_model.joblib"
)

TEXT_COLUMN: Final[str] = "texto"
TARGET_COLUMN: Final[str] = "label"
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Carrega o dataset preparado.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        DataFrame contendo os dados de treinamento.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se as colunas obrigatórias não forem encontradas.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    data = pd.read_csv(path)

    required_columns = {TEXT_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes no dataset: "
            f"{sorted(missing_columns)}"
        )

    data = data.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).copy()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str).str.strip()
    data = data.loc[data[TEXT_COLUMN].str.len() > 0]

    if data.empty:
        raise ValueError("O dataset não contém registros válidos.")

    return data.reset_index(drop=True)


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.Series[str], pd.Series[str], pd.Series[str], pd.Series[str]]:
    """Divide os dados em conjuntos de treinamento e teste.

    A divisão utiliza estratificação para preservar a distribuição das
    classes nos dois conjuntos.

    Args:
        data: DataFrame com as colunas de texto e classificação.

    Returns:
        Tupla contendo ``x_train``, ``x_test``, ``y_train`` e ``y_test``.
    """
    features = data[TEXT_COLUMN]
    target = data[TARGET_COLUMN]

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def create_pipeline() -> Pipeline:
    """Cria o pipeline de vetorização e classificação.

    Returns:
        Pipeline contendo TF-IDF e regressão logística.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_model(
    model: Pipeline,
    x_train: pd.Series[str],
    y_train: pd.Series[str],
) -> Pipeline:
    """Treina o pipeline com os dados de treinamento.

    Args:
        model: Pipeline que será treinado.
        x_train: Textos de treinamento.
        y_train: Rótulos de treinamento.

    Returns:
        Pipeline treinado.
    """
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    x_test: pd.Series[str],
    y_test: pd.Series[str],
) -> None:
    """Avalia o modelo utilizando o conjunto de teste.

    Args:
        model: Pipeline treinado.
        x_test: Textos de teste.
        y_test: Rótulos reais do conjunto de teste.
    """
    predictions = model.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    logger.info("Relatório de classificação:\n%s", report)


def save_model(model: Pipeline, path: Path) -> None:
    """Salva o modelo treinado em formato Joblib.

    Args:
        model: Pipeline treinado.
        path: Caminho onde o modelo será salvo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    logger.info("Modelo salvo em: %s", path)


def main() -> None:
    """Executa o carregamento, treinamento, avaliação e salvamento."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Carregando dataset...")
    data = load_data(DATA_PATH)

    logger.info("Total de registros: %d", len(data))
    logger.info(
        "Distribuição das classes:\n%s",
        data[TARGET_COLUMN].value_counts().to_string(),
    )

    x_train, x_test, y_train, y_test = split_data(data)

    logger.info(
        "Dados divididos: treino=%d, teste=%d",
        len(x_train),
        len(x_test),
    )

    model = create_pipeline()
    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
