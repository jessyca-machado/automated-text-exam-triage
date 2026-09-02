"""Baixa o Medical Abstracts TC Corpus via Kaggle e grava data/laudos.csv."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import kagglehub
import pandas as pd


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUT_PATH: Final[Path] = ROOT / "data" / "laudos.csv"
DATASET_ID: Final[str] = "saharalaa/medical-abstracts-tc-corpus"

LABELS: Final[dict[int, str]] = {
    1: "neoplasms",
    2: "digestive",
    3: "nervous",
    4: "cardiovascular",
    5: "general",
}

logger = logging.getLogger(__name__)


def download_dataset() -> Path:
    """Baixa a versão mais recente do dataset pelo Kaggle.

    Returns:
        Path do diretório que contém os arquivos do dataset.

    Raises:
        FileNotFoundError: Se o diretório baixado não existir.
    """
    logger.info("Iniciando download do dataset: %s", DATASET_ID)

    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Diretório do dataset não encontrado: {dataset_path}"
        )

    logger.info("Dataset disponível em: %s", dataset_path)

    return dataset_path


def find_csv_files(directory: Path) -> list[Path]:
    """Localiza arquivos CSV dentro do diretório do dataset.

    Args:
        directory: Diretório raiz onde os arquivos serão procurados.

    Returns:
        Lista ordenada de arquivos CSV encontrados.

    Raises:
        FileNotFoundError: Se nenhum arquivo CSV for encontrado.
    """
    csv_files = sorted(directory.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV encontrado em: {directory}"
        )

    logger.info("%d arquivo(s) CSV encontrado(s)", len(csv_files))

    return csv_files


def read_csv_files(csv_files: list[Path]) -> pd.DataFrame:
    """Lê e combina os arquivos CSV do dataset.

    Args:
        csv_files: Lista de caminhos dos arquivos CSV.

    Returns:
        DataFrame contendo os dados combinados.
    """
    dataframes = [
        pd.read_csv(csv_file)
        for csv_file in csv_files
    ]

    data = pd.concat(dataframes, ignore_index=True)

    logger.info("Dados carregados: %d registros", len(data))

    return data


def prepare_data() -> pd.DataFrame:
    """Baixa, combina e padroniza os dados do corpus.

    Os rótulos numéricos são convertidos para nomes de categorias.
    Registros sem texto ou sem categoria válida são removidos.

    Returns:
        DataFrame contendo as colunas ``texto`` e ``label``.

    Raises:
        ValueError: Se as colunas obrigatórias não forem encontradas.
    """
    directory_dataset = download_dataset()
    csv_files = find_csv_files(directory_dataset)
    data = read_csv_files(csv_files)

    mandatory_columns = {
        "medical_abstract",
        "condition_label",
    }
    missing_columns = mandatory_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes no dataset: "
            f"{sorted(missing_columns)}"
        )

    data = data.rename(
        columns={"medical_abstract": "texto"}
    )

    data["condition_label"] = pd.to_numeric(
        data["condition_label"],
        errors="coerce",
    )
    data["label"] = data["condition_label"].map(LABELS)
    data["texto"] = data["texto"].astype("string").str.strip()

    data = data.dropna(subset=["texto", "label"])
    data = data.loc[
        data["texto"].str.len() > 0,
        ["texto", "label"],
    ]

    data = data.reset_index(drop=True)

    logger.info(
        "Dados preparados com sucesso: %d registros",
        len(data),
    )

    return data


def save_data(data: pd.DataFrame, path: Path) -> None:
    """Salva os dados processados em um arquivo CSV.

    Args:
        data: DataFrame que será salvo.
        path: Path de destino do arquivo CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    logger.info("Arquivo salvo em: %s", path)


def main() -> None:
    """Executa o download, processamento e salvamento dos dados."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data = prepare_data()
    save_data(data, OUT_PATH)

    logger.info("Total de registros: %d", len(data))
    logger.info(
        "Distribuição das classes:\n%s",
        data["label"].value_counts().to_string(),
    )


if __name__ == "__main__":
    main()
