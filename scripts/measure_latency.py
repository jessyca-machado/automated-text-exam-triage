"""Mede a latência local do endpoint de classificação da API."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Final

import httpx


DEFAULT_URL: Final[str] = "http://127.0.0.1:8000/predict"
DEFAULT_TEXT: Final[str] = (
    "Paciente apresenta alterações no sistema cardiovascular."
)
DEFAULT_REQUESTS: Final[int] = 30
DEFAULT_WARMUP_REQUESTS: Final[int] = 5
DEFAULT_TIMEOUT: Final[float] = 30.0

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Processa os argumentos da linha de comando.

    Returns:
        Argumentos informados pelo usuário.
    """
    parser = argparse.ArgumentParser(
        description="Mede a latência local da API de classificação."
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL do endpoint. Padrão: {DEFAULT_URL}",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help=(
            "Quantidade de requisições medidas. "
            f"Padrão: {DEFAULT_REQUESTS}"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_REQUESTS,
        help=(
            "Quantidade de requisições de aquecimento. "
            f"Padrão: {DEFAULT_WARMUP_REQUESTS}"
        ),
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Texto enviado para o endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout em segundos. Padrão: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Arquivo JSON opcional para salvar os resultados.",
    )

    return parser.parse_args()


def send_request(
    client: httpx.Client,
    url: str,
    text: str,
) -> float:
    """Envia uma requisição e mede o tempo total de resposta.

    Args:
        client: Cliente HTTP utilizado na requisição.
        url: URL do endpoint de classificação.
        text: Texto do laudo enviado à API.

    Returns:
        Tempo total da requisição em segundos.

    Raises:
        RuntimeError: Se a API retornar um status diferente de 200.
    """
    payload = {"texto": text}

    started_at = time.perf_counter()
    response = client.post(url, json=payload)
    elapsed = time.perf_counter() - started_at

    if response.status_code != httpx.codes.OK:
        raise RuntimeError(
            f"API retornou status {response.status_code}: "
            f"{response.text}"
        )

    return elapsed


def calculate_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """Calcula um percentil simples sobre uma lista de valores.

    Args:
        values: Valores de latência em segundos.
        percentile: Percentual desejado entre 0 e 100.

    Returns:
        Valor correspondente ao percentil em segundos.

    Raises:
        ValueError: Se a lista estiver vazia ou o percentual for inválido.
    """
    if not values:
        raise ValueError("Não há valores para calcular o percentil.")

    if not 0 <= percentile <= 100:
        raise ValueError("O percentil deve estar entre 0 e 100.")

    sorted_values = sorted(values)
    index = round((percentile / 100) * (len(sorted_values) - 1))

    return sorted_values[index]


def build_results(
    latencies: list[float],
    url: str,
    requests_count: int,
    warmup_count: int,
) -> dict[str, object]:
    """Cria um resumo estatístico das latências medidas.

    Args:
        latencies: Latências das requisições em segundos.
        url: URL utilizada no teste.
        requests_count: Quantidade de requisições medidas.
        warmup_count: Quantidade de requisições de aquecimento.

    Returns:
        Dicionário com as métricas do benchmark.
    """
    return {
        "url": url,
        "requests": requests_count,
        "warmup_requests": warmup_count,
        "min_ms": round(min(latencies) * 1000, 3),
        "mean_ms": round(statistics.mean(latencies) * 1000, 3),
        "median_ms": round(statistics.median(latencies) * 1000, 3),
        "p95_ms": round(calculate_percentile(latencies, 95) * 1000, 3),
        "max_ms": round(max(latencies) * 1000, 3),
    }


def run_benchmark(
    url: str,
    text: str,
    requests_count: int,
    warmup_count: int,
    timeout: float,
) -> dict[str, object]:
    """Executa o benchmark de latência da API.

    Args:
        url: URL do endpoint de classificação.
        text: Texto enviado nas requisições.
        requests_count: Quantidade de requisições medidas.
        warmup_count: Quantidade de requisições de aquecimento.
        timeout: Timeout de cada requisição em segundos.

    Returns:
        Métricas calculadas para as requisições.

    Raises:
        ValueError: Se a quantidade de requisições for inválida.
        RuntimeError: Se ocorrer erro de comunicação com a API.
    """
    if requests_count <= 0:
        raise ValueError("A quantidade de requisições deve ser maior que zero.")

    if warmup_count < 0:
        raise ValueError("A quantidade de aquecimento não pode ser negativa.")

    latencies: list[float] = []

    try:
        with httpx.Client(timeout=timeout) as client:
            logger.info(
                "Executando %d requisições de aquecimento.",
                warmup_count,
            )

            for _ in range(warmup_count):
                send_request(client, url, text)

            logger.info(
                "Medindo latência de %d requisições.",
                requests_count,
            )

            for _ in range(requests_count):
                latencies.append(send_request(client, url, text))

    except httpx.HTTPError as error:
        raise RuntimeError(
            f"Não foi possível acessar a API em {url}: {error}"
        ) from error

    return build_results(
        latencies=latencies,
        url=url,
        requests_count=requests_count,
        warmup_count=warmup_count,
    )


def save_results(results: dict[str, object], output_path: Path) -> None:
    """Salva os resultados do benchmark em formato JSON.

    Args:
        results: Métricas calculadas.
        output_path: Caminho do arquivo de saída.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Resultados salvos em: %s", output_path)


def main() -> None:
    """Executa a medição e exibe as métricas de latência."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    arguments = parse_arguments()

    results = run_benchmark(
        url=arguments.url,
        text=arguments.text,
        requests_count=arguments.requests,
        warmup_count=arguments.warmup,
        timeout=arguments.timeout,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if arguments.output is not None:
        save_results(results, arguments.output)


if __name__ == "__main__":
    main()
