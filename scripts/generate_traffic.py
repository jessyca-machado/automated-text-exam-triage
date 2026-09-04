"""Gera tráfego para popular as métricas da API."""

from __future__ import annotations

import argparse
import logging
from typing import Final

import httpx


DEFAULT_URL: Final[str] = "http://127.0.0.1:8000/predict"
DEFAULT_REQUESTS: Final[int] = 200
DEFAULT_TEXT: Final[str] = (
    "Paciente apresenta alterações no sistema cardiovascular."
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Processa os argumentos da linha de comando.

    Returns:
        Argumentos informados pelo usuário.
    """
    parser = argparse.ArgumentParser(
        description="Gera tráfego para a API de classificação."
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help=f"Quantidade de requisições. Padrão: {DEFAULT_REQUESTS}",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL da API. Padrão: {DEFAULT_URL}",
    )

    return parser.parse_args()


def generate_traffic(url: str, requests_count: int) -> None:
    """Envia requisições para a API.

    Args:
        url: URL do endpoint de classificação.
        requests_count: Quantidade de requisições a enviar.

    Raises:
        ValueError: Se a quantidade de requisições for inválida.
        httpx.HTTPError: Se ocorrer erro de comunicação.
    """
    if requests_count <= 0:
        raise ValueError("A quantidade de requisições deve ser maior que zero.")

    payload = {"texto": DEFAULT_TEXT}

    with httpx.Client(timeout=30.0) as client:
        for request_number in range(1, requests_count + 1):
            response = client.post(url, json=payload)

            logger.info(
                "Requisição %d/%d - status=%d",
                request_number,
                requests_count,
                response.status_code,
            )


def main() -> None:
    """Executa a geração de tráfego."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    arguments = parse_arguments()
    generate_traffic(
        url=arguments.url,
        requests_count=arguments.requests,
    )


if __name__ == "__main__":
    main()
