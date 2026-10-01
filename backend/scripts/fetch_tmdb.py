"""Descarga el catálogo crudo de TMDB (películas y series) una sola vez, offline.

Guarda JSON sin transformar en `data/raw/tmdb/`:

    genres_movie.json, genres_tv.json   -> listados de géneros (es-ES)
    ids_movie.json, ids_tv.json         -> ids del discover (se regeneran siempre)
    movie/{id}.json, tv/{id}.json       -> detalle + keywords, un archivo por ítem

Uso (desde `backend/`):

    uv run python -m scripts.fetch_tmdb --limit 20
    uv run python -m scripts.fetch_tmdb --min-votes 1000 --max-items-per-type 2000

El script es reanudable: si el JSON de un ítem ya existe no lo vuelve a pedir.
Los listados de ids NO sirven de caché, se regeneran en cada ejecución.

Ojo con la clave: la API key viaja en el query string, así que el texto de
cualquier error de httpx puede incluirla. Por eso ningún log imprime URLs
completas ni excepciones crudas, solo el path sin query y el status.

Nota: hay que ejecutarlo con cwd=backend/ porque `app/config.py` lee el `.env`
con una ruta relativa al cwd.
"""

import argparse
import itertools
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Future
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

import httpx
import tenacity
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

BASE_URL = "https://api.themoviedb.org/3"
PRIMARY_LANG = "es-ES"
FALLBACK_LANG = "en-US"
MEDIA_TYPES = ("movie", "tv")

DEFAULT_MIN_VOTES = 500
DEFAULT_MAX_ITEMS_PER_TYPE = 5000
MAX_PAGE = 500
PROGRESS_EVERY = 100

REQUEST_TIMEOUT_S = 30.0
REQUEST_PAUSE_S = 0.2
MAX_ATTEMPTS = 5
RETRY_MIN_WAIT_S = 1.0
RETRY_MAX_WAIT_S = 30.0

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = BACKEND_DIR / "data" / "raw" / "tmdb"

LOGGER = logging.getLogger("fetch_tmdb")

EXIT_OK = 0
EXIT_PROBLEMA = 1
EXIT_SIN_CREDENCIAL = 2


class Status(StrEnum):
    SAVED = "saved"
    SKIPPED = "skipped"
    MISSING = "missing"
    FAILED = "failed"


class AuthError(RuntimeError):
    """TMDB rechazó la credencial. No tiene sentido reintentar."""


class TmdbStatusError(RuntimeError):
    """Status HTTP inesperado. El mensaje solo lleva status y path."""


def safe_path(url_or_path: str) -> str:
    """Deja solo el path de una URL: el query string es donde viaja la api_key."""
    return urlparse(url_or_path).path


def silence_http_client_logs() -> None:
    """Baja el nivel de los logs de httpx: en INFO imprimen la URL con la api_key."""
    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def write_json_atomic(path: Path, payload: object) -> None:
    """Escribe en `path.tmp` y renombra, para que un corte no deje JSON a medias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def build_client(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Cliente con la credencial en los params por defecto: la key vive en un sitio."""
    return httpx.Client(
        base_url=BASE_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        headers={"accept": "application/json"},
        params={"api_key": api_key},
        transport=transport,
    )


def _pause() -> None:
    time.sleep(REQUEST_PAUSE_S)


def _sleep(seconds: float) -> None:
    """Espera entre reintentos. Va aparte para que los tests no esperen de verdad."""
    time.sleep(seconds)


def _is_retryable_status(response: httpx.Response) -> bool:
    return response.status_code == 429 or response.status_code >= 500


def _make_retry(path: str) -> Retrying:
    """Reintenta solo 429, 5xx y errores de red. 401/403 y 404 no se reintentan."""

    def log_retry(retry_state: RetryCallState) -> None:
        outcome = retry_state.outcome
        if outcome is None:
            return
        if outcome.failed:
            # No imprimimos la excepción: httpx puede incluir la URL con la key.
            motivo = "error de red"
        else:
            motivo = f"HTTP {outcome.result().status_code}"
        LOGGER.warning(
            "Reintento %d/%d en %s (%s)",
            retry_state.attempt_number + 1,
            MAX_ATTEMPTS,
            safe_path(path),
            motivo,
        )

    return Retrying(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT_S, max=RETRY_MAX_WAIT_S),
        retry=(
            retry_if_exception_type(httpx.TransportError)
            | retry_if_result(_is_retryable_status)
        ),
        before_sleep=log_retry,
        sleep=_sleep,
        reraise=True,
    )


def get_json(
    client: httpx.Client,
    path: str,
    params: dict[str, object] | None = None,
) -> dict | None:
    """GET con reintentos. Devuelve None si el recurso no existe (404)."""
    response = _make_retry(path)(lambda: _get_with_auth_guard(client, path, params))
    if response.status_code == 404:
        return None
    if not response.is_success:
        raise TmdbStatusError(
            f"TMDB respondió {response.status_code} en {safe_path(path)}"
        )
    return response.json()


def _get_with_auth_guard(
    client: httpx.Client,
    path: str,
    params: dict[str, object] | None,
) -> httpx.Response:
    response = client.get(path, params=params)
    if response.status_code in (401, 403):
        raise AuthError(
            f"TMDB rechazó la credencial (HTTP {response.status_code}). "
            "Revisa TMDB_API_KEY en tu .env."
        )
    return response


def fetch_genres(client: httpx.Client, raw_dir: Path) -> None:
    """Descarga los géneros en es-ES de películas y series."""
    for media_type in MEDIA_TYPES:
        payload = get_json(client, f"/genre/{media_type}/list", {"language": PRIMARY_LANG})
        if payload is None:
            LOGGER.warning("TMDB no devolvió el listado de géneros de %s", media_type)
            continue
        write_json_atomic(raw_dir / f"genres_{media_type}.json", payload)
        LOGGER.info(
            "Géneros de %s guardados (%d)", media_type, len(payload.get("genres", []))
        )
        _pause()


def fetch_ids(
    client: httpx.Client,
    media_type: str,
    raw_dir: Path,
    min_votes: int,
    max_items: int,
) -> list[int]:
    """Pagina /discover y guarda el listado de ids. Siempre se regenera."""
    ids: list[int] = []
    page = 1
    total_pages: int | None = None

    while page <= MAX_PAGE and len(ids) < max_items:
        payload = get_json(
            client,
            f"/discover/{media_type}",
            {
                "language": PRIMARY_LANG,
                "sort_by": "vote_count.desc",
                "vote_count.gte": min_votes,
                "page": page,
            },
        )
        if payload is None:
            LOGGER.warning("El listado de %s dejó de existir en la página %d", media_type, page)
            break
        total_pages = payload.get("total_pages") or total_pages
        results = payload.get("results") or []
        if not results:
            break
        ids.extend(item["id"] for item in results if isinstance(item.get("id"), int))
        LOGGER.info(
            "Listado de %s: página %d de %s, %d ids",
            media_type,
            page,
            total_pages,
            len(ids),
        )
        if total_pages is not None and page >= total_pages:
            break
        page += 1
        _pause()

    ids = ids[:max_items]
    write_json_atomic(
        raw_dir / f"ids_{media_type}.json",
        {
            "media_type": media_type,
            "min_votes": min_votes,
            "count": len(ids),
            "ids": ids,
        },
    )
    LOGGER.info("Ids de %s guardados: %d", media_type, len(ids))
    return ids


def item_path(raw_dir: Path, media_type: str, item_id: int) -> Path:
    return raw_dir / media_type / f"{item_id}.json"


def fetch_detail(
    client: httpx.Client,
    media_type: str,
    item_id: int,
    raw_dir: Path,
) -> Status:
    """Descarga el detalle de un ítem. Si el archivo existe, no hace ninguna petición."""
    path = item_path(raw_dir, media_type, item_id)
    if path.exists():
        return Status.SKIPPED

    endpoint = f"/{media_type}/{item_id}"
    try:
        payload = get_json(
            client, endpoint, {"language": PRIMARY_LANG, "append_to_response": "keywords"}
        )
        if payload is None:
            LOGGER.warning("%s %d no existe en TMDB (404), se omite", media_type, item_id)
            return Status.MISSING
        if not str(payload.get("overview") or "").strip():
            payload = _fill_english_overview(client, endpoint, payload)
        write_json_atomic(path, payload)
        return Status.SAVED
    except AuthError:
        raise
    except (tenacity.RetryError, TmdbStatusError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "No se pudo descargar %s %d: %s", media_type, item_id, _safe_reason(exc)
        )
        return Status.FAILED
    except OSError as exc:
        LOGGER.error(
            "No se pudo escribir el archivo de %s %d: %s",
            media_type,
            item_id,
            _safe_reason(exc),
        )
        return Status.FAILED
    finally:
        _pause()


def _fill_english_overview(
    client: httpx.Client, endpoint: str, payload: dict
) -> dict:
    """Repite la petición en en-US porque el overview en español vino vacío."""
    fallback = get_json(
        client, endpoint, {"language": FALLBACK_LANG, "append_to_response": "keywords"}
    )
    if fallback is None:
        return payload
    overview = str(fallback.get("overview") or "").strip()
    if overview:
        payload["overview"] = overview
    payload["overview_lang"] = "en"
    return payload


def _safe_reason(exc: BaseException) -> str:
    """Resumen del error sin el texto crudo (las URLs de httpx llevan la api_key)."""
    if isinstance(exc, tenacity.RetryError):
        return _describe_outcome(exc.last_attempt)
    if isinstance(exc, TmdbStatusError):
        return str(exc)
    if isinstance(exc, json.JSONDecodeError):
        return "respuesta no es JSON válido"
    return type(exc).__name__


def _describe_outcome(outcome: Future) -> str:
    if outcome.failed:
        return "errores de red hasta agotar los reintentos"
    return f"HTTP {outcome.result().status_code} hasta agotar los reintentos"


def round_robin(ids_by_type: dict[str, list[int]]) -> Iterator[tuple[str, int]]:
    """Intercala los ids de cada tipo: movie, tv, movie, tv..."""
    media_types = list(ids_by_type)
    rows = itertools.zip_longest(
        *(ids_by_type[media_type] for media_type in media_types), fillvalue=None
    )
    for row in rows:
        for media_type, item_id in zip(media_types, row):
            if item_id is not None:
                yield media_type, item_id


def _format_counters(counters: dict[str, Counter]) -> str:
    return " | ".join(
        f"{media_type} saved={counters[media_type][Status.SAVED]} "
        f"skipped={counters[media_type][Status.SKIPPED]} "
        f"missing={counters[media_type][Status.MISSING]} "
        f"failed={counters[media_type][Status.FAILED]}"
        for media_type in counters
    )


def _totals(counters: dict[str, Counter]) -> tuple[int, int]:
    fallidos = sum(c[Status.FAILED] for c in counters.values())
    guardados = sum(c[Status.SAVED] + c[Status.SKIPPED] for c in counters.values())
    return fallidos, guardados


def run(
    args: argparse.Namespace,
    raw_dir: Path = DEFAULT_RAW_DIR,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Ejecuta la descarga completa. Devuelve el código de salida."""
    key = settings.tmdb_api_key if api_key is None else api_key
    if not key:
        LOGGER.error("Falta TMDB_API_KEY: ponla en backend/.env antes de descargar")
        return EXIT_SIN_CREDENCIAL

    max_items = args.max_items_per_type
    if args.limit is not None:
        max_items = min(max_items, args.limit)

    raw_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Descargando en %s", raw_dir)

    counters: dict[str, Counter] = {media_type: Counter() for media_type in MEDIA_TYPES}
    with build_client(key, transport=transport) as client:
        try:
            fetch_genres(client, raw_dir)
            ids_by_type = {
                media_type: fetch_ids(
                    client, media_type, raw_dir, args.min_votes, max_items
                )
                for media_type in MEDIA_TYPES
            }
            LOGGER.info(
                "A descargar: %s",
                ", ".join(f"{mt}={len(ids)}" for mt, ids in ids_by_type.items()),
            )

            for descargados, (media_type, item_id) in enumerate(
                round_robin(ids_by_type), start=1
            ):
                if args.limit is not None and descargados > args.limit:
                    break
                estado = fetch_detail(client, media_type, item_id, raw_dir)
                counters[media_type][estado] += 1
                if descargados % PROGRESS_EVERY == 0:
                    LOGGER.info(
                        "Progreso: %d ítems (%s)", descargados, _format_counters(counters)
                    )
        except AuthError as exc:
            LOGGER.error("%s", exc)
            return EXIT_SIN_CREDENCIAL

    fallidos, guardados = _totals(counters)
    LOGGER.info("Resumen: %s", _format_counters(counters))
    if fallidos or not guardados:
        return EXIT_PROBLEMA
    return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga películas y series de TMDB como JSON crudo (una sola vez).",
    )
    parser.add_argument(
        "--max-items-per-type",
        type=int,
        default=DEFAULT_MAX_ITEMS_PER_TYPE,
        help=f"Máximo de ítems por tipo (default: {DEFAULT_MAX_ITEMS_PER_TYPE}).",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=DEFAULT_MIN_VOTES,
        help=f"vote_count.gte del discover (default: {DEFAULT_MIN_VOTES}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Descarga solo N ítems en total, repartidos entre película y serie. "
            "También acota el listado de ids, que se sobrescribe."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    silence_http_client_logs()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
