"""Descarga el catálogo crudo de anime de Jikan (proxy de MyAnimeList) una sola vez.

Guarda JSON sin transformar en `data/raw/jikan/`:

    pages/page_{n}.json   -> cada página de /top/anime, tal cual llega
    anime.json            -> lista aplanada, deduplicada por mal_id y filtrada

Uso (desde `backend/`):

    uv run python -m scripts.fetch_jikan --limit 50
    uv run python -m scripts.fetch_jikan --min-scored-by 500 --max-items 8000

Jikan no necesita API key. El ritmo es conservador, ~1 s entre peticiones: la API
limita a 3 req/s y los reintentos respetan el header Retry-After.

Ojo con `type`: no se manda en la petición. No está documentado que /top/anime
acepte una lista separada por comas y el endpoint de MAL que hay debajo solo
acepta un valor, así que el filtro de tipo (tv, movie,ova, ona) se aplica al
aplanar, comparando sin distinguir mayúsculas porque Jikan responde `TV`,
`Movie`, `OVA`, `ONA`. En la página cruda, por tanto, también entran los music,
cm, pv y tv_special; el aplanado es el que los deja fuera.

El script es reanudable: si pages/page_{n}.json ya existe no se vuelve a pedir, se
lee del disco. Pero el orden de /top/anime puede cambiar entre ejecuciones (cambian
las puntuaciones y la popularidad), así que al reanudar se pueden quedar huecos
entre páginas y algún mal_id repetido. El aplanado deduplica por mal_id, pero un
hueco no se recupera: para un catálogo limpio hay que borrar pages/ y empezar de
cero.

Los filtros de calidad (--min-scored-by y descartar sinopsis vacía) se aplican al
aplanar, no al descargar: así se puede cambiar el umbral sin volver a pedir nada.

Códigos de salida: 0 si todo fue bien, 2 si los flags no cuadran, 1 si alguna
página falló, si no se guardó ni se saltó ninguna página, o si el aplanado no
dejó ni un anime (en ese caso anime.json no se toca, para no pisar un catálogo
bueno con una lista vacía).

Nota: hay que ejecutarlo con cwd=backend/ porque `app/config.py` lee el `.env`
con una ruta relativa al cwd.
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
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

BASE_URL = "https://api.jikan.moe/v4"
TOP_PATH = "/top/anime"

KEPT_TYPES = ("tv", "movie", "ova", "ona")
PAGE_LIMIT = 25

DEFAULT_MAX_ITEMS = 4000
DEFAULT_MIN_SCORED_BY = 1000

MAX_PAGE = 500
PROGRESS_EVERY = 10

REQUEST_TIMEOUT_S = 30.0
REQUEST_PAUSE_S = 1.0
MAX_ATTEMPTS = 5
RETRY_MIN_WAIT_S = 1.0
RETRY_MAX_WAIT_S = 30.0

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = BACKEND_DIR / "data" / "raw" / "jikan"
PAGES_DIRNAME = "pages"
FLAT_FILENAME = "anime.json"

LOGGER = logging.getLogger("fetch_jikan")

EXIT_OK = 0
EXIT_PROBLEMA = 1
EXIT_CONFIG = 2


class Status(StrEnum):
    SAVED = "saved"
    SKIPPED = "skipped"
    EMPTY = "empty"
    MISSING = "missing"
    FAILED = "failed"


class JikanStatusError(RuntimeError):
    """Status HTTP inesperado. El mensaje solo lleva status y path."""


def safe_path(url_or_path: str) -> str:
    """Deja solo el path de una URL: los logs no sacan query strings."""
    return urlparse(url_or_path).path


def silence_http_client_logs() -> None:
    """Baja el nivel de los logs de httpx: en INFO imprimen la URL entera."""
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


def build_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Cliente contra la base de Jikan. Sin credenciales: la API es abierta."""
    return httpx.Client(
        base_url=BASE_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        headers={"accept": "application/json"},
        transport=transport,
    )


def _pause() -> None:
    time.sleep(REQUEST_PAUSE_S)


def _sleep(seconds: float) -> None:
    """Espera entre reintentos. Va aparte para que los tests no esperen de verdad."""
    time.sleep(seconds)


def _is_retryable_status(response: httpx.Response) -> bool:
    return response.status_code == 429 or response.status_code >= 500


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """El header Retry-After de Jikan viene en segundos, no como fecha HTTP."""
    valor = response.headers.get("retry-after")
    if valor is None:
        return None
    try:
        return max(0.0, float(valor.strip()))
    except ValueError:
        return None


_ESPERA_EXPONENCIAL = wait_exponential(
    multiplier=1, min=RETRY_MIN_WAIT_S, max=RETRY_MAX_WAIT_S
)


def _wait_retry(retry_state: RetryCallState) -> float:
    """Con 429 manda el Retry-After; si no hay, backoff exponencial."""
    outcome = retry_state.outcome
    if outcome is not None and not outcome.failed:
        retry_after = _retry_after_seconds(outcome.result())
        if retry_after is not None:
            return min(retry_after, RETRY_MAX_WAIT_S)
    return _ESPERA_EXPONENCIAL(retry_state)


def _make_retry(path: str) -> Retrying:
    """Reintenta 429, 5xx y errores de red. 404 y el resto de 4xx no."""

    def log_retry(retry_state: RetryCallState) -> None:
        outcome = retry_state.outcome
        if outcome is None:
            return
        if outcome.failed:
            LOGGER.warning(
                "Reintento %d/%d en %s (error de red)",
                retry_state.attempt_number + 1,
                MAX_ATTEMPTS,
                safe_path(path),
            )
            return
        LOGGER.warning(
            "Reintento %d/%d en %s (HTTP %d)",
            retry_state.attempt_number + 1,
            MAX_ATTEMPTS,
            safe_path(path),
            outcome.result().status_code,
        )

    return Retrying(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=_wait_retry,
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
    response = _make_retry(path)(lambda: client.get(path, params=params))
    if response.status_code == 404:
        return None
    if not response.is_success:
        raise JikanStatusError(
            f"Jikan respondió {response.status_code} en {safe_path(path)}"
        )
    return response.json()


def _safe_reason(exc: BaseException) -> str:
    """Resumen del error sin el texto crudo de httpx."""
    if isinstance(exc, tenacity.RetryError):
        return _describe_outcome(exc.last_attempt)
    if isinstance(exc, JikanStatusError):
        return str(exc)
    if isinstance(exc, json.JSONDecodeError):
        return "respuesta no es JSON válido"
    return type(exc).__name__


def _describe_outcome(outcome) -> str:
    if outcome.failed:
        return "errores de red hasta agotar los reintentos"
    return f"HTTP {outcome.result().status_code} hasta agotar los reintentos"


def pages_dir(raw_dir: Path) -> Path:
    return raw_dir / PAGES_DIRNAME


def page_path(directory: Path, page: int) -> Path:
    return directory / f"page_{page}.json"


def _read_json(path: Path) -> dict | None:
    """Lee un JSON ya guardado. None si no se puede leer o no es un objeto."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def fetch_page(client: httpx.Client, page: int, directory: Path) -> tuple[Status, dict]:
    """Descarga una página de /top/anime. Si el archivo existe, no hace peticiones."""
    path = page_path(directory, page)
    if path.exists():
        guardada = _read_json(path)
        if guardada is not None:
            return Status.SKIPPED, guardada
        LOGGER.warning("La página %d guardada está corrupta, se vuelve a pedir", page)

    try:
        payload = get_json(client, TOP_PATH, {"page": page, "limit": PAGE_LIMIT})
    except (tenacity.RetryError, JikanStatusError, json.JSONDecodeError) as exc:
        LOGGER.warning("No se pudo descargar la página %d: %s", page, _safe_reason(exc))
        return Status.FAILED, {}
    finally:
        _pause()

    if payload is None:
        LOGGER.warning("La página %d no existe en Jikan (404), se omite", page)
        return Status.MISSING, {}
    if not payload.get("data"):
        return Status.EMPTY, payload
    write_json_atomic(path, payload)
    return Status.SAVED, payload


def fetch_pages(
    client: httpx.Client,
    directory: Path,
    max_items: int,
    counters: Counter,
) -> None:
    """Pagina /top/anime hasta el tope de ítems, o hasta que no haya más."""
    page = 1
    vistos = 0

    while page <= MAX_PAGE:
        estado, payload = fetch_page(client, page, directory)
        counters[estado] += 1

        if estado in (Status.FAILED, Status.MISSING):
            LOGGER.warning(
                "Sin la página %d no se puede seguir: fin de la descarga", page
            )
            break

        items = payload.get("data") or []
        if not items:
            LOGGER.info("La página %d vino vacía: fin de la descarga", page)
            break

        vistos += len(items)
        LOGGER.info(
            "Página %d: %d ítems, %d acumulados (tope %d)",
            page,
            len(items),
            vistos,
            max_items,
        )
        if page % PROGRESS_EVERY == 0:
            LOGGER.info("Progreso: %d páginas descargadas", page)

        if vistos >= max_items:
            LOGGER.info("Tope de %d ítems alcanzado", max_items)
            break
        if not (payload.get("pagination") or {}).get("has_next_page"):
            LOGGER.info("has_next_page=false en la página %d: fin de la descarga", page)
            break
        page += 1


def es_tipo_permitido(tipo: object) -> bool:
    """Compara sin distinguir mayúsculas: Jikan responde `TV`, `Movie`, `OVA`."""
    return str(tipo or "").strip().casefold() in KEPT_TYPES


def _int(valor: object) -> int:
    return valor if isinstance(valor, int) else 0


def _nombres(entradas: object) -> list[str]:
    """`genres[].name` y compañía: solo los nombres, sin los objetos de Jikan."""
    if not isinstance(entradas, list):
        return []
    nombres: list[str] = []
    for entrada in entradas:
        if isinstance(entrada, dict):
            nombre = str(entrada.get("name") or "").strip()
            if nombre:
                nombres.append(nombre)
    return nombres


def flatten_item(entrada: dict) -> dict:
    """Un ítem con los campos que nos interesan, del JSON de Jikan al nuestro.

    `aired.from` sale plano como `aired_from` y las listas de objetos salen como
    listas de nombres. La sinopsis viene recortada.
    """
    aired = entrada.get("aired")
    images = entrada.get("images")
    jpg = images.get("jpg") if isinstance(images, dict) else None
    return {
        "mal_id": entrada.get("mal_id"),
        "title": entrada.get("title"),
        "title_english": entrada.get("title_english"),
        "title_japanese": entrada.get("title_japanese"),
        "type": entrada.get("type"),
        "episodes": entrada.get("episodes"),
        "status": entrada.get("status"),
        "duration": entrada.get("duration"),
        "rating": entrada.get("rating"),
        "score": entrada.get("score"),
        "scored_by": _int(entrada.get("scored_by")),
        "popularity": _int(entrada.get("popularity")),
        "synopsis": str(entrada.get("synopsis") or "").strip(),
        "year": entrada.get("year"),
        "season": entrada.get("season"),
        "aired_from": aired.get("from") if isinstance(aired, dict) else None,
        "genres": _nombres(entrada.get("genres")),
        "themes": _nombres(entrada.get("themes")),
        "demographics": _nombres(entrada.get("demographics")),
        "studios": _nombres(entrada.get("studios")),
        "poster_url": jpg.get("large_image_url") if isinstance(jpg, dict) else None,
    }


def flatten(pages: list[dict], min_scored_by: int) -> tuple[list[dict], Counter]:
    """Aplana las páginas, deduplica por mal_id y aplica los filtros de calidad."""
    stats: Counter = Counter()
    vistos: set[int] = set()
    anime: list[dict] = []

    for page in pages:
        for entrada in page.get("data") or []:
            if not isinstance(entrada, dict):
                continue
            stats["vistos"] += 1
            mal_id = entrada.get("mal_id")
            if not isinstance(mal_id, int):
                stats["sin_id"] += 1
                continue
            if mal_id in vistos:
                stats["duplicados"] += 1
                continue
            vistos.add(mal_id)

            if not es_tipo_permitido(entrada.get("type")):
                stats["fuera_de_tipo"] += 1
                continue

            item = flatten_item(entrada)
            if not item["synopsis"]:
                stats["sin_sinopsis"] += 1
                continue
            if item["scored_by"] < min_scored_by:
                stats["pocos_votos"] += 1
                continue

            anime.append(item)

    return anime, stats


def _page_number(path: Path) -> int:
    sufijo = path.stem.removeprefix("page_")
    return int(sufijo) if sufijo.isdigit() else 0


def read_pages(directory: Path) -> list[dict]:
    """Lee todas las páginas guardadas, en orden de número."""
    paginas: list[dict] = []
    for path in sorted(directory.glob("page_*.json"), key=_page_number):
        payload = _read_json(path)
        if payload is None:
            LOGGER.warning("Se ignora %s: no se puede leer como JSON", path.name)
            continue
        paginas.append(payload)
    return paginas


def items_de_paginas(limit: int) -> int:
    """Redondea N ítems hacia arriba a páginas completas de Jikan."""
    return -(-limit // PAGE_LIMIT) * PAGE_LIMIT


def _validar(args: argparse.Namespace) -> str | None:
    """Mensaje del flag que no cuadra, o None si todo está bien."""
    if args.max_items < 1:
        return f"--max-items tiene que ser >= 1 (recibido {args.max_items})"
    if args.min_scored_by < 0:
        return f"--min-scored-by no puede ser negativo (recibido {args.min_scored_by})"
    if args.limit is not None and args.limit < 1:
        return f"--limit tiene que ser >= 1 (recibido {args.limit})"
    return None


def _format_counters(counters: Counter) -> str:
    return " | ".join(
        f"{status.value}={counters[status]}"
        for status in (
            Status.SAVED,
            Status.SKIPPED,
            Status.EMPTY,
            Status.MISSING,
            Status.FAILED,
        )
    )


def _format_stats(stats: Counter) -> str:
    return (
        f"vistos={stats['vistos']} fuera_de_tipo={stats['fuera_de_tipo']} "
        f"sin_sinopsis={stats['sin_sinopsis']} pocos_votos={stats['pocos_votos']} "
        f"duplicados={stats['duplicados']} sin_id={stats['sin_id']}"
    )


def run(
    args: argparse.Namespace,
    raw_dir: Path = DEFAULT_RAW_DIR,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Ejecuta la descarga y el aplanado. Devuelve el código de salida."""
    problema = _validar(args)
    if problema:
        LOGGER.error("%s", problema)
        return EXIT_CONFIG

    max_items = args.max_items
    if args.limit is not None:
        max_items = min(max_items, items_de_paginas(args.limit))

    directory = pages_dir(raw_dir)
    directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Descargando en %s", directory)

    counters: Counter = Counter()
    with build_client(transport=transport) as client:
        fetch_pages(client, directory, max_items, counters)

    paginas = read_pages(directory)
    if not paginas:
        LOGGER.error("No hay páginas en %s: no se genera %s", directory, FLAT_FILENAME)
        return EXIT_PROBLEMA

    anime, stats = flatten(paginas, args.min_scored_by)
    if anime:
        write_json_atomic(raw_dir / FLAT_FILENAME, anime)
    else:
        LOGGER.warning(
            "Ningún anime pasa los filtros: %s no se toca, prueba con --min-scored-by menor",
            FLAT_FILENAME,
        )
    LOGGER.info("Aplanado: %d anime (%s)", len(anime), _format_stats(stats))

    fallidos = counters[Status.FAILED] + counters[Status.MISSING]
    guardadas = counters[Status.SAVED] + counters[Status.SKIPPED]
    LOGGER.info(
        "Resumen: páginas %s | anime=%d", _format_counters(counters), len(anime)
    )

    if fallidos or not guardadas or not anime:
        return EXIT_PROBLEMA
    return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga anime de Jikan (MyAnimeList) como JSON crudo (una sola vez).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help=f"Máximo de ítems a descargar (default: {DEFAULT_MAX_ITEMS}).",
    )
    parser.add_argument(
        "--min-scored-by",
        type=int,
        default=DEFAULT_MIN_SCORED_BY,
        help=(
            "Votos mínimos para conservar un anime. Se aplica al aplanar, no al "
            f"descargar (default: {DEFAULT_MIN_SCORED_BY})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Descarga solo N ítems, redondeados a páginas completas de 25. "
            "Para pruebas rápidas."
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
