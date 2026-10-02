"""Descarga el ranking crudo de anime de la API oficial de MyAnimeList (v2) una vez.

Guarda JSON sin transformar en `data/raw/mal/`:

    pages/page_{n}.json   -> cada página de /anime/ranking, tal cual llega
    anime.json            -> lista aplanada, deduplicada por id y filtrada

Uso (desde `backend/`):

    uv run python -m scripts.fetch_mal --limit 50
    uv run python -m scripts.fetch_mal --min-scoring-users 500 --max-items 8000

La client id viaja en el header `X-MAL-CLIENT-ID`, nunca en la URL, así que ningún
log imprime URLs completas ni excepciones crudas: solo el path sin query y el
status. Aquí no hace falta silenciar los logs de httpx por la clave, pero se hace
igualmente porque en INFO también escupen el query string.

Sobre la API (todo verificado contra api.myanimelist.net, no de memoria):

* `limit` admite de 1 a 500. Con 501 o 999 responde `400 {"error": "limit"}`.
  Aquí se piden 50 por página porque es lo que hace `--limit 50` exacto; con 500
  el run completo seriam 8 peticiones en vez de 80.
* Cada ítem viene como `{"node": {...}, "ranking": {"rank": N}}`: los `fields`
  caen dentro de `node`. `main_picture` llega siempre, sin pedirlo.
* `paging` trae solo `next`, una URL absoluta con el offset (y con los params que
  le mandamos, `fields` incluido). Aun así el offset lo calculamos nosotros y no
  seguimos ese `next`: porque al final de la lista esa clave DESAPARECE (no viene
  en null) y porque el offset propio es lo que hace reanudable una descarga a
  mitad, sin depender de una URL guardada.
* Una client id inválida no da 401: responde `400 {"error": "bad_request",
  "message": "Invalid client id"}`. De ahí que el guardia de credencial vigile 401
  y 403, y también 400 con "client id" en el mensaje.
* El ranking `all` tiene más de 14.000 entradas, así que el tope de 4.000 ítems
  es el que manda, no el final de la lista.

El script es reanudable: si pages/page_{n}.json ya existe no se vuelve a pedir, se
lee del disco. Pero el ranking puede cambiar entre ejecuciones (cambian las notas y
la popularidad), así que al reanudar se pueden quedar huecos entre páginas y algún
id repetido. El aplanado deduplica por id, pero un hueco no se recupera: para un
catálogo limpio hay que borrar pages/ y empezar de cero.

Los filtros de calidad (--min-scoring-users, sinopsis no vacía y media_type
permitido) se aplican al aplanar, no al descargar: así se puede cambiar el umbral
sin volver a pedir nada.

Códigos de salida: 0 si todo fue bien, 2 si falta la client id, si MAL la rechaza
o si los flags no cuadran, 1 si alguna página falló, si no se guardó ni se saltó
ninguna página, o si el aplanado no dejó ni un anime (en ese caso anime.json no se
toca, para no pisar un catálogo bueno con una lista vacía).

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

from app.config import settings

BASE_URL = "https://api.myanimelist.net/v2"
RANKING_PATH = "/anime/ranking"
RANKING_TYPE = "all"

# MAL rechaza limit > 500. Bajamos a 50 solo para que --limit 50 sea exacto.
PAGE_LIMIT = 50

FIELDS = (
    "id",
    "title",
    "alternative_titles",
    "start_date",
    "synopsis",
    "mean",
    "rank",
    "popularity",
    "num_list_users",
    "num_scoring_users",
    "genres",
    "media_type",
    "status",
    "num_episodes",
    "start_season",
    "average_episode_duration",
    "rating",
    "studios",
    "main_picture",
)

KEPT_MEDIA_TYPES = ("tv", "movie", "ova", "ona")

DEFAULT_MAX_ITEMS = 4000
DEFAULT_MIN_SCORING_USERS = 1000

MAX_PAGE = 500
PROGRESS_EVERY = 10

REQUEST_TIMEOUT_S = 30.0
REQUEST_PAUSE_S = 1.0
MAX_ATTEMPTS = 5
RETRY_MIN_WAIT_S = 1.0
RETRY_MAX_WAIT_S = 30.0

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = BACKEND_DIR / "data" / "raw" / "mal"
PAGES_DIRNAME = "pages"
FLAT_FILENAME = "anime.json"

LOGGER = logging.getLogger("fetch_mal")

EXIT_OK = 0
EXIT_PROBLEMA = 1
EXIT_CONFIG = 2


class Status(StrEnum):
    SAVED = "saved"
    SKIPPED = "skipped"
    EMPTY = "empty"
    MISSING = "missing"
    FAILED = "failed"


class MalAuthError(RuntimeError):
    """MAL rechazó la client id. No tiene sentido reintentar."""


class MalStatusError(RuntimeError):
    """Status HTTP inesperado. El mensaje solo lleva status y path."""


def safe_path(url_or_path: str) -> str:
    """Deja solo el path de una URL: los logs no sacan query strings."""
    return urlparse(url_or_path).path


def silence_http_client_logs() -> None:
    """Baja el nivel de los logs de httpx: en INFO imprimen la URL con el query."""
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


def build_client(
    client_id: str,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Cliente con la client id en un header: la clave no aparece en ninguna URL."""
    return httpx.Client(
        base_url=BASE_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        headers={"X-MAL-CLIENT-ID": client_id, "accept": "application/json"},
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
    """El header Retry-After de MAL viene en segundos, no como fecha HTTP."""
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
    response = _make_retry(path)(lambda: _get_with_auth_guard(client, path, params))
    if response.status_code == 404:
        return None
    if not response.is_success:
        raise MalStatusError(
            f"MAL respondió {response.status_code} en {safe_path(path)}"
        )
    return response.json()


def _get_with_auth_guard(
    client: httpx.Client,
    path: str,
    params: dict[str, object] | None,
) -> httpx.Response:
    response = client.get(path, params=params)
    if _es_rechazo_de_credencial(response):
        raise MalAuthError(
            f"MAL rechazó la client id (HTTP {response.status_code}). "
            "Revisa MAL_CLIENT_ID en tu .env."
        )
    return response


def _es_rechazo_de_credencial(response: httpx.Response) -> bool:
    """401/403 y, por lo que hace MAL con una client id mala, también 400."""
    if response.status_code in (401, 403):
        return True
    if response.status_code != 400:
        return False
    return "client id" in _mensaje_de_error(response).casefold()


def _mensaje_de_error(response: httpx.Response) -> str:
    """El campo `message` del cuerpo de error de MAL.

    Solo se usa para el check de credencial de arriba: ni se registra ni entra en
    el texto de MalAuthError, que se queda en el status.
    """
    try:
        payload = response.json()
    except ValueError:
        return ""
    mensaje = payload.get("message") if isinstance(payload, dict) else None
    return mensaje if isinstance(mensaje, str) else ""


def _safe_reason(exc: BaseException) -> str:
    """Resumen del error sin el texto crudo de httpx."""
    if isinstance(exc, tenacity.RetryError):
        return _describe_outcome(exc.last_attempt)
    if isinstance(exc, MalStatusError):
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


def page_offset(page: int) -> int:
    """MAL pagina por offset, no por número de página."""
    return (page - 1) * PAGE_LIMIT


def _read_json(path: Path) -> dict | None:
    """Lee un JSON ya guardado. None si no se puede leer o no es un objeto."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def fetch_page(client: httpx.Client, page: int, directory: Path) -> tuple[Status, dict]:
    """Descarga una página del ranking. Si el archivo existe, no hace peticiones."""
    path = page_path(directory, page)
    if path.exists():
        guardada = _read_json(path)
        if guardada is not None:
            return Status.SKIPPED, guardada
        LOGGER.warning("La página %d guardada está corrupta, se vuelve a pedir", page)

    try:
        payload = get_json(
            client,
            RANKING_PATH,
            {
                "ranking_type": RANKING_TYPE,
                "limit": PAGE_LIMIT,
                "offset": page_offset(page),
                "fields": ",".join(FIELDS),
            },
        )
    except (tenacity.RetryError, MalStatusError, json.JSONDecodeError) as exc:
        LOGGER.warning("No se pudo descargar la página %d: %s", page, _safe_reason(exc))
        return Status.FAILED, {}
    finally:
        _pause()

    if payload is None:
        LOGGER.warning("La página %d no existe en MAL (404), se omite", page)
        return Status.MISSING, {}
    if not payload.get("data"):
        return Status.EMPTY, payload
    write_json_atomic(path, payload)
    return Status.SAVED, payload


def hay_siguiente_pagina(payload: dict) -> bool:
    """MAL omite `paging.next` en la última página, así que `.get` y listo."""
    paging = payload.get("paging")
    return bool(paging.get("next")) if isinstance(paging, dict) else False


def fetch_pages(
    client: httpx.Client,
    directory: Path,
    max_items: int,
    counters: Counter,
) -> None:
    """Pagina el ranking hasta el tope de ítems, o hasta que no haya más."""
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
            "Página %d (offset %d): %d ítems, %d acumulados (tope %d)",
            page,
            page_offset(page),
            len(items),
            vistos,
            max_items,
        )
        if page % PROGRESS_EVERY == 0:
            LOGGER.info("Progreso: %d páginas descargadas", page)

        if vistos >= max_items:
            LOGGER.info("Tope de %d ítems alcanzado", max_items)
            break
        if not hay_siguiente_pagina(payload):
            LOGGER.info(
                "MAL no da paging.next en la página %d: fin de la descarga", page
            )
            break
        page += 1


def es_tipo_permitido(media_type: object) -> bool:
    """Compara sin distinguir mayúsculas: MAL usa `tv`, pero no rodamos."""
    return str(media_type or "").strip().casefold() in KEPT_MEDIA_TYPES


def _int(valor: object) -> int:
    return valor if isinstance(valor, int) else 0


def _float(valor: object) -> float | None:
    return (
        valor
        if isinstance(valor, (int, float)) and not isinstance(valor, bool)
        else None
    )


def _nombres(entradas: object) -> list[str]:
    """`genres` y `studios` traen objetos con id y name: aquí solo los nombres."""
    if not isinstance(entradas, list):
        return []
    nombres: list[str] = []
    for entrada in entradas:
        if isinstance(entrada, dict):
            nombre = str(entrada.get("name") or "").strip()
            if nombre:
                nombres.append(nombre)
    return nombres


def _poster_url(main_picture: object) -> str | None:
    """`large` si está, si no `medium`. Hay entradas sin ninguna de las dos."""
    if not isinstance(main_picture, dict):
        return None
    for clave in ("large", "medium"):
        url = main_picture.get(clave)
        if isinstance(url, str) and url.strip():
            return url
    return None


def _anio_de_start_date(start_date: object) -> int | None:
    """Año de un "2023-09-29". Es el fallback cuando no hay start_season."""
    texto = str(start_date or "").strip()
    if len(texto) < 4 or not texto[:4].isdigit():
        return None
    return int(texto[:4])


def flatten_item(node: dict, ranking: dict) -> dict:
    """Un ítem del ranking con los campos que nos interesan, del JSON de MAL al nuestro.

    `average_episode_duration` viene en segundos y sale truncado a minutos, así que
    1470 s son 24 min. `alternative_titles.en` y `.ja` salen como `title_english` y
    `title_japanese`. El año sale de `start_season` y, si no existe, del año de
    `start_date`.
    """
    alternativos = node.get("alternative_titles")
    if not isinstance(alternativos, dict):
        alternativos = {}
    temporada = node.get("start_season")
    if not isinstance(temporada, dict):
        temporada = {}
    return {
        "id": node.get("id"),
        "title": node.get("title"),
        "title_english": alternativos.get("en"),
        "title_japanese": alternativos.get("ja"),
        "media_type": node.get("media_type"),
        "num_episodes": node.get("num_episodes"),
        "status": node.get("status"),
        "duration_minutes": _int(node.get("average_episode_duration")) // 60,
        "rating": node.get("rating"),
        "mean": _float(node.get("mean")),
        "rank": _int(node.get("rank") or ranking.get("rank")),
        "popularity": _int(node.get("popularity")),
        "num_scoring_users": _int(node.get("num_scoring_users")),
        "synopsis": str(node.get("synopsis") or "").strip(),
        "year": temporada.get("year") or _anio_de_start_date(node.get("start_date")),
        "season": temporada.get("season"),
        "start_date": node.get("start_date"),
        "genres": _nombres(node.get("genres")),
        "studios": _nombres(node.get("studios")),
        "poster_url": _poster_url(node.get("main_picture")),
    }


def flatten(pages: list[dict], min_scoring_users: int) -> tuple[list[dict], Counter]:
    """Aplana las páginas, deduplica por id y aplica los filtros de calidad."""
    stats: Counter = Counter()
    vistos: set[int] = set()
    anime: list[dict] = []

    for page in pages:
        for entrada in page.get("data") or []:
            if not isinstance(entrada, dict):
                continue
            stats["vistos"] += 1
            node = entrada.get("node")
            if not isinstance(node, dict):
                stats["sin_node"] += 1
                continue
            anime_id = node.get("id")
            if not isinstance(anime_id, int):
                stats["sin_id"] += 1
                continue
            if anime_id in vistos:
                stats["duplicados"] += 1
                continue
            vistos.add(anime_id)

            if not es_tipo_permitido(node.get("media_type")):
                stats["fuera_de_tipo"] += 1
                continue

            ranking = entrada.get("ranking")
            item = flatten_item(node, ranking if isinstance(ranking, dict) else {})
            if not item["synopsis"]:
                stats["sin_sinopsis"] += 1
                continue
            if item["num_scoring_users"] < min_scoring_users:
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
    """Redondea N ítems hacia arriba a páginas completas de PAGE_LIMIT."""
    return -(-limit // PAGE_LIMIT) * PAGE_LIMIT


def _validar(args: argparse.Namespace) -> str | None:
    """Mensaje del flag que no cuadra, o None si todo está bien."""
    if args.max_items < 1:
        return f"--max-items tiene que ser >= 1 (recibido {args.max_items})"
    if args.min_scoring_users < 0:
        return (
            f"--min-scoring-users no puede ser negativo "
            f"(recibido {args.min_scoring_users})"
        )
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
    client_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Ejecuta la descarga y el aplanado. Devuelve el código de salida."""
    problema = _validar(args)
    if problema:
        LOGGER.error("%s", problema)
        return EXIT_CONFIG

    key = settings.mal_client_id if client_id is None else client_id
    if not key:
        LOGGER.error(
            "Falta MAL_CLIENT_ID: ponla en backend/.env antes de descargar "
            "(no hace falta el client secret)"
        )
        return EXIT_CONFIG

    max_items = args.max_items
    if args.limit is not None:
        max_items = min(max_items, items_de_paginas(args.limit))

    directory = pages_dir(raw_dir)
    directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Descargando en %s", directory)

    counters: Counter = Counter()
    try:
        with build_client(key, transport=transport) as client:
            fetch_pages(client, directory, max_items, counters)
    except MalAuthError as exc:
        LOGGER.error("%s", exc)
        return EXIT_CONFIG

    paginas = read_pages(directory)
    if not paginas:
        LOGGER.error("No hay páginas en %s: no se genera %s", directory, FLAT_FILENAME)
        return EXIT_PROBLEMA

    anime, stats = flatten(paginas, args.min_scoring_users)
    if anime:
        write_json_atomic(raw_dir / FLAT_FILENAME, anime)
    else:
        LOGGER.warning(
            "Ningún anime pasa los filtros: %s no se toca, prueba con "
            "--min-scoring-users menor",
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
        description="Descarga el ranking de anime de MyAnimeList como JSON crudo (una vez).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help=f"Máximo de ítems a descargar (default: {DEFAULT_MAX_ITEMS}).",
    )
    parser.add_argument(
        "--min-scoring-users",
        type=int,
        default=DEFAULT_MIN_SCORING_USERS,
        help=(
            "Votos mínimos para conservar un anime. Se aplica al aplanar, no al "
            f"descargar (default: {DEFAULT_MIN_SCORING_USERS})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Descarga solo N ítems, redondeados a páginas completas de "
            f"{PAGE_LIMIT}. Para pruebas rápidas."
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
