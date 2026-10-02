"""Unifica el catálogo crudo de TMDB y MAL en un único parquet.

Lee (rutas por defecto, relativas a este fichero):

    data/raw/tmdb/movie/*.json   -> 5000 películas (detalle de TMDB en es-ES)
    data/raw/tmdb/tv/*.json      -> 1163 series (detalle de TMDB en es-ES)
    data/raw/mal/anime.json      -> 3467 anime ya aplanados por scripts/fetch_mal.py

Escribe `data/processed/catalog.parquet`, una fila por título. Uso:

    uv run python -m scripts.build_catalog
    uv run python -m scripts.build_catalog --out data/processed/catalogo.parquet

Lo que está aquí está-contrastado con los archivos reales, no de memoria. Lo que
más sorprende al comparar movie y tv de TMDB:

* la serie **no tiene `runtime`**. Usa `episode_run_time`, que es una lista y
  además viene vacía en 525 de 1163. Para esas el fallback es
  `last_episode_to_air.runtime`, con lo cual solo quedan 5 series sin duración por
  episodio.
* los `keywords` cuelgan de una clave distinta según el tipo: en movie de
  `keywords.keywords` y en tv de `keywords.results`. No comparten clave.
* los títulos van en `title`/`original_title` en movie y en `name`/`original_name`
  en tv, con `release_date` frente a `first_air_date`.
* las series traen géneros combinados que hay que abrir: `Sci-Fi & Fantasy`,
  `Action & Adventure` y `War & Politics`.

Sobre `media_type`: el contrato con el frontend es `movie | tv | anime` y no se
negocia. Lo que dice MAL de sí mismo va en `mal_type` (`tv`, `movie`, `ova`, `ona`),
que es informativo y no entra en `embed_text`. Los 3467 anime salen siempre como
`anime`, aunque MAL los tenga como `tv`, `movie`, `ova` u `ona`, y ningún título de
TMDB sale como `anime`.

Sobre las escalas: `rating` viene de `vote_average` (TMDB, sobre los votos de TMDB)
y de `mean` (MAL, sobre los votos de MAL). Son dos notas distintas y no son
comparables entre filas. `vote_count` es `vote_count` en TMDB y `num_scoring_users`
en MAL.

La duración va partida en dos columnas, y no en una, porque antes `runtime` mezclaba
dos unidades: en las películas de TMDB era la duración total y en las series de TMDB y en
el anime de MAL eran los minutos por episodio. Con una sola columna, un umbral del
tipo "menos de 30 minutos" eliminaba el 0.6 % de las películas y el 76.1 % del anime, y en
el anime por el motivo equivocado. Así que ahora:

* `runtime_total`: duración total de la obra, en minutos. Solo la llevan las películas de
  TMDB (`media_type` `movie`) y el anime que MAL clasifica como `movie`.
* `runtime_episode`: minutos por episodio. Los llevan las series de TMDB y el anime
  con `mal_type` `tv`, `ova` u `ona`.

Una fila nunca tiene las dos con valor: según de dónde venga, el valor va a una o a
la otra, y se queda en la que le corresponde según lo que dice MAL de sí mismo. El
cero se traduce a nulo en los tres casos, como antes.

Exclusiones, en este orden y con el primer motivo que aplica:

  1. `adult` a True en TMDB. Hoy descarta 0: los 6163 archivos vienen a False.
  2. Sin estrenar, según MAPA_STATUS_TMDB: `In Production`, `Planned` y
     `Post Production`. También 0 hoy. `Canceled` va a `other` y sí se queda.
  3. Sin sinopsis tras limpiar texto. También 0 hoy: los fetch ya rellenaron los
     vacíos con el original en inglés.
  4. Anime de MAL con `rating` `rx`. También 0 hoy: no hay ni uno en anime.json.
  5. Animación japonesa duplicada con MAL: `original_language` `ja` **y** con
     `Animación` entre los géneros. Aquí sí: 233 títulos (75 películas y 158 series).
     Estrechar por el género es a propósito. Hay 24 títulos más con `ja` que no son
     animación (Godzilla, Harakiri, Battle Royale, Alice in Borderland...) y MAL no
     los cubre, así que se quedan en el catálogo.

Todo lo que no se puede mapear al vocabulario canónico de géneros acaba en
`keywords` en vez de inventarse una equivalencia, y se reporta con su conteo. Lo
mismo con los géneros de MAL: solo se mapean los 18 equivalentes obvios, así que
`genres` de un anime puede salir vacía y los 58 géneros restantes en `keywords`.

Un aviso más sobre MAL: en `anime.json` los géneros son una lista de cadenas, no
una lista de `{"id": ..., "name": ...}` como en TMDB. Es un detalle que se lleva
por delante los 3467 anime sin que salte ningún error, así que _generos_mal los
lee aparte y cuenta los que vengan en otro formato.

Códigos de salida: 0 si todo fue bien, 1 si no hay datos, si alguna fuente no
entrega filas, o si hay ids duplicados. Que falte la duración de alguna fila no es
un error: con los datos de hoy son 9 (5 series y 4 anime), que se quedan con las dos
columnas de duración a nulo, y sale en el resumen.

Nota: hay que ejecutarlo con cwd=backend/, porque `app/config.py` lee el `.env` con
una ruta relativa al cwd (aunque este script no usa ninguna credencial).
"""

import argparse
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TMDB_DIR = BACKEND_DIR / "data" / "raw" / "tmdb"
DEFAULT_MAL_JSON = BACKEND_DIR / "data" / "raw" / "mal" / "anime.json"
DEFAULT_OUT = BACKEND_DIR / "data" / "processed" / "catalog.parquet"

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
ANIME_LANGUAGE = "ja"
ANIME_OVERVIEW_LANG = "en"
DEFAULT_OVERVIEW_LANG = "es"
MAX_KEYWORDS_EMBED = 15

PLANTILLA_EMBED_TEXT = (
    "{title}. Tipo: {media_type}. Géneros: {generos}. Temas: {temas}. {overview}"
)

# Vocabulario canónico. Es cerrado: lo que no está aquí no se inventa, va a keywords.
GENEROS_CANONICOS = (
    "Acción",
    "Aventura",
    "Animación",
    "Comedia",
    "Crimen",
    "Documental",
    "Drama",
    "Familia",
    "Fantasía",
    "Historia",
    "Terror",
    "Música",
    "Misterio",
    "Romance",
    "Ciencia ficción",
    "Suspense",
    "Bélica",
    "Western",
    "Slice of Life",
    "Deportes",
)


def _identidad(nombres: tuple[str, ...]) -> dict[str, list[str]]:
    """Cada nombre canónico se mapea a sí mismo, con su propia lista.

    Ojo con dict.fromkeys: shares la MISMA lista entre todas las claves.
    """
    return {nombre: [nombre] for nombre in nombres}


# Los géneros de movie ya vienen en español y 18 de 19 caen 1:1 en el vocabulario.
MAPA_TMDB_PELICULA: dict[str, list[str] | None] = dict(_identidad(GENEROS_CANONICOS))
MAPA_TMDB_PELICULA["Película de TV"] = None  # sin equivalente: va a keywords

# En tv los combinaciones se abren y los cinco sin equivalente van a keywords.
MAPA_TMDB_SERIE: dict[str, list[str] | None] = dict(_identidad(GENEROS_CANONICOS))
MAPA_TMDB_SERIE.update(
    {
        "Sci-Fi & Fantasy": ["Ciencia ficción", "Fantasía"],  # 472
        "Action & Adventure": ["Acción", "Aventura"],  # 394
        "War & Politics": ["Bélica"],  # 22
        "Kids": None,  # 73
        "Soap": None,  # 28
        "Reality": None,  # 8
        "News": None,  # 3
        "Talk": None,  # 3
    }
)

# De los 76 géneros de MAL solo estos 18 son equivalentes obvios. Los otros 58
# (Supernatural, Mecha, Isekai, Shounen, Ecchi...) se quedan como keywords en vez
# de forzarles un género canónico que no les corresponde.
MAPA_MAL: dict[str, list[str] | None] = {
    "Action": ["Acción"],
    "Adventure": ["Aventura"],
    "Comedy": ["Comedia"],
    "Drama": ["Drama"],
    "Fantasy": ["Fantasía"],
    "Romance": ["Romance"],
    "Sci-Fi": ["Ciencia ficción"],
    "Mystery": ["Misterio"],
    "Historical": ["Historia"],
    "Military": ["Bélica"],
    "Suspense": ["Suspense"],
    "Horror": ["Terror"],
    "Music": ["Música"],
    "Slice of Life": ["Slice of Life"],
    "Sports": ["Deportes"],
    "Team Sports": ["Deportes"],
    "Combat Sports": ["Deportes"],
    "Racing": ["Deportes"],
}

# `Canceled` se queda: es una serie que ya existed, no una que no se haya estrenado.
MAPA_STATUS_TMDB = {
    "Released": "ended",
    "Ended": "ended",
    "Returning Series": "ongoing",
    "In Production": "upcoming",
    "Planned": "upcoming",
    "Post Production": "upcoming",
    "Pilot": "upcoming",
    "Canceled": "other",
}

MAPA_STATUS_MAL = {
    "finished_airing": "ended",
    "currently_airing": "ongoing",
    "not_yet_aired": "upcoming",
}

# El anime trae un único campo de duración, `duration_minutes`, y son minutos por
# episodio salvo cuando MAL lo clasifica como `movie`: ahí es la duración total de la
# película. Estos dos nombres son lo que decide a cuál de las dos columnas va el valor.
MAL_TYPE_PELICULA = "movie"
MAL_TYPE_POR_EPISODIO = ("tv", "ova", "ona")

SUFIJOS_ATRIBUCION = (
    re.compile(r"\[Written by MAL Rewrite\]", re.IGNORECASE),
    re.compile(r"\(Source:[^)]*\)", re.IGNORECASE),
)

ORDEN_COLUMNAS = (
    "id",
    "source",
    "media_type",
    "mal_type",
    "title",
    "original_title",
    "year",
    "overview",
    "overview_lang",
    "genres",
    "keywords",
    "rating",
    "vote_count",
    "popularity",
    "runtime_total",
    "runtime_episode",
    "episodes",
    "seasons",
    "status",
    "age_rating",
    "language",
    "poster_url",
    "embed_text",
)

# id, source, media_type, title, status, language y embed_text nunca son nulos. Que
# pyarrow lo compruebe al escribir es una invariante gratis.
ESQUEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("media_type", pa.string(), nullable=False),
        # Lo que dice MAL de sí mismo (tv/movie/ova/ona). Solo informativo: `media_type`
        # es el campo que consume el frontend, y para MAL siempre vale "anime".
        pa.field("mal_type", pa.string(), nullable=True),
        pa.field("title", pa.string(), nullable=False),
        pa.field("original_title", pa.string(), nullable=True),
        pa.field("year", pa.int64(), nullable=True),
        pa.field("overview", pa.string(), nullable=True),
        pa.field("overview_lang", pa.string(), nullable=True),
        pa.field("genres", pa.list_(pa.string()), nullable=False),
        pa.field("keywords", pa.list_(pa.string()), nullable=False),
        pa.field("rating", pa.float64(), nullable=True),
        pa.field("vote_count", pa.int64(), nullable=True),
        pa.field("popularity", pa.float64(), nullable=True),
        # Duración partida por unidad, para no mezclar minutos totales con minutos por
        # episodio en la misma columna. Nunca hay las dos con valor a la vez: `total`
        # es de películas de TMDB y de anime con mal_type `movie`; `episode` es de series
        # de TMDB y de anime con mal_type `tv`, `ova` u `ona`.
        pa.field("runtime_total", pa.int64(), nullable=True),
        pa.field("runtime_episode", pa.int64(), nullable=True),
        pa.field("episodes", pa.int64(), nullable=True),
        pa.field("seasons", pa.int64(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
        pa.field("age_rating", pa.string(), nullable=True),
        pa.field("language", pa.string(), nullable=False),
        pa.field("poster_url", pa.string(), nullable=True),
        pa.field("embed_text", pa.string(), nullable=False),
    ]
)

COLUMNAS_INT = (
    "year",
    "vote_count",
    "runtime_total",
    "runtime_episode",
    "episodes",
    "seasons",
)
COLUMNAS_FLOAT = ("rating", "popularity")

EXIT_OK = 0
EXIT_PROBLEMA = 1

LOGGER = logging.getLogger("build_catalog")

MOTIVOS = (
    "tmdb_adulto",
    "tmdb_no_estrenado",
    "tmdb_sinopsis_vacia",
    "mal_adulto",
    "mal_no_estrenado",
    "mal_sinopsis_vacia",
    "tmdb_animacion_japonesa",
)

ETIQUETAS_MOTIVO = {
    "tmdb_adulto": "TMDB adult=true",
    "tmdb_no_estrenado": "TMDB sin estrenar",
    "tmdb_sinopsis_vacia": "TMDB sin sinopsis",
    "mal_adulto": "MAL rating=rx",
    "mal_no_estrenado": "MAL sin airstrear",
    "mal_sinopsis_vacia": "MAL sin sinopsis",
    "tmdb_animacion_japonesa": "TMDB animación ja (duplicada en MAL)",
}


def normalizar_texto(texto: object) -> str:
    """Colapsa espacios y saltos de línea, y quita los bordes."""
    return re.sub(r"\s+", " ", str(texto or "")).strip()


def limpiar_sinopsis_mal(synopsis: object) -> str:
    """Quita los pies de MAL y colapsa espacios.

    Vienen al final de la sinopsis: `[Written by MAL Rewrite]` y `(Source: AniDB)`
    o su variante `(Source: MangaHelpers, edited)`.
    """
    texto = str(synopsis or "")
    for patron in SUFIJOS_ATRIBUCION:
        texto = patron.sub(" ", texto)
    return normalizar_texto(texto)


def _int_positivo(valor: object) -> int | None:
    return valor if isinstance(valor, int) and valor > 0 else None


def _float(valor: object) -> float | None:
    return float(valor) if isinstance(valor, (int, float)) else None


def anio_desde_fecha(fecha: object) -> int | None:
    """`1998-05-20` o el ISO de MAL con hora -> 1998."""
    coincidencia = re.match(r"(\d{4})", str(fecha or ""))
    return int(coincidencia.group(1)) if coincidencia else None


def mapear_generos(
    generos_fuente: list[str],
    mapa: dict[str, list[str] | None],
    no_mapeados: Counter | None = None,
) -> tuple[list[str], list[str]]:
    """Reparte los géneros del origen entre el vocabulario canónico y keywords.

    Cada valor del mapa es la lista de géneros canónicos a los que pasa el de
    origen. Lo que no está en `mapa` (o está con valor None) se devuelve aparte
    para que el llamante lo añada a `keywords`, y se anota en `no_mapeados`.
    """
    canonicos: list[str] = []
    sin_mapear: list[str] = []
    for genero in generos_fuente:
        destino = mapa.get(genero)
        if destino is None:
            sin_mapear.append(genero)
            if no_mapeados is not None:
                no_mapeados[genero] += 1
            continue
        for canonico in destino:
            if canonico not in canonicos:
                canonicos.append(canonico)
    return canonicos, sin_mapear


def _nombres_tmdb(entradas: object) -> list[str]:
    """Saca los `name` de una lista de objetos de TMDB."""
    if not isinstance(entradas, list):
        return []
    nombres = []
    for entrada in entradas:
        if isinstance(entrada, dict):
            nombre = normalizar_texto(entrada.get("name"))
            if nombre:
                nombres.append(nombre)
    return nombres


def _generos_mal(entradas: object, ctx: "Contexto") -> list[str]:
    """Los géneros de MAL son una lista de cadenas, no de objetos.

    Cuidado: `anime.json` no los guarda como `{"id": ..., "name": ...}` como hace
    TMDB, así que leerlos con `_nombres_tmdb` los pierde en silencio. Si algún día
    empiezan a venir como objetos, se cuenta aquí para que se note.
    """
    if not isinstance(entradas, list):
        return []
    nombres = []
    for entrada in entradas:
        if isinstance(entrada, str):
            nombre = normalizar_texto(entrada)
            if nombre:
                nombres.append(nombre)
        else:
            ctx.generos_mal_inesperados += 1
    return nombres


def keywords_tmdb(detalle: dict, clave: str) -> list[str]:
    """Lee `keywords` de TMDB con la clave que usa ese tipo.

    Movie anida la lista en `keywords.keywords` y tv en `keywords.results`, así que
    la clave va por parámetro en vez de asumir una común.
    """
    bloque = detalle.get("keywords")
    if not isinstance(bloque, dict):
        return []
    return _nombres_tmdb(bloque.get(clave) or [])


def estado_tmdb(estado: object) -> str:
    return MAPA_STATUS_TMDB.get(str(estado or "").strip(), "other")


def estado_mal(estado: object) -> str:
    return MAPA_STATUS_MAL.get(str(estado or "").strip(), "other")


def runtime_pelicula(detalle: dict) -> int | None:
    """Duración total de la película, en minutos."""
    return _int_positivo(detalle.get("runtime"))


def runtime_serie(detalle: dict) -> int | None:
    """Minutos por episodio.

    `episode_run_time` es una lista y viene vacía en 525 de las 1163 series. Para esas
    se mira `last_episode_to_air.runtime`, que es el único campo de duración por
    episodio que trae el detalle de una serie; con él quedan 5 series sin dato.
    """
    por_episodio = detalle.get("episode_run_time")
    if isinstance(por_episodio, list) and por_episodio:
        valor = _int_positivo(por_episodio[0])
        if valor is not None:
            return valor
    for clave in ("last_episode_to_air", "next_episode_to_air"):
        valor = _int_positivo((detalle.get(clave) or {}).get("runtime"))
        if valor is not None:
            return valor
    return None


def runtime_anime(item: dict) -> int | None:
    """`average_episode_duration` ya vieneminutes, 0 si no lo sabían."""
    return _int_positivo(item.get("duration_minutes"))


def poster_tmdb(detalle: dict) -> str | None:
    """`poster_path` ya empieza por `/`, así que se concatena tal cual."""
    path = detalle.get("poster_path")
    return f"{TMDB_IMAGE_BASE}{path}" if path else None


def overview_lang_tmdb(detalle: dict) -> str:
    """`overview_lang` solo está en los detalles que vinieron en inglés."""
    return detalle.get("overview_lang") or DEFAULT_OVERVIEW_LANG


def es_animacion_japonesa(detalle: dict) -> bool:
    """Duplicado con MAL: `ja` **y** con `Animación` entre los géneros.

    El genres es lo que estrecha la regla. Hay 24 títulos con `ja` que no son
    animación y que MAL no cubre, así que filtrar solo por idioma los perdería.
    """
    if detalle.get("original_language") != ANIME_LANGUAGE:
        return False
    generos = _nombres_tmdb(detalle.get("genres"))
    return "Animación" in generos


def construir_embed_text(
    title: str,
    media_type: str,
    genres: list[str],
    keywords: list[str],
    overview: str,
    max_keywords: int = MAX_KEYWORDS_EMBED,
) -> str:
    """Rellena PLANTILLA_EMBED_TEXT, con los primeros `max_keywords` keywords."""
    return normalizar_texto(
        PLANTILLA_EMBED_TEXT.format(
            title=title,
            media_type=media_type,
            generos=", ".join(genres) or "sin género",
            temas=", ".join(keywords[:max_keywords]) or "sin temas",
            overview=overview or "sin sinopsis",
        )
    )


class Contexto:
    """Lo que las tres fuentes necesitan ir contando mientras leen."""

    def __init__(self) -> None:
        self.descartes: Counter = Counter()
        self.generos_no_mapeados: Counter = Counter()
        self.generos_mal_inesperados = 0
        self.ejemplos_ja: list[str] = []

    def descartar(self, motivo: str, titulo: str = "") -> None:
        self.descartes[motivo] += 1
        if motivo == "tmdb_animacion_japonesa" and len(self.ejemplos_ja) < 10:
            self.ejemplos_ja.append(titulo)

    @property
    def total_descartado(self) -> int:
        return sum(self.descartes.values())


def fila_base(
    identificador: str,
    source: str,
    media_type: str,
    mal_type: str | None,
    title: str,
    original_title: str | None,
    year: int | None,
    overview: str,
    overview_lang: str,
    genres: list[str],
    keywords: list[str],
    rating: float | None,
    vote_count: int | None,
    popularity: float | None,
    runtime_total: int | None,
    runtime_episode: int | None,
    episodes: int | None,
    seasons: int | None,
    status: str,
    age_rating: str | None,
    language: str,
    poster_url: str | None,
) -> dict:
    """Arma la fila con el embed_text ya calculado."""
    return {
        "id": identificador,
        "source": source,
        "media_type": media_type,
        "mal_type": mal_type,
        "title": title,
        "original_title": original_title,
        "year": year,
        "overview": overview,
        "overview_lang": overview_lang,
        "genres": genres,
        "keywords": keywords,
        "rating": rating,
        "vote_count": vote_count,
        "popularity": popularity,
        "runtime_total": runtime_total,
        "runtime_episode": runtime_episode,
        "episodes": episodes,
        "seasons": seasons,
        "status": status,
        "age_rating": age_rating,
        "language": language,
        "poster_url": poster_url,
        "embed_text": construir_embed_text(
            title, media_type, genres, keywords, overview
        ),
    }


def fila_de_pelicula(detalle: dict, ctx: Contexto) -> dict | None:
    """Convierte un detalle de movie de TMDB en fila, o None si se descarta."""
    titulo = normalizar_texto(detalle.get("title"))
    if detalle.get("adult"):
        ctx.descartar("tmdb_adulto", titulo)
        return None
    if estado_tmdb(detalle.get("status")) == "upcoming":
        ctx.descartar("tmdb_no_estrenado", titulo)
        return None

    overview = normalizar_texto(detalle.get("overview"))
    if not overview:
        ctx.descartar("tmdb_sinopsis_vacia", titulo)
        return None
    if es_animacion_japonesa(detalle):
        ctx.descartar("tmdb_animacion_japonesa", titulo)
        return None

    genres, sin_mapear = mapear_generos(
        _nombres_tmdb(detalle.get("genres")), MAPA_TMDB_PELICULA, ctx.generos_no_mapeados
    )
    keywords = keywords_tmdb(detalle, "keywords") + sin_mapear

    return fila_base(
        identificador=f"tmdb-movie-{detalle['id']}",
        source="tmdb",
        media_type="movie",
        mal_type=None,
        title=titulo,
        original_title=detalle.get("original_title") or None,
        year=anio_desde_fecha(detalle.get("release_date")),
        overview=overview,
        overview_lang=overview_lang_tmdb(detalle),
        genres=genres,
        keywords=keywords,
        rating=_float(detalle.get("vote_average")),
        vote_count=detalle.get("vote_count") if isinstance(detalle.get("vote_count"), int) else None,
        popularity=_float(detalle.get("popularity")),
        runtime_total=runtime_pelicula(detalle),
        runtime_episode=None,  # una película no tiene minutos por episodio
        episodes=None,
        seasons=None,
        status=estado_tmdb(detalle.get("status")),
        age_rating=None,  # TMDB no trae una clasificación por edad en estos detalles
        language=detalle.get("original_language") or "",
        poster_url=poster_tmdb(detalle),
    )


def fila_de_serie(detalle: dict, ctx: Contexto) -> dict | None:
    """Convierte un detalle de tv de TMDB en fila, o None si se descarta."""
    titulo = normalizar_texto(detalle.get("name"))
    if detalle.get("adult"):
        ctx.descartar("tmdb_adulto", titulo)
        return None
    if estado_tmdb(detalle.get("status")) == "upcoming":
        ctx.descartar("tmdb_no_estrenado", titulo)
        return None

    overview = normalizar_texto(detalle.get("overview"))
    if not overview:
        ctx.descartar("tmdb_sinopsis_vacia", titulo)
        return None
    if es_animacion_japonesa(detalle):
        ctx.descartar("tmdb_animacion_japonesa", titulo)
        return None

    genres, sin_mapear = mapear_generos(
        _nombres_tmdb(detalle.get("genres")), MAPA_TMDB_SERIE, ctx.generos_no_mapeados
    )
    keywords = keywords_tmdb(detalle, "results") + sin_mapear

    return fila_base(
        identificador=f"tmdb-tv-{detalle['id']}",
        source="tmdb",
        media_type="tv",
        mal_type=None,
        title=titulo,
        original_title=detalle.get("original_name") or None,
        year=anio_desde_fecha(detalle.get("first_air_date")),
        overview=overview,
        overview_lang=overview_lang_tmdb(detalle),
        genres=genres,
        keywords=keywords,
        rating=_float(detalle.get("vote_average")),
        vote_count=detalle.get("vote_count") if isinstance(detalle.get("vote_count"), int) else None,
        popularity=_float(detalle.get("popularity")),
        runtime_total=None,  # una serie no tiene duración total aquí
        runtime_episode=runtime_serie(detalle),
        episodes=_int_positivo(detalle.get("number_of_episodes")),
        seasons=_int_positivo(detalle.get("number_of_seasons")),
        status=estado_tmdb(detalle.get("status")),
        age_rating=None,
        language=detalle.get("original_language") or "",
        poster_url=poster_tmdb(detalle),
    )


def fila_de_anime(item: dict, ctx: Contexto) -> dict | None:
    """Convierte un item de MAL aplanado en fila, o None si se descarta."""
    titulo = normalizar_texto(item.get("title"))
    if item.get("rating") == "rx":
        ctx.descartar("mal_adulto", titulo)
        return None
    if estado_mal(item.get("status")) == "upcoming":
        ctx.descartar("mal_no_estrenado", titulo)
        return None

    overview = limpiar_sinopsis_mal(item.get("synopsis"))
    if not overview:
        ctx.descartar("mal_sinopsis_vacia", titulo)
        return None

    # MAL no tiene género "Animation", así que casi ningún anime conserva
    # `Animación` en `genres`. No se lo cuelga a mano: `media_type` ya lo distingue.
    genres, sin_mapear = mapear_generos(
        _generos_mal(item.get("genres"), ctx), MAPA_MAL, ctx.generos_no_mapeados
    )

    mal_type = item.get("media_type") or None
    # `duration_minutes` son minutos por episodio, salvo si MAL llamó `movie` a la
    # obra: ahí es la duración total. El mismo campo va a una columna u a la otra.
    # Los tipos raros que MAL añada en el futuro (`special`, `music`) caen en
    # `runtime_episode`, que es el lado más cercano de los dos.
    duracion = runtime_anime(item)
    es_pelicula = mal_type == MAL_TYPE_PELICULA

    return fila_base(
        identificador=f"mal-anime-{item['id']}",
        source="mal",
        media_type="anime",
        mal_type=mal_type,
        title=titulo,
        original_title=item.get("title_japanese") or item.get("title_english") or None,
        year=anio_desde_fecha(item.get("year") or item.get("start_date")),
        overview=overview,
        overview_lang=ANIME_OVERVIEW_LANG,  # anime.json solo trae sinopsis en inglés
        genres=genres,
        keywords=sin_mapear,
        rating=_float(item.get("mean")),
        vote_count=item.get("num_scoring_users") if isinstance(item.get("num_scoring_users"), int) else None,
        popularity=_float(item.get("popularity")),
        runtime_total=duracion if es_pelicula else None,
        runtime_episode=None if es_pelicula else duracion,
        episodes=_int_positivo(item.get("num_episodes")),
        seasons=None,  # MAL no lleva temporadas
        status=estado_mal(item.get("status")),
        age_rating=item.get("rating") or None,
        language=ANIME_LANGUAGE,
        poster_url=item.get("poster_url") or None,
    )


def _leer_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def leer_tmdb(
    tmdb_dir: Path, subcarpeta: str, constructor, ctx: Contexto
) -> list[dict]:
    """Lee todos los detalles de `<tmdb_dir>/<subcarpeta>` y los pasa por `constructor`."""
    directorio = tmdb_dir / subcarpeta
    filas: list[dict] = []
    if not directorio.is_dir():
        LOGGER.warning("No existe el directorio %s", directorio)
        return filas

    for path in sorted(directorio.glob("*.json")):
        detalle = _leer_json(path)
        if detalle is None or "id" not in detalle:
            LOGGER.warning("Se ignora %s: no se lee como detalle de TMDB", path.name)
            continue
        fila = constructor(detalle, ctx)
        if fila is not None:
            filas.append(fila)
    return filas


def leer_mal(ruta: Path, ctx: Contexto) -> list[dict]:
    """Lee anime.json (la lista aplanada por scripts/fetch_mal.py)."""
    if not ruta.is_file():
        LOGGER.warning("No existe %s", ruta)
        return []

    try:
        items = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("No se puede leer %s: %s", ruta, error)
        return []
    if not isinstance(items, list):
        LOGGER.warning("%s no contiene una lista de anime", ruta)
        return []

    filas = []
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            continue
        fila = fila_de_anime(item, ctx)
        if fila is not None:
            filas.append(fila)
    return filas


def comprobar_ids(filas: list[dict]) -> None:
    """Falla con el detalle si algún id está repetido, antes de escribir nada."""
    vistos: Counter = Counter(fila["id"] for fila in filas)
    duplicados = {id_: n for id_, n in vistos.items() if n > 1}
    if duplicados:
        detalle = ", ".join(f"{id_} x{n}" for id_, n in sorted(duplicados.items()))
        raise ValueError(f"Hay {len(duplicados)} ids duplicados en el catálogo: {detalle}")


def construir_dataframe(filas: list[dict]) -> pd.DataFrame:
    """DataFrame con el orden de columnas y los tipos del esquema."""
    df = pd.DataFrame(filas, columns=list(ORDEN_COLUMNAS))
    for columna in COLUMNAS_INT:
        df[columna] = pd.array(df[columna], dtype="Int64")
    for columna in COLUMNAS_FLOAT:
        df[columna] = pd.array(df[columna], dtype="Float64")
    return df


def escribir_parquet_atomico(df: pd.DataFrame, salida: Path) -> None:
    """Escribe en `<salida>.tmp` y renombra, para no dejar un parquet a medias."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = salida.with_name(f"{salida.name}.tmp")
    df.to_parquet(temporal, schema=ESQUEMA, index=False)
    os.replace(temporal, salida)


def resumen(df: pd.DataFrame, ctx: Contexto) -> str:
    """Filas por tipo, descartes con motivo, nulos por columna y unas ejemplos."""
    def conteos(columna: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in df[columna].value_counts().sort_index().items()}

    sin_keywords = df["keywords"].map(len) == 0
    con_generos = df["genres"].map(len) > 0
    total = len(df)
    lineas = [
        f"Filas: {total}",
        f"  por media_type: {conteos('media_type')}",
        f"  por source: {conteos('source')}",
        f"  sin keywords: {int(sin_keywords.sum())} ({sin_keywords.mean() * 100:.1f}%)",
        f"  con genres: {int(con_generos.sum())} ({con_generos.mean() * 100:.1f}%)",
        f"Descartes: {ctx.total_descartado}",
    ]
    for motivo in MOTIVOS:
        if ctx.descartes[motivo]:
            lineas.append(f"  {ETIQUETAS_MOTIVO[motivo]}: {ctx.descartes[motivo]}")

    nulos = df.isna().sum()
    total = len(df)
    lineas.append("Nulos por columna:")
    for columna in ORDEN_COLUMNAS:
        cantidad = int(nulos[columna])
        if cantidad == 0:
            continue
        lineas.append(f"  {columna}: {cantidad} ({cantidad / total * 100:.1f}%)")

    ejemplos = df.head(3)
    lineas.append("Ejemplos:")
    for fila in ejemplos.to_dict("records"):
        duracion = fila["runtime_total"]
        unidad = "min en total"
        if duracion is None or pd.isna(duracion):
            duracion = fila["runtime_episode"]
            unidad = "min por episodio"
        lineas.append(
            f"  {fila['id']} | {fila['title']} | {fila['year']} | "
            f"{fila['genres']} | {duracion} {unidad}"
        )
    return "\n".join(lineas)


def _reportar_generos_no_mapeados(ctx: Contexto) -> None:
    if not ctx.generos_no_mapeados:
        return
    total = sum(ctx.generos_no_mapeados.values())
    LOGGER.info(
        "Géneros sin equivalente canónico (a keywords): %d distintos, %d apariciones",
        len(ctx.generos_no_mapeados),
        total,
    )
    for genero, cuenta in ctx.generos_no_mapeados.most_common():
        LOGGER.info("  %-24s %d", genero, cuenta)


def run(tmdb_dir: Path, mal_json: Path, salida: Path) -> int:
    ctx = Contexto()

    LOGGER.info("Leyendo películas de %s", tmdb_dir / "movie")
    peliculas = leer_tmdb(tmdb_dir, "movie", fila_de_pelicula, ctx)
    LOGGER.info("  %d películas tras exclusiones", len(peliculas))

    LOGGER.info("Leyendo series de %s", tmdb_dir / "tv")
    series = leer_tmdb(tmdb_dir, "tv", fila_de_serie, ctx)
    LOGGER.info("  %d series tras exclusiones", len(series))

    LOGGER.info("Leyendo anime de %s", mal_json)
    anime = leer_mal(mal_json, ctx)
    LOGGER.info("  %d anime tras exclusiones", len(anime))

    filas = peliculas + series + anime
    if not filas:
        LOGGER.error("No hay filas: revisa las rutas de entrada")
        return EXIT_PROBLEMA
    if not (peliculas and series and anime):
        LOGGER.error(
            "Alguna fuente no entregó filas (películas=%d series=%d anime=%d)",
            len(peliculas),
            len(series),
            len(anime),
        )
        return EXIT_PROBLEMA

    try:
        comprobar_ids(filas)
    except ValueError as error:
        LOGGER.error("%s", error)
        return EXIT_PROBLEMA

    df = construir_dataframe(filas)
    escribir_parquet_atomico(df, salida)

    sin_duracion = df[df["runtime_total"].isna() & df["runtime_episode"].isna()]
    LOGGER.info("Sin duración en ninguna de las dos columnas: %d", len(sin_duracion))
    if len(sin_duracion):
        LOGGER.info(
            "  ejemplos: %s",
            ", ".join(
                f"{r['id']} ({r['title']})"
                for r in sin_duracion.head(10).to_dict("records")
            ),
        )
    if ctx.ejemplos_ja:
        LOGGER.info(
            "Ejemplos de animación ja excluida (%d de %d): %s",
            len(ctx.ejemplos_ja),
            ctx.descartes["tmdb_animacion_japonesa"],
            ", ".join(ctx.ejemplos_ja),
        )
    if ctx.generos_mal_inesperados:
        LOGGER.warning(
            "%d géneros de MAL venían en un formato que no es una cadena; revisa anime.json",
            ctx.generos_mal_inesperados,
        )
    _reportar_generos_no_mapeados(ctx)

    LOGGER.info("Escrito %s", salida)
    LOGGER.info("\n%s", resumen(df, ctx))
    return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unifica el catálogo crudo de TMDB y MAL en un parquet.",
    )
    parser.add_argument(
        "--tmdb-dir",
        type=Path,
        default=DEFAULT_TMDB_DIR,
        help=f"Directorio con movie/ y tv/ (default: {DEFAULT_TMDB_DIR}).",
    )
    parser.add_argument(
        "--mal-json",
        type=Path,
        default=DEFAULT_MAL_JSON,
        help=f"anime.json de MAL (default: {DEFAULT_MAL_JSON}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Ruta del parquet de salida (default: {DEFAULT_OUT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(args.tmdb_dir, args.mal_json, args.out)


if __name__ == "__main__":
    raise SystemExit(main())