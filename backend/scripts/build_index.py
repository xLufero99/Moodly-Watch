"""Construye el índice de búsqueda vectorial del catálogo en ChromaDB.

Lee `data/processed/catalog.parquet`, lo convierte en embeddings con
`intfloat/multilingual-e5-small` y los guarda en una colección persistente de ChromaDB
en `data/index/`. Uso:

    uv run python -m scripts.build_index
    uv run python -m scripts.build_index --limit 200
    uv run python -m scripts.build_index --out data/index/prueba

Lo que hay que tener presente antes de tocar nada:

* **El prefijo va en el lado del documento, no en el del texto guardado.** Lo que se
  mete en `document` es el `embed_text` tal cual sale del parquet, sin tocar. El
  `passage: ` lo aplica `app/services/embedder.py` por dentro, solo sobre lo que se le
  pasa al modelo. Si el prefijo acabara dentro de `document` en el índice, las
  sinopsis del catálogo quedarían contaminadas y una búsqueda por texto plano dejaría
  de encontrar nada. El módulo compartido lo pone, y el índice guarda el texto limpio.

* **`None` no vale como valor de metadato en ChromaDB.** Esto no es de leer el
  código, es de haberlo probado contra la versión instalada (1.5.9): la capa de
  validación en Rust contesta `Cannot convert Python object to MetadataValue` y
  revienta el `add` entero. El validador de Python de arriba sí admite `None`, así que
  el fallo parece contradictorio y no lo es: gana el de abajo. Por eso una columna con
  nulos se guarda **sin esa clave**, y no con un centinela inventado. Al filtrar, la
  ausencia de la clave significa "no lo sé", que es la verdad, y un `-1` mentiría.

* **Las listas también son un problema, y aquí hay dos.** ChromaDB acepta listas en
  los metadatos, pero no acepta la lista vacía y sí exige que sean homogéneas. Los
  géneros salen del parquet como `ndarray`, y hay 58 géneros de MAL sin equivalente
  canónico, así que hay filas con la lista vacía de verdad: pasarla tal cual aborta la
  carga. Por eso `genres` va como texto unido por `|`. Es más plano que una lista
  estructurada, pero se filtra con `where={"genres": {"$contains": "Terror"}}` sin
  problemas y evita la lista vacía.

* **Lo que sale del parquet no es de los tipos que ChromaDB acepta.** Se comprobó que
  `isinstance(np.int64(5), int)` es `False` en Python, así que el `year` o el
  `vote_count` de una fila se rechazan tal cual si se pasan sin convertir. `_escalar`
  los baja a `int`/`float`/`str` nativos antes de dárselos a ChromaDB. Los `float64` de
  numpy sí son subclase de `float` y pasarían, pero se bajan igual para no depender de
  ese detalle.

Sobre los percentiles, dos cosas que salen de medir y no de suponer:

* Se calculan **dentro de cada `source`** y no sobre el catálogo entero. `vote_count`
  va de 500 a 3 115 304, y la mediana de MAL (37 765) es casi 19 veces la de TMDB
  (2018). Un percentil global mezclaría las dos escalas y mediría sobre todo "de qué
  fuente es el título" en lugar de "qué popular es".

* **No se aplica `log` a `vote_count`, y es a propósito.** El primer impulso es
  loguear porque la distribución está muy desviada, y parece lo sensato, pero un
  percentil se calcula por rangos y el logaritmo es monótono: no reordena nada. Se
  comprobó sobre el catálogo real y el percentil de `log10(vote_count)` sale
  idéntico al del valor crudo, en las dos fuentes y en todos los tramos. Si algún día
  esto deja de ser un percentil y pasa a ser un umbral absoluto (`más de 10 000
  votos`), entonces el log sí cambia el resultado y hay que aplicarlo ahí.

* Los empates se resuelten con `method="average"`, que es además el valor por defecto
  de pandas. Es el que mantiene la propiedad que importa: dos títulos con el mismo
  `vote_count` en la misma fuente reciben el mismo `popularity_pct`. Con `method="min"`
  o `"max"` siguen siendo iguales entre sí pero saltan a un extremo del tramo, y
  `first` directamente los reparte (0.5 y 0.75 para dos titles empatados), que es lo
  que no queremos.

El truncamiento no se corrige, se reporta. `intfloat/multilingual-e5-small` tiene
`max_seq_length` 512 y hay sinopsis de MAL que lo pasan de largo. Se mide con el
tokenizer real, sobre el texto que de verdad entra al modelo (con el prefijo puesto, que
también cuenta), y se dicen cuántos son y cuáles son sin tocar `embed_text`: acortar
una sinopsis para que quepa es cambiar el catálogo, y eso es otra decisión.

Por defecto **reconstruye**: borra la colección y la vuelve a crear. Si se añadiera a
una colección existente, un título que cambie de sinopsis se quedaría con el vector
viejo al lado del nuevo y el índice no sabría cuál de los dos responde. Empezar de cero
es más lento una vez y correcto siempre.

Códigos de salida: 0 si todo fue bien, 1 si no hay filas, si falta el parquet o si no
se pudo escribir el índice.

Nota: hay que ejecutarlo con cwd=backend/, porque `app/config.py` lee el `.env` con
una ruta relativa al cwd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.embedder import (
    PREFIJO_CONSULTA,
    PREFIJO_DOCUMENTO,
    Embedder,
    cargar_embedder,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARCQUET = BACKEND_DIR / "data" / "processed" / "catalog.parquet"
DEFAULT_DIR_INDICE = BACKEND_DIR / "data" / "index"

NOMBRE_COLECCION = "catalog"
NOMBRE_INFO = "index_info.json"
METRICA = "cosine"
LOTE_CHROMA = 5000
LOTE_TOKENIZADOR = 512
EJEMPLOS_TRUNCAMIENTO = 5

EXIT_OK = 0
EXIT_PROBLEMA = 1

LOGGER = logging.getLogger(__name__)

# Escalares que se copian tal cual a los metadatos, en el orden en que se recorren.
CAMPOS_ESCALARES = (
    "media_type",
    "source",
    "mal_type",
    "title",
    "year",
    "rating",
    "vote_count",
    "popularity",
    "runtime_total",
    "runtime_episode",
    "episodes",
    "seasons",
    "status",
    "language",
    "poster_url",
)

# Campos que en el parquet son listas y en los metadatos son texto con "|".
CAMPOS_LISTA = ("genres",)

# Los percentiles no están en el parquet: los calcula este script.
CAMPOS_PERCENTIL = ("rating_pct", "popularity_pct")


# --------------------------------------------------------------------------- #
# Tipos: de lo que sale de pandas a lo que acepta ChromaDB
# --------------------------------------------------------------------------- #


def _nulo(valor: Any) -> bool:
    """True si el valor es None, NaN o pd.NA.

    `pd.isna` devuelve un array cuando le pasas una lista, y un array no se puede
    convertir a bool, así que el except no es hipotético: se dispara con `genres`.
    """
    if valor is None:
        return True
    try:
        resultado = pd.isna(valor)
    except (TypeError, ValueError):
        return False
    if isinstance(resultado, bool):
        return resultado
    return False


def _escalar(valor: Any) -> Any:
    """Baja un valor de pandas/numpy a un tipo nativo, o None si no hay dato.

    El orden de las comprobaciones no es arbitrario. `bool` va antes que `int` porque
    `isinstance(True, int)` es `True` en Python, y sin ese orden un booleano saldría
    como 1 o 0 en vez de como True o False. Y los enteros nativos van explícitos porque
    no son subclase de `np.integer`: un `int` de Python que se colara sin convertir
    acababa en el `str()` de abajo y se guardaba como texto, que es lo que hacía este
    función antes de que lo cogiera un test.
    """
    if _nulo(valor):
        return None
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        numero = float(valor)
        return None if math.isnan(numero) else numero
    if isinstance(valor, (str, np.str_)):
        texto = str(valor)
        return texto if texto else None
    return str(valor)


def _lista_a_texto(valor: Any) -> str | None:
    """Convierte una lista de géneros en texto unido por `|`.

    Devuelve None si está vacía, y no un texto vacío: una lista vacía es exactamente
    el caso que ChromaDB rechaza, y guardarla como "" además significaría "género
    vacío" en vez de "sin géneros".
    """
    if _nulo(valor):
        return None
    if isinstance(valor, (str, np.str_)):
        texto = str(valor).strip()
        return texto or None
    if isinstance(valor, (list, tuple, set, np.ndarray)):
        partes = [str(x).strip() for x in valor]
        unido = "|".join(p for p in partes if p)
        return unido or None
    return None


def metadatos_de_fila(fila: Mapping[str, Any]) -> dict[str, Any]:
    """Metadatos de una fila del catálogo, sin nulos y con tipos nativos.

    Las claves cuyo valor sale vacío no aparecen en el dict. Es lo que permite que
    `add` no reviente y lo que hace que la ausencia de una clave signifique "no lo sé".
    """
    meta: dict[str, Any] = {}
    for campo in CAMPOS_ESCALARES:
        valor = _escalar(fila.get(campo))
        if valor is not None:
            meta[campo] = valor
    for campo in CAMPOS_LISTA:
        valor = _lista_a_texto(fila.get(campo))
        if valor is not None:
            meta[campo] = valor
    for campo in CAMPOS_PERCENTIL:
        valor = _escalar(fila.get(campo))
        if valor is not None:
            meta[campo] = valor
    return meta


# --------------------------------------------------------------------------- #
# Percentiles
# --------------------------------------------------------------------------- #


def anadir_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Añade `rating_pct` y `popularity_pct`, ambos dentro de cada `source`.

    Se usa `method="average"` a propósito. Es el valor por defecto de pandas, pero se
    pone explícito porque es justo lo que garantiza que dos títulos con el mismo
    `vote_count` en la misma fuente salgan con el mismo `popularity_pct`, y eso es una
    de las cosas que cubren los tests.

    Sobre el logaritmo, que aquí no se aplica: el percentil va por rangos y el log es
    monótono, así que `log10(vote_count)` y `vote_count` dan exactamente el mismo
    percentil. Se comprobó sobre el catálogo entero, en las dos fuentes. El log solo
    empezaría a importar si esto deixara de ser un percentil y fuera un umbral
    absoluto, y ese día hay que aplicarlo en el sitio nuevo.
    """
    df = df.copy()
    df["rating_pct"] = df.groupby("source")["rating"].rank(pct=True, method="average")
    df["popularity_pct"] = df.groupby("source")["vote_count"].rank(
        pct=True, method="average"
    )
    return df


# --------------------------------------------------------------------------- #
# Selección de filas
# --------------------------------------------------------------------------- #


def limitar(df: pd.DataFrame, limite: int | None) -> pd.DataFrame:
    """Se queda con `limite` filas, repartidas proporcionalmente entre media_type.

    Proporcional y no equitable a propósito: un reparto por mitades daría el mismo
    peso a las series (10.7 % del catálogo) que a las películas (52.4 %), y el tiempo que
    tarda en construir el índice dejaría de servir para extrapolar al catálogo completo.

    El redondeo es por resto mayor: las cuotas enteras suman menos que `limite`, y el
    sobrante va a los grupos con mayor parte decimal. Así salen exactamente `limite`
    filas sin pasarse de las que hay de cada tipo.
    """
    if limite is None or limite >= len(df):
        return df

    conteo = df["media_type"].value_counts()
    exacto = conteo / conteo.sum() * limite
    cuotas = np.floor(exacto).astype(int)
    sobran = limite - int(cuotas.sum())
    if sobran > 0:
        resto = exacto - np.floor(exacto)
        for tipo in resto.sort_values(ascending=False).index[:sobran]:
            cuotas[tipo] += 1

    # sort_values("id") para que el recorte sea determinista y no dependa del orden
    # que casualmente tuviera el parquet.
    ordenado = df.sort_values("id")
    partes = []
    for tipo, cuota in cuotas.items():
        sub = ordenado[ordenado["media_type"] == tipo]
        partes.append(sub.head(min(int(cuota), len(sub))))
    return pd.concat(partes).sort_index()


# --------------------------------------------------------------------------- #
# Truncamiento
# --------------------------------------------------------------------------- #


def medir_truncamiento(
    tokenizador: Any,
    ids: Sequence[str],
    textos: Sequence[str],
    max_seq_length: int,
    ejemplos: int = EJEMPLOS_TRUNCAMIENTO,
) -> dict[str, Any] | None:
    """Cuenta cuántos textos pasan de `max_seq_length`, sin truncar nada.

    Se tokeniza con `truncation=False` para medir la longitud real. Los `textos` que se
    le pasan ya llevan el prefijo `passage: ` puesto, porque es lo que entra al modelo
    de verdad: el prefijo también gasta tokens y medir sin él daría un número
    optimista.
    """
    if tokenizador is None:
        return None

    largos: list[int] = []
    for inicio in range(0, len(textos), LOTE_TOKENIZADOR):
        lote = textos[inicio : inicio + LOTE_TOKENIZADOR]
        codigos = tokenizador(
            list(lote),
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )["input_ids"]
        largos.extend(len(c) for c in codigos)

    if not largos:
        return None

    exceden = [i for i, largo in enumerate(largos) if largo > max_seq_length]
    return {
        "total": len(largos),
        "exceden": len(exceden),
        "porcentaje": 100.0 * len(exceden) / len(largos),
        "maximo_observado": max(largos),
        "max_seq_length": max_seq_length,
        "ejemplos": [{"id": str(ids[i]), "tokens": largos[i]} for i in exceden[:ejemplos]],
    }


def _preparar_textos(embedder: Embedder, textos: Sequence[str]) -> list[str]:
    """Los textos tal como los verá el modelo, prefijo incluido, para poder medirlos."""
    from app.services.embedder import documento_para_indexar

    return [documento_para_indexar(t) for t in textos]


# --------------------------------------------------------------------------- #
# Incrustar
# --------------------------------------------------------------------------- #


def incrustar_con_progreso(
    embedder: Embedder,
    textos: Sequence[str],
    batch_size: int,
    log_cada: int,
    inicio: float,
) -> list[list[float]]:
    """Incrusta en trozos, para poder ir diciendo por dónde va y a qué ritmo.

    Trocear aquí y no dentro del embedder es lo que permite el log intermedio: el
    `encode` entero solo avisa cuando ha terminado, y con 9397 documentos eso son
    minutos de pantalla en blanco.
    """
    total = len(textos)
    por_trozo = max(1, batch_size * max(1, log_cada))
    vectores: list[list[float]] = []

    for inicio_trozo in range(0, total, por_trozo):
        trozo = textos[inicio_trozo : inicio_trozo + por_trozo]
        vectores.extend(embedder.incrustar_documentos(list(trozo), batch_size=batch_size))
        hechos = inicio_trozo + len(trozo)
        transcurrido = time.monotonic() - inicio
        LOGGER.info(
            "  %d/%d documentos (%.1f docs/s)",
            hechos,
            total,
            hechos / transcurrido if transcurrido > 0 else 0.0,
        )
    return vectores


# --------------------------------------------------------------------------- #
# Índice
# --------------------------------------------------------------------------- #


def sha256_de_archivo(ruta: Path) -> str:
    """SHA-256 del parquet, para poder saber con qué catálogo se hizo el índice."""
    digest = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def escribir_index_info(
    ruta: Path,
    embedder: Embedder,
    documentos: int,
    parquet: Path,
    limite: int | None,
    metadatos_extra: dict[str, Any] | None = None,
) -> None:
    """Deja en `index_info.json` con qué modelo y con qué catálogo se hizo el índice.

    Sin esto, dentro de un mes no hay forma de saber si un resultado raro viene de un
    cambio en el modelo, del prefijo o de que el catálogo se rehízo. El SHA-256 del
    parquet es lo que ata el índice a su entrada concreta.
    """
    info: dict[str, Any] = {
        "coleccion": NOMBRE_COLECCION,
        "modelo": embedder.nombre,
        "dimension": embedder.dimension,
        "prefijo_documento": PREFIJO_DOCUMENTO,
        "prefijo_consulta": PREFIJO_CONSULTA,
        "max_seq_length": embedder.max_seq_length,
        "metrica": METRICA,
        "documentos": documentos,
        "limit": limite,
        "parquet": str(parquet),
        "parquet_sha256": sha256_de_archivo(parquet),
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if metadatos_extra:
        info.update(metadatos_extra)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporal, ruta)


def construir_coleccion(dir_indice: Path) -> Any:
    """Borra la colección anterior si existe y devuelve una nueva, en coseno."""
    import chromadb

    dir_indice.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(dir_indice))

    existentes = {c.name for c in cliente.list_collections()}
    if NOMBRE_COLECCION in existentes:
        LOGGER.info("La colección %s ya existe: se borra y se reconstruye", NOMBRE_COLECCION)
        cliente.delete_collection(NOMBRE_COLECCION)

    return cliente.create_collection(
        NOMBRE_COLECCION,
        metadata={"hnsw:space": METRICA},
    )


def volcar_en_coleccion(coleccion: Any, filas: list[Mapping[str, Any]]) -> None:
    """Mete las filas en la colección por lotes, con los metadatos ya saneados."""
    for inicio in range(0, len(filas), LOTE_CHROMA):
        trozo = filas[inicio : inicio + LOTE_CHROMA]
        coleccion.add(
            ids=[str(f["id"]) for f in trozo],
            documents=[f["embed_text"] for f in trozo],
            embeddings=[f["_embedding"] for f in trozo],
            metadatas=[metadatos_de_fila(f) for f in trozo],
        )


def _limpiar_filas(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Filas como dicts, con los ids y los textos ya como cadenas limpias."""
    filas: list[dict[str, Any]] = []
    for registro in df.to_dict("records"):
        fila = dict(registro)
        fila["id"] = str(fila["id"])
        texto = fila.get("embed_text")
        fila["embed_text"] = "" if texto is None else str(texto)
        filas.append(fila)
    return filas


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #


def run(
    parquet: Path,
    dir_indice: Path,
    limite: int | None = None,
    batch_size: int = 32,
    log_cada: int = 10,
    embedder: Embedder | None = None,
    limpiar: bool = False,
) -> int:
    if not parquet.exists():
        LOGGER.error("No existe el parquet %s; ejecuta antes scripts.build_catalog", parquet)
        return EXIT_PROBLEMA

    LOGGER.info("Leyendo %s", parquet)
    df = pd.read_parquet(parquet)
    if df.empty:
        LOGGER.error("El parquet %s no tiene filas", parquet)
        return EXIT_PROBLEMA

    LOGGER.info("  %d filas, %d columnas", len(df), len(df.columns))

    # Los percentiles se calculan sobre el catálogo entero, antes de limitar. Si se
    # calcularan después, un --limit 200 daría percentiles de un catálogo de 200 filas,
    # que no significan nada.
    df = anadir_percentiles(df)

    if limite is not None:
        total = len(df)
        df = limitar(df, limite)
        LOGGER.info(
            "Limitando a %d de %d filas: %s",
            len(df),
            total,
            dict(df["media_type"].value_counts().sort_index()),
        )

    if embedder is None:
        embedder = cargar_embedder()

    ids = df["id"].astype(str).tolist()
    textos = df["embed_text"].astype(str).tolist()

    informe = medir_truncamiento(
        embedder.tokenizador,
        ids,
        _preparar_textos(embedder, textos),
        embedder.max_seq_length,
    )
    if informe is None:
        LOGGER.warning(
            "El embedder no expone tokenizer: no se ha medido el truncamiento"
        )
    else:
        LOGGER.info(
            "Truncamiento: %d de %d textos pasan de %d tokens (%.1f %%)",
            informe["exceden"],
            informe["total"],
            informe["max_seq_length"],
            informe["porcentaje"],
        )
        if informe["exceden"]:
            LOGGER.info(
                "  longitud máxima observada: %d tokens",
                informe["maximo_observado"],
            )
            LOGGER.info(
                "  ejemplos: %s",
                ", ".join(f"{e['id']} ({e['tokens']})" for e in informe["ejemplos"]),
            )

    inicio = time.monotonic()
    vectores = incrustar_con_progreso(embedder, textos, batch_size, log_cada, inicio)
    tiempo_embeddings = time.monotonic() - inicio
    LOGGER.info(
        "Embeddings listos en %.1fs (%.1f docs/s)",
        tiempo_embeddings,
        len(textos) / tiempo_embeddings if tiempo_embeddings > 0 else 0.0,
    )

    if limpiar and dir_indice.exists():
        LOGGER.info("Borrando %s antes de reconstruir", dir_indice)
        shutil.rmtree(dir_indice)

    filas = _limpiar_filas(df)
    for fila, vector in zip(filas, vectores, strict=True):
        fila["_embedding"] = vector

    LOGGER.info("Escribiendo la colección %s en %s", NOMBRE_COLECCION, dir_indice)
    coleccion = construir_coleccion(dir_indice)
    volcar_en_coleccion(coleccion, filas)
    tiempo_indice = time.monotonic() - inicio

    escribir_index_info(
        dir_indice / NOMBRE_INFO,
        embedder,
        len(filas),
        parquet,
        limite,
        {
            "truncamiento": informe,
            "segundos_embeddings": round(tiempo_embeddings, 2),
            "segundos_total": round(tiempo_indice, 2),
            "docs_por_segundo": round(
                len(filas) / tiempo_indice if tiempo_indice > 0 else 0.0, 2
            ),
        },
    )

    LOGGER.info(
        "Índice listo: %d documentos en %.1fs (%.1f docs/s)",
        len(filas),
        tiempo_indice,
        len(filas) / tiempo_indice if tiempo_indice > 0 else 0.0,
    )
    LOGGER.info("Escrito %s", dir_indice / NOMBRE_INFO)
    return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el índice de búsqueda vectorial del catálogo.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARCQUET,
        help=f"Ruta del parquet de entrada (default: {DEFAULT_PARCQUET}).",
    )
    parser.add_argument(
        "--dir-indice",
        type=Path,
        default=DEFAULT_DIR_INDICE,
        help=f"Directorio del índice (default: {DEFAULT_DIR_INDICE}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Construye solo N documentos, repartidos por media_type (para probar).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de lote para el modelo (default: 32).",
    )
    parser.add_argument(
        "--log-cada",
        type=int,
        default=10,
        help="Log de progreso cada N lotes (default: 10).",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        help="Borra el directorio del índice entero antes de construir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.limit is not None and args.limit <= 0:
        LOGGER.error("--limit tiene que ser un entero positivo")
        return EXIT_PROBLEMA
    return run(
        parquet=args.parquet,
        dir_indice=args.dir_indice,
        limite=args.limit,
        batch_size=args.batch_size,
        log_cada=args.log_cada,
        limpiar=args.limpiar,
    )


if __name__ == "__main__":
    raise SystemExit(main())