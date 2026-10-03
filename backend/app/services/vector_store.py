"""Abre el índice de ChromaDB y lo deja listo para buscar, una vez y cacheado.

La razón de que este módulo exista es el coste, no la comodidad. Abrir la colección es
barato, pero cargar el modelo de embeddings son varios segundos, y si eso pasa por
request el endpoint no sirve para nada. Con `--reload` de uvicorn el proceso se
reinicia en cada cambio y vuelve a cargar, así que aquí no hay atajo posible: son unos
segundos por arranque, no por consulta.

Decisión consciente: el estado es global de módulo con `functools.lru_cache`. Es lo más
simple que funciona con FastAPI y no necesita dependencias ni eventos de arranque. El
coste es que el estado es **por proceso**: si algún día se lanzan varios workers
(`--workers 4`), cada uno carga su propia copia del modelo y su propio handle de
ChromaDB. Con cuatro workers serían cuatro copias de ~700 MB de pesos y cuatro procesos
leyendo el mismo `chroma.sqlite3`. Hoy no es un problema porque se corre un solo worker,
pero si algún día se escala hay que sustituir el global por algo compartido antes, no
después.

Las tres funciones de aquí salen de `scripts/search_demo.py`, que era el único sitio que
sabía abrir el índice. Están aquí porque van a tener dos usuarios, el script y
`app/services/search.py`, y dos caminos para abrir el mismo índice divergen sin que nadie
lo note hasta que un día dan resultados distintos.

`leer_index_info` no es obligatorio: el índice sirve sin él. Lo único que se pierde es el
aviso cuando el modelo con el que se busca no es el del que se indexó, que es un fallo
silencioso y de los que dan resultados raro en vez de un error.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any

from app.services.embedder import Embedder

LOGGER = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DIR_INDICE = BACKEND_DIR / "data" / "index"

NOMBRE_COLECCION = "catalog"
NOMBRE_INFO = "index_info.json"


class IndiceNoDisponibleError(RuntimeError):
    """El índice no se puede abrir, o no está donde se esperaba.

    Es una excepción propia y no un `FileNotFoundError` porque a quien la de nada le
    sirve distinguir "falta el directorio" de "falta la colección": lo que necesita saber
    es que no hay índice, y el mensaje dice ya cuál de las dos cosas pasó. El router
    la traduce a un 503 con un texto accionable.
    """


def ruta_indice_por_defecto() -> Path:
    """El directorio del índice completo.

    No sale de settings a propósito: `settings` no tiene ningún campo para esto y
    meterlo en `app/config.py` con un valor por defecto sería una variable configurable
    que nadie configura. Si algún día hace falta moverlo de verdad, se añade entonces.
    """
    return DEFAULT_DIR_INDICE


def leer_index_info(dir_indice: Path) -> dict[str, Any] | None:
    """Lee `index_info.json` si está. No es obligatorio: el índice sirve sin él."""
    ruta = dir_indice / NOMBRE_INFO
    if not ruta.exists():
        LOGGER.warning(
            "No hay %s; no se puede comprobar el modelo con el que se indexó", ruta
        )
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("No se pudo leer %s: %s", ruta, error)
        return None


def abrir_coleccion(dir_indice: Path) -> Any:
    """Abre la colección del catálogo. Lanza `IndiceNoDisponibleError` si no está.

    "Si no está" cubre las tres formas de no estar: que falte el directorio, que no
    tenga la colección y que chroma no sepa abrirlo (directorio corrupto, permisos).
    Las tres se traducen a la misma excepción porque para quien las sufre da igual
    cuál sea: no puede buscar.

    No va cacheada a propósito: es la función cruda que usan los tests y `search_demo`,
    y el caché vive en `obtener_coleccion`.
    """
    import chromadb

    if not dir_indice.exists():
        raise IndiceNoDisponibleError(f"no existe el directorio del índice {dir_indice}")
    try:
        cliente = chromadb.PersistentClient(path=str(dir_indice))
        nombres = {c.name for c in cliente.list_collections()}
    except Exception as error:
        # No se limita a `ChromaError` porque chroma falla de formas distintas
        # según el momento: con una base corrupta el primer intento lanza
        # `InternalError` ("file is not a database"), pero deja el sistema a
        # medias y el siguiente lanza su propio `AttributeError` (comprobado con
        # una base de mentira). Cualquiera de las dos significa lo mismo aquí.
        raise IndiceNoDisponibleError(
            f"el índice {dir_indice} no se puede abrir: {type(error).__name__}: {error}"
        ) from error
    if NOMBRE_COLECCION not in nombres:
        raise IndiceNoDisponibleError(
            f"el índice {dir_indice} no tiene la colección {NOMBRE_COLECCION}"
        )
    return cliente.get_collection(NOMBRE_COLECCION)


def comprobar_modelo(info: dict[str, Any] | None, embedder: Embedder) -> None:
    """Avisa si se está buscando con un modelo distinto al del indexado.

    No lanza. Es una comprobación de consistencia, no de validez: el índice existe y se
    puede consultar, solo que los números no significarán nada. Cortar la búsqueda aquí
    sería peor que avisar.
    """
    if not info:
        return
    guardado = info.get("modelo")
    if guardado and guardado != embedder.nombre:
        LOGGER.warning(
            "El índice se construyó con %s y se está buscando con %s; "
            "las distancias no son comparables",
            guardado,
            embedder.nombre,
        )


@functools.lru_cache(maxsize=4)
def _coleccion_cacheada(dir_indice_resuelto: str) -> Any:
    """Abre la colección una vez por directorio. No llames directamente: ver arriba."""
    coleccion = abrir_coleccion(Path(dir_indice_resuelto))
    LOGGER.info("Colección %s abierta en %s", NOMBRE_COLECCION, dir_indice_resuelto)
    return coleccion


def obtener_coleccion(dir_indice: Path | None = None) -> Any:
    """Devuelve la colección, reutilizando la ya abierta si el directorio es el mismo."""
    objetivo = dir_indice or ruta_indice_por_defecto()
    return _coleccion_cacheada(str(objetivo.resolve()))


def limpiar_cache() -> None:
    """Olvida las colecciones abiertas. Para tests y para el script."""
    _coleccion_cacheada.cache_clear()