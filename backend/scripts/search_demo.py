"""Prueba de humo del índice: busca en el catálogo por una frase y enseña qué sale.

Uso:

    uv run python -m scripts.search_demo "un planeta de ladrones con ciudades colgantes"
    uv run python -m scripts.search_demo "no quiero anime, quiero algo real" --top 5
    uv run python -m scripts.search_demo --sin-indice-info
    uv run python -m scripts.search_demo "algo real y corto" --sin-anime

No es el endpoint `/recommend`: este script no lo suplente, solo deja comprobar que lo
que construyó `scripts.build_index` se puede abrir, que la pregunta llega al modelo con
el prefijo correcto y que lo que sale parece sensato. Si un día esto devuelve puras
tonterías, lo primero que hay que mirar es si la consulta entró con `query: ` y los
documentos con `passage: `, porque son prefijos distintos por diseño y mezclarlos degrada el resultado sin dar ningún error.

La lógica está en `app/services/search.py` y en `app/services/vector_store.py`, no aquí.
Este fichero es la línea de comandos: parsea argumentos, llama al servicio y formatea. La
razón es que el servicio tiene que ser el único camino para abrir el índice, y si el
script se guardara una copia de la apertura aparecen dos caminos que divergen sin que
nadie lo note. Aquí ya no se abre nada: se usa lo mismo que usaría el router.

Muestra el `score` reescalado que ve el usuario y, en la misma línea, la `distance` cruda
de ChromaDB. La distancia va de 0 a 2 y cuanto más baja mejor (0 es idéntico, 1 es
ortogonal, 2 es el opuesto); con los vectores normalizados equivale a `1 - similitud`. El
score reescalado no es comparable entre consultas, la distancia sí.

Sobre negaciones: este script no las entiende y no lo disimula. "no quiero anime" con
`--sin-anime` funciona porque el filtro es exacto; sin ese flag, la palabra "no" acerca
el vector a títulos que empiezan por "No" y sale anime. La limpieza con regex se
descartó a propósito. Ver la nota del módulo de búsqueda.

Códigos de salida: 0 si se buscó bien, 1 si no hay índice o no hay resultados.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.services.embedder import PREFIJO_CONSULTA, Embedder, cargar_embedder
from app.services.search import FiltrosBusqueda, ResultadoBusqueda, buscar
from app.services.vector_store import (
    DEFAULT_DIR_INDICE,
    IndiceNoDisponibleError,
    abrir_coleccion,
    comprobar_modelo,
    leer_index_info,
)

EXIT_OK = 0
EXIT_PROBLEMA = 1

LOGGER = logging.getLogger(__name__)

TOP_POR_DEFECTO = 5


def formatear_resultado(posicion: int, res: ResultadoBusqueda) -> str:
    """Una línea por resultado con lo justo para juzgar si la búsqueda acierta."""
    detalles = [res.media_type]
    if res.year is not None:
        detalles.append(str(res.year))
    if res.genres:
        detalles.append("|".join(res.genres))

    cruda = f"d={res.distance:.4f}" if res.distance is not None else "d=?"
    # El score reescalado sale arriba, como en la app, y la distancia cruda debajo: sin
    # la cruda no hay forma de saber si un 87% viene de un 0.85 bueno o de un 0.30 malísimo.
    cabecera = (
        f"{posicion:>2}. {res.title}  [{' · '.join(detalles)}]"
        f"  score={res.score * 100:.0f}%  {cruda}"
    )
    return cabecera


def buscar_una(
    consulta: str,
    top: int,
    media_types: list[str],
    dir_indice: Path,
    embedder: Embedder | None = None,
) -> int:
    """Busca una frase y la imprime. Devuelve un código de salida."""
    try:
        coleccion = abrir_coleccion(dir_indice)
    except IndiceNoDisponibleError as error:
        LOGGER.error("%s", error)
        LOGGER.error("Construye el índice antes con: uv run python -m scripts.build_index")
        return EXIT_PROBLEMA

    total = coleccion.count()
    if total == 0:
        LOGGER.error("La colección está vacía")
        return EXIT_PROBLEMA

    LOGGER.info("Índice con %d documentos; buscando %d", total, top)
    LOGGER.info("Consulta: %s%s", PREFIJO_CONSULTA, consulta)

    if embedder is not None:
        info = leer_index_info(dir_indice)
        comprobar_modelo(info, embedder)

    resultados = buscar(
        consulta,
        filtros=FiltrosBusqueda(media_types=media_types),
        top_k=top,
        embedder=embedder,
        coleccion=coleccion,
    )

    if not resultados:
        LOGGER.warning("Sin resultados")
        return EXIT_OK

    print()
    for posicion, res in enumerate(resultados, start=1):
        print(formatear_resultado(posicion, res))
    return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca en el índice del catálogo y muestra los resultados.",
    )
    parser.add_argument(
        "consulta",
        nargs="+",
        help="Texto por el que buscar. Con varias palabras se busca cada una por separado.",
    )
    parser.add_argument(
        "--dir-indice",
        type=Path,
        default=DEFAULT_DIR_INDICE,
        help=f"Directorio del índice (default: {DEFAULT_DIR_INDICE}).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_POR_DEFECTO,
        help=f"Cuántos resultados mostrar (default: {TOP_POR_DEFECTO}).",
    )
    parser.add_argument(
        "--sin-anime",
        action="store_true",
        help="Quita anime de los tipos buscados (filtro exacto de media_type).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.top <= 0:
        LOGGER.error("--top tiene que ser un entero positivo")
        return EXIT_PROBLEMA

    # El modelo se carga una vez y se reutiliza para todas las consultas. Cargar por
    # consulta son unos segundos por búsqueda, y con varias consultas se notaba: el
    # peso estaba en la caché pero SentenceTransformer lo reconstruye cada vez.
    try:
        embedder = cargar_embedder()
    except Exception as error:  # noqa: BLE001
        LOGGER.error("No se pudo cargar el modelo de embeddings: %s", error)
        return EXIT_PROBLEMA

    media_types = ["movie", "tv"] if args.sin_anime else ["movie", "tv", "anime"]

    codigo = EXIT_OK
    for consulta in args.consulta:
        if len(args.consulta) > 1:
            print(f"\n=== {consulta} ===")
        resultado = buscar_una(
            consulta, args.top, media_types, args.dir_indice, embedder=embedder
        )
        if resultado != EXIT_OK:
            codigo = resultado
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())