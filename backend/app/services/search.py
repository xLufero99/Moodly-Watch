"""Busca en el índice del catálogo y devuelve los resultados con la forma del contrato.

Este módulo es la traducción de una frase escrita por alguien a una lista de
recomendaciones. No es el endpoint: no sabe de HTTP, no genera la `explanation` (eso es
Groq, en otro paso) y no sabe de prompts. Recibe texto, busca y devuelve estructuras con
los mismos nombres de campo que `app.models.schemas.Recommendation`, para que cablear el
router más adelante sea solo importar.

Lo que hay que tener presente al leerlo:

**El filtro de `media_types` va a ChromaDB, no a Python.** Con `where`, el filtro se
aplica durante la búsqueda y se devuelven exactamente los `n_results` válidos. Filtrando
en Python habría que pedir más de los necesarios y podar, sin saber de antemano cuántos
de los primeros veinte son anime. Medido: `"no quiero anime"` devolvía 3 anime de 5, o
sea que el filtro tiene que actuar antes de elegir el top, no después.

**Los `liked_ids` sí se podan en Python**, porque no son un filtro sino una exclusión, y
ChromaDB no tiene un `$nin` cómodo para ids. De ahí la sobrepetición de `top_k * 2`.

**El `score` es relativo a la respuesta, no una confianza absoluta.** Las distancias de
este modelo están comprimidas: medido, el top-1 daba 0.1368 y el top-5 0.1435, o sea
similitudes 0.8632 y 0.8565. Con el `score` crudo los cinco resultados salían a 0.86 y
el frontend, que hace `Math.round(score * 100)`, pintaba "86%" cinco veces: cero
información. Así que se reescala con min-max sobre lo devuelto. La consecuencia que se
acepta a sabiendas: **el primer resultado siempre marca 100%**, porque es el techo del
reescalado, no porque sea una coincidencia perfecta. La `distance` cruda se conserva en
cada resultado como `distance` para poder mirar debajo cuando el top-5 salga raro.

**Nada aquí entiende negaciones.** `"no quiero anime, algo real y corto"` devuelve anime,
porque un vector cercano no es un vector opuesto. Limpiar la frase con regex se
descartó a propósito: aguanta "no quiero anime" y se rompe con "fuera de anime" o "algo
que no sea anime", y un arreglo que funciona en el caso que pruebas y falla en el
siguiente es peor que ninguno. Lo que sí hace el filtro es exacto mientras el tipo de
medio venga marcado en la petición. La interpretación del texto libre es trabajo del
parser de intención, que necesita el LLM y va después.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.embedder import Embedder, cargar_embedder
from app.services.vector_store import (
    IndiceNoDisponibleError,
    comprobar_modelo,
    leer_index_info,
    obtener_coleccion,
    ruta_indice_por_defecto,
)

LOGGER = logging.getLogger(__name__)

# Factor por el que se piden más resultados de los necesarios para poder descartar los
# ya vistos. Con 2 basta salvo que `liked_ids` traiga media pantalla; el tope de 50
# corta el caso extremo de un usuario que ha marcado casi todo.
SOBREPETICION = 2
TOPE_SOBREPETICION = 50

# Rango del score. No llega a 0 porque el peor de los resultados que se devuelven sigue
# siendo una recomendación razonable, y no a 1 porque el 100% se reserva para algo que
# de verdad lo sea.
SCORE_MINIMO = 0.5
SCORE_MAXIMO = 1.0


@dataclass(frozen=True)
class FiltrosBusqueda:
    """Los filtros que se aplican a la búsqueda.

    Solo `media_types` va a ChromaDB, porque es el único que el `where` sabe expresar.
    Existe como dataclass y no como tres parámetros sueltos porque la siguiente cosa que
    va a entrar aquí es `max_runtime`, que llega con el parser de intención, y en ese
    momento el filtro será sobre metadatos también. Añadirlo después no debería obligar
    a cambiar la firma de `buscar`.

    `liked_ids` no se traduce a un `where`: se podan en Python, y está aquí para que
    quien llame no tenga que acordarse de hacerlo.
    """

    media_types: list[str] = field(default_factory=lambda: ["movie", "tv", "anime"])
    liked_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResultadoBusqueda:
    """Un resultado de búsqueda, con los campos del contrato y la distancia cruda.

    Los cinco primeros son exactamente los de `Recommendation`. El último no está en el
    contrato de la API y no se serializa: es interno, para depurar. `score` va reescalado
    y `distance` es lo que sale de ChromaDB, o sea 0 es idéntico y 1 es ortogonal.
    """

    id: str
    title: str
    media_type: str
    year: int | None
    genres: list[str]
    poster_url: str | None
    score: float
    explanation: str | None = None
    distance: float | None = None

    def al_contrato(self) -> dict[str, Any]:
        """El diccionario tal y como lo quiere el contrato.

        `distance` y `_distance` se quedan fuera a propósito: son datos de depuración y
        el contrato no los pide, así que no viajan a la API.
        """
        return {
            "id": self.id,
            "title": self.title,
            "media_type": self.media_type,
            "year": self.year,
            "genres": self.genres,
            "poster_url": self.poster_url,
            "score": self.score,
            "explanation": self.explanation,
        }


def parsear_generos(metadatos: dict[str, Any]) -> list[str]:
    """Convierte `genres` de texto con "|" a lista. Nunca devuelve None.

    El índice guarda las listas como texto unido porque ChromaDB no acepta listas en los
    metadatos. Al volver hay que deshacerlo, y `ResultCard.jsx:66` hace `genres.map()`:
    un `None` aquí revienta el frontend, así que la clave ausente o vacía es `[]` y ya
    está.
    """
    bruto = metadatos.get("genres")
    if not bruto:
        return []
    if not isinstance(bruto, str):
        LOGGER.debug("genres inesperado (%s); se ignora", type(bruto).__name__)
        return []
    partes = [p.strip() for p in bruto.split("|")]
    return [p for p in partes if p]


def construir_where(media_types: list[str]) -> dict[str, Any] | None:
    """El `where` de ChromaDB para quedarse con los tipos indicados.

    Devuelve `None` si no hay restricción, porque `where=None` y `where={}` no son lo
    mismo en la API de ChromaDB y no merece la pena arriesgarse.
    """
    if not media_types:
        return None
    if len(media_types) == 1:
        return {"media_type": media_types[0]}
    return {"media_type": {"$in": list(media_types)}}


def reescalar_score(distancias: list[float]) -> list[float]:
    """Convierte distancias en scores dentro de [SCORE_MINIMO, SCORE_MAXIMO].

    Min-max sobre el conjunto recibido: el mejor sale con SCORE_MAXIMO y el peor con
    SCORE_MINIMO. Se hace así, y no con la similitud cruda, porque la diferencia entre el
    top-1 y el top-5 es de décimas de punto porcentual y el frontend redondea a entero.
    Sin reescalar, todos los resultados se ven iguales.

    Casos que devuelve la entrada sin tocar, porque no hay nada que reescalar:
    un solo resultado, lista vacía, o todas las distancias iguales.
    """
    if len(distancias) < 2:
        return [SCORE_MAXIMO] * len(distancias)

    mejor = min(distancias)
    peor = max(distancias)
    amplitud = peor - mejor
    if amplitud <= 0:
        return [SCORE_MAXIMO] * len(distancias)

    escala = SCORE_MAXIMO - SCORE_MINIMO
    return [
        SCORE_MAXIMO - ((d - mejor) / amplitud) * escala for d in distancias
    ]


def construir_resultado(
    identificador: str,
    metadatos: dict[str, Any] | None,
    distance: float | None,
    score: float,
) -> ResultadoBusqueda:
    """Junta un id, sus metadatos y su score en un `ResultadoBusqueda`.

    Todos los campos del índice son opcionales por diseño: `build_index.py` omite las
    claves que están a null en lugar de inventar un valor. Por eso todo se lee con `.get`
    y `year` acaba como `None`. La razón es que no hay que elegir un centinela y así no
    hay un `0` que signifique "no sé" separado de un `0` de verdad.
    """
    meta = metadatos or {}
    return ResultadoBusqueda(
        id=identificador,
        title=str(meta.get("title") or "(sin título)"),
        media_type=str(meta.get("media_type") or "movie"),
        year=int(meta["year"]) if meta.get("year") is not None else None,
        genres=parsear_generos(meta),
        poster_url=meta.get("poster_url"),
        score=score,
        explanation=None,
        distance=float(distance) if distance is not None else None,
    )


def _resolver(
    embedder: Embedder | None,
    coleccion: Any | None,
    dir_indice: Any | None,
) -> tuple[Embedder, Any, Any]:
    """Devuelve modelo, colección e `index_info`, decidiendo qué se usa si no se pasa nada.

    Los tres son inyectables para que los tests no descarguen 700 MB de pesos ni dependan
    del índice completo. Aquí está el único punto donde se decide.

    La colección sale de `obtener_coleccion`, que va cacheada, y no de `abrir_coleccion`:
    por este camino es por el que pasa cada request, y abrirla cada vez dejaría el
    singleton sin uso.
    """
    modelo = embedder if embedder is not None else cargar_embedder()
    if coleccion is not None:
        return modelo, coleccion, None
    objetivo = dir_indice or ruta_indice_por_defecto()
    return modelo, obtener_coleccion(objetivo), leer_index_info(objetivo)


def buscar(
    texto: str,
    filtros: FiltrosBusqueda | None = None,
    top_k: int = 5,
    embedder: Embedder | None = None,
    coleccion: Any | None = None,
    dir_indice: Any | None = None,
) -> list[ResultadoBusqueda]:
    """Busca `texto` en el catálogo y devuelve hasta `top_k` resultados.

    Puede devolver menos de `top_k`, y está bien: si con los filtros y el descarte de
    `liked_ids` no salen suficientes, rellenar con títulos ya vistos sería peor que
    devolver una lista más corta. El contrato no exige un número exacto.
    """
    if top_k <= 0:
        return []
    activos = filtros or FiltrosBusqueda()
    modelo, coleccion_abierta, info = _resolver(embedder, coleccion, dir_indice)

    if not info:
        LOGGER.debug("Buscando sin index_info.json; no se puede comprobar el modelo")
    else:
        comprobar_modelo(info, modelo)

    LOGGER.info("Consulta: %r con %s", texto[:80], activos.media_types)

    vectores = modelo.incrustar_consultas([texto])
    where = construir_where(activos.media_types)

    pedidos = _n_para_pedir(top_k, activos)
    LOGGER.info("Pidiendo %d resultados a Chroma (where=%s)", pedidos, where)
    respuesta = coleccion_abierta.query(
        query_embeddings=[vectores[0]],
        n_results=pedidos,
        where=where,
        include=["metadatas", "distances"],
    )

    identificadores = (respuesta.get("ids") or [[]])[0]
    metadatos = (respuesta.get("metadatas") or [[]])[0]
    distancias = (respuesta.get("distances") or [[]])[0]

    if not identificadores:
        LOGGER.info("Sin resultados para %r", texto[:80])
        return []

    vistos = set(activos.liked_ids)
    if vistos:
        LOGGER.debug("Descartando %d ya vistos", len(vistos))

    # Podar antes de reescalar: si se reescalara con los descartados dentro, el mejor
    # resultado se quedaría sin el SCORE_MAXIMO solo por haber salido un título visto.
    quedan = [
        (identificador, meta, distancia)
        for identificador, meta, distancia in zip(identificadores, metadatos, distancias)
        if identificador not in vistos
    ][:top_k]

    if not quedan:
        return []

    LOGGER.info(
        "Quedan %d de %d pedidos; distances crudas: %s",
        len(quedan),
        len(identificadores),
        ", ".join(f"{float(d):.4f}" for _, _, d in quedan),
    )

    scores = reescalar_score([float(d) for _, _, d in quedan])
    return [
        construir_resultado(identificador, meta, distancia, score)
        for (identificador, meta, distancia), score in zip(quedan, scores)
    ]


def _n_para_pedir(top_k: int, filtros: FiltrosBusqueda) -> int:
    """Cuántos resultados pedir a Chroma: `top_k`, o más si hay que descartar vistos."""
    if not filtros.liked_ids:
        return top_k
    return min(top_k * SOBREPETICION, TOPE_SOBREPETICION)


def buscar_como_contrato(
    texto: str,
    media_types: list[str] | None = None,
    liked_ids: list[str] | None = None,
    top_k: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Atajo con la forma de `RecommendResponse`: `{"results": [...]}`.

    Para cuando el router llegue. `explanation` sale a None porque este módulo no genera
    texto; Groq rellena ese hueco después.
    """
    resultados = buscar(
        texto,
        filtros=FiltrosBusqueda(
            media_types=list(media_types) if media_types else ["movie", "tv", "anime"],
            liked_ids=list(liked_ids) if liked_ids else [],
        ),
        top_k=top_k,
        **kwargs,
    )
    return {"results": [r.al_contrato() for r in resultados]}


__all__ = [
    "FiltrosBusqueda",
    "IndiceNoDisponibleError",
    "ResultadoBusqueda",
    "buscar",
    "buscar_como_contrato",
    "construir_resultado",
    "construir_where",
    "obtener_coleccion",
    "parsear_generos",
    "reescalar_score",
]