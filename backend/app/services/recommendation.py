"""El pipeline completo de `/recommend`: parsear, buscar, explicar.

Vive aparte del router para que `routes.py` sea tres líneas y para que el orden de los
tres pasos se lea en un sitio. Los tres servicios de los que depende (`filter_parser`,
`search`, `explainer`) no saben nada de este módulo, así que los tests los pueden usar por
separado sin montar nada.

    texto -> `parsear_filtros` -> `buscar` -> `explicar_resultados` -> contrato

## La regla de mezcla de `media_types`

Hay **dos** fuentes de tipos y a veces las dos dicen algo: el selector de la UI
(`request.media_types`) y lo que el modelo entendió del texto (`filtros.media_types`). Se
combinan así:

    request vacío + parser vacío     -> sin filtro
    request con tipos + parser vacío -> manda el request
    request vacío + parser con tipos -> manda el parser
    request con tipos + los dos      -> intersección

La intersección es lo correcto porque las dos son restricciones, y un filtro que se
descarte por sí solo es un filtro que no se pidió. Si alguien marca "anime" en la UI y
escribe "no quiero anime", gana la frase: el selector es un filtro y la palabra escrita es
una exclusión, y el selector no puede tapar una negación.

## `None` no es `[]`, y por qué se nota

El último caso tiene un final que no es "sin filtro" sino **cero resultados**: si alguien
pide anime en la UI y "no anime" en el texto, la intersección es `[]` y no hay ningún
título que_valga. Devolver todo el catálogo ahí sería el bug más grave posible de este
módulo, porque es exactamente el fallo que este pipeline existe para arreglar.

La tentación es representar las dos cosas con la misma lista vacía, y ahí está el
peligro: con un solo `[]` no hay forma de distinguir "nadie pidió nada" (buscar sin
filtro) de "se contradijeron" (no buscar nada), así que el código acaba recurriendo a
heurísticas y el bug vuelve. Por eso la función que decide devuelve **`None` para
"sin filtro" y `[]` para "cero resultados"**:

    None -> where=None, se busca en todo el catálogo
    []   -> no se llama a ChromaDB y se devuelve []

El tipo lo dice solo y no hay bandera que mantener sincronizada. Un `where={}` sería lo
mismo que `None` en la intención y distinto en la práctica, y `construir_where` ya lo tiene
así documentado.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from app.services.explainer import explicar_por_generos, explicar_resultados
from app.services.filter_parser import parsear_filtros
from app.services.search import (
    FiltrosBusqueda,
    IndiceNoDisponibleError,
    ResultadoBusqueda,
    buscar,
)

LOGGER = logging.getLogger(__name__)

TOP_K = 6


def mezclar_media_types(
    del_request: list[str] | None,
    del_parser: list[str] | None,
) -> list[str] | None:
    """Combina los tipos de los dos sitios. **`None` no es lo mismo que `[]`.**

    - `None` -> nadie pidió restricción: buscar sin filtro (`where=None`)
    - `[]`   -> los dos pidieron algo y no hay intersección: **cero resultados**

    Ver el docstring del módulo para el porqué de no colapsar los dos casos.
    """
    pide_request = bool(del_request)
    pide_parser = bool(del_parser)

    if not pide_request and not pide_parser:
        return None
    if pide_request and not pide_parser:
        return list(del_request or [])
    if pide_parser and not pide_request:
        return list(del_parser or [])

    comun = [t for t in (del_request or []) if t in set(del_parser or [])]
    if not comun:
        LOGGER.warning(
            "Los tipos del request (%s) y los del texto (%s) no se cruzan; "
            "no hay nada que recomendar",
            del_request,
            del_parser,
        )
        return []
    return comun


def aplicar_explicaciones(
    resultados: list[ResultadoBusqueda],
    explicaciones: list[Any],
) -> list[ResultadoBusqueda]:
    """Devuelve copias de los resultados con su explicación puesta.

    Se reconstruyen en vez de mutarse porque `ResultadoBusqueda` es `frozen=True`. Que sea
    inmutable es lo que permite que el explainer no sepa nada del buscador y el buscador
    no sepa nada del explainer.

    El emparejamiento es por `id`, no por posición, igual que en el explainer: si el
    explainer devolvió las explicaciones en otro orden, `zip` pegaría el texto de una
    película en la tarjeta de otra.
    """
    por_id = {e.id: e.texto for e in explicaciones}
    return [
        replace(r, explanation=por_id.get(r.id))
        if r.id in por_id
        else replace(r, explanation=explicar_por_generos(r))
        for r in resultados
    ]


def recomendar(
    texto: str,
    media_types: list[str] | None = None,
    liked_ids: list[str] | None = None,
    *,
    top_k: int = TOP_K,
    cliente: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """De una frase a `{"results": [...]}` con las explicaciones dentro.

    `**kwargs` va al buscador, y son los puntos de inyección de los tests: `embedder`,
    `coleccion` y `dir_indice`. `cliente` va al parser y al explainer, que comparten el
    mismo doble para que un test pueda ver las dos llamadas.

    No lanza por falta de Groq: sin key, el parser degrada a texto crudo sin filtros, la
    búsqueda va igual y el explainer cae a plantilla. Solo lanza `IndiceNoDisponibleError`
    si no hay índice, que es un problema del despliegue y lo traduce el router a un 503.
    """
    filtros = parsear_filtros(texto, cliente=cliente)

    tipos = mezclar_media_types(media_types, filtros.media_types)
    if tipos is not None and not tipos:
        LOGGER.info("Los tipos se contradicen; se devuelve vacío sin llamar a ChromaDB")
        return {"results": []}

    resultados = buscar(
        filtros.query or texto,
        filtros=FiltrosBusqueda(
            media_types=list(tipos) if tipos else ["movie", "tv", "anime"],
            liked_ids=list(liked_ids or []),
            max_runtime_total=filtros.max_runtime_total,
            max_runtime_episode=filtros.max_runtime_episode,
        ),
        top_k=top_k,
        **kwargs,
    )

    if not resultados:
        return {"results": []}

    explicaciones = explicar_resultados(texto, filtros, resultados, cliente=cliente)
    con_explicacion = aplicar_explicaciones(resultados, explicaciones)

    LOGGER.info(
        "%d resultados, %d explicados por Groq",
        len(con_explicacion),
        sum(1 for e in explicaciones if e.generada_por_llm),
    )
    return {"results": [r.al_contrato() for r in con_explicacion]}


__all__ = [
    "TOP_K",
    "IndiceNoDisponibleError",
    "aplicar_explicaciones",
    "mezclar_media_types",
    "recomendar",
]