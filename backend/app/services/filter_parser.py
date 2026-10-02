"""Convierte lo que alguien escribe en filtros exactos de búsqueda, con Groq.

Este módulo es lo que hace que el recomendador entienda "no quiero anime". Sin él, la
búsqueda vectorial no entiende negaciones: medido, `"no quiero anime, algo real y corto"`
devuelve 3 anime de 5 y el primero es *Hokuto no Ken Movie*. La razón es que un vector
cercano no es un vector opuesto, y limpiar la frase con regex no lo arregla: aguanta
"no quiero anime" y se rompe con "fuera de anime" o "algo que no sea anime". Un arreglo que
funciona en el caso que pruebas y falla en el siguiente da confianza falsa, así que se
descartó.

La solución es no intentar entender la frase, sino **traducirla a un filtro**. Un filtro
sobre metadatos es exacto: `media_type = "movie"` excluye anime de verdad, no de
aproximadamente. Eso es lo que sale de aquí.

**No es un modelo de intención, es un extractor de filtros.** Por eso el dataclass se llama
`FiltrosConsulta` y no `QueryIntent`, y el fichero `filter_parser.py` y no
`intent_parser.py`. "Intención" prometería que se entiende lo que la persona quiere en el
sentido general; lo que hay es una traducción de unos pocos campos que el catálogo puede
respetar.

## Dos reglas semánticas que van en el prompt

Las dos están porque los datos del catálogo no son homogéneos, y están medidas sobre
`catalog.parquet`:

| media_type | `runtime_total` | `runtime_episode` |
|---|---|---|
| movie | 100 % | 0 % |
| tv | 0 % | 100 % |
| anime | 22 % | 78 % |

1. **`max_runtime_total` fuerza `media_types: ["movie"]`.** `runtime_total` no existe para
   ninguna serie, así que prometer ese filtro con series o anime en la lista sería mentir:
   devolvería vacío, y vacío parece un fallo. Solo `movie`, donde el dato está al 100 %.
   No `["movie", "anime"]`: en anime el dato solo está en el 22 % de las filas, o sea que
   el filtro descartaría títulos que sí duran lo que se pidió.

2. **"series de menos de 2 horas" es duración de episodio, no total.** Una serie de 8
   episodios de 45 minutos no dura 2 horas en total, así que el filtro total sería
   imposible de satisfacer y devolvería nada. La lectura útil es la del capítulo. Va como
   regla del prompt, no como parche en Python: la información para decidirla está en la
   frase que escribe la persona, no en los metadatos.

**El buscador no lleva lógica por tipo.** Solo aplica lo que le llega con un `where`. Si
recibe `max_runtime_total` con `media_types: ["movie"]`, monta el `$and` y ya. La decisión
de qué es coherente está aquí, una vez.

## Degradación

Si Groq falla (sin key, 429, timeout, JSON inválido, `query` vacía), se devuelve el texto
crudo **sin filtros**, y se marca `degradado=True`. Nunca se inventan filtros: buscar "no
quiero anime" sin filtro devuelve anime, y eso ya está medido y documentado, pero es un
resultado honesto. Un filtro inventado sería peor, porque el usuario no tiene forma de
saber que se lo pusimos nosotros.

`degradado` significa **fallo de infraestructura**, no "el modelo no pidió nada". Si Groq
contesta `"media_types": []` y `null` en las duraciones, eso es una respuesta legítima y
legítima es `degradado=False`: la persona no pidió filtros. Esa distinción es la razón de
que el campo exista.

Con dos llamadas por búsqueda (este parser y el explainer que vendrá), el free tier de
`openai/gpt-oss-20b` son 30 peticiones por minuto y 8 000 tokens por minuto, o sea del
orden de 15 búsquedas por minuto antes de empezar a recibir 429. Ver el README de Groq
para los límites actuales.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.services.groq_client import (
    avisar_si_no_soporta_strict,
    hay_cliente,
    obtener_cliente,
)

LOGGER = logging.getLogger(__name__)

NOMBRE_SCHEMA = "filtros_consulta"

# Los tipos que el catálogo tiene. Coincide con el enum del schema, que es lo que hace el
# decoding constreñido: si el modelo intenta devolver "serie" no le deja.
MEDIA_TYPES = ["movie", "tv", "anime"]

# Todo en `required` y con `additionalProperties: false`, como exige `strict: true`.
# Un campo fuera de `required` o un objeto sin `additionalProperties` da 400, así que el
# test `test_el_schema_es_valido_para_strict` recorre esto y lo comprueba.
#
# Las duradas van como unión con null a propósito. La doc de Groq dice que los campos
# opcionales no se soportan y que hay que usar uniones con null, manteniendo el campo en
# `required`. Null significa "no pidió duración", que es un valor honesto. Si el campo
# fuera solo `integer` sin null, el modelo estaría obligado a inventar un número, que es
# justo lo que no se quiere.
SCHEMA_FILTROS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "La búsqueda, sin negaciones ni filtros de duración.",
        },
        "media_types": {
            "type": "array",
            "items": {"type": "string", "enum": MEDIA_TYPES},
            "description": "Tipos admitidos. Lista vacía si no dijo nada.",
        },
        "max_runtime_total": {
            "type": ["integer", "null"],
            "description": (
                "Duración máxima del título en minutos. Solo para películas. Null si no pidió."
            ),
        },
        "max_runtime_episode": {
            "type": ["integer", "null"],
            "description": (
                "Duración máxima de un episodio en minutos, para series y anime. "
                "Null si no pidió."
            ),
        },
    },
    "required": [
        "query",
        "media_types",
        "max_runtime_total",
        "max_runtime_episode",
    ],
    "additionalProperties": False,
}

SISTEMA = """\
Eres el parser de filtros de un buscador de películas, series y anime. Conviertes lo que
escribe una persona en filtros que un buscador puede aplicar de forma exacta.

Devuelves un JSON con cuatro campos: query, media_types, max_runtime_total y
max_runtime_episode.

Reglas:

1. `query` es lo que realmente se busca: el tema, el tono, la situación. Quita de ahí las
   negaciones y las condiciones de duración. "quiero algo real y corto" deja
   query="algo real" y la duración a su campo.

   `query` puede quedar vacía, y es lo correcto cuando la persona solo ha dicho filtros
   sin decir de qué tema. "series de menos de 2 horas" es duración, no tema: deja
   query="". Está bien, no lo rellenes con el texto entero ni repitas los filtros ahí.
   Rellénala solo cuando haya tema, tono o situación.

2. `media_types` es la lista de tipos que admite. Si no dice nada, lista vacía. Si dice
   que no quiere anime, deja ["movie", "tv"] y NO dejes "anime" dentro.

3. `max_runtime_total` es para películas, en minutos. "menos de 2 horas" son 120 minutos.
   Cuando lo rellenes, `media_types` tiene que ser exactamente ["movie"]: la duración
   total solo existe en el catálogo para películas, así que combinada con series o anime el
   filtro no se podría cumplir.

4. `max_runtime_episode` es la duración de un capítulo, para series y anime. "episodios
   cortos" son 25 minutos. Cuando lo rellenes, `media_types` solo puede ser ["tv"] o
   ["anime"].

5. Si pide duración para series ("series de menos de 2 horas"), ponla en
   `max_runtime_episode`, nunca en `max_runtime_total`. El total de una serie son muchas
   horas, así que un límite total dejaría sin resultados.

Ejemplos:

"no quiero anime, algo real y corto"
-> {"query": "algo real", "media_types": ["movie"], "max_runtime_total": 120,
    "max_runtime_episode": null}

"series de menos de 2 horas"
-> {"query": "", "media_types": ["tv"], "max_runtime_total": null,
    "max_runtime_episode": 120}

"películas de menos de 90 minutos sobreSubmission"
-> {"query": "Submission", "media_types": ["movie"], "max_runtime_total": 90,
    "max_runtime_episode": null}

"series con capítulos cortos"
-> {"query": "", "media_types": ["tv"], "max_runtime_total": null,
    "max_runtime_episode": 25}

"algo triste y lento"
-> {"query": "algo triste y lento", "media_types": [], "max_runtime_total": null,
    "max_runtime_episode": null}
"""


@dataclass(frozen=True)
class FiltrosConsulta:
    """Lo que se busca y con qué restricciones. Todo opcional menos `query`.

    `media_types` vacío y las duradas a None significan lo mismo: "no pidió ese filtro".
    `degradado` no es un filtro: es la marca de que Groq no respondió y estos filtros
    están sin construir, no rellenados por el modelo.
    """

    query: str
    media_types: list[str] = field(default_factory=list)
    max_runtime_total: int | None = None
    max_runtime_episode: int | None = None
    degradado: bool = False
    motivo: str | None = None


def _sin_filtros(texto: str, motivo: str) -> FiltrosConsulta:
    """La degradación: el texto tal cual, sin filtros, y por qué.

    Nunca inventa filtros. Buscar "no quiero anime" sin filtro devuelve anime, que es un
    resultado malo pero honesto y ya está documentado como tal.
    """
    LOGGER.warning("El parser se degrada (%s); buscando el texto crudo sin filtros", motivo)
    return FiltrosConsulta(
        query=texto, media_types=[], degradado=True, motivo=motivo
    )


def _validar(datos: dict[str, Any], texto: str) -> FiltrosConsulta:
    """Convierte el dict de Groq en `FiltrosConsulta`, tirando lo que no encaje.

    Con `strict: true` el JSON encaja, pero no por eso se confía: el 400 también puede
    venir del servidor, y un filtro con un tipo inventado es peor que no tener filtro.
    """
    consulta = str(datos.get("query") or "").strip()

    tipos = datos.get("media_types")
    tipos_validos = [t for t in tipos if t in MEDIA_TYPES] if isinstance(tipos, list) else []
    if isinstance(tipos, list) and len(tipos_validos) != len(tipos):
        LOGGER.warning(
            "Groq devolvió tipos que no son del catálogo (%s); se descartan",
            [t for t in tipos if t not in MEDIA_TYPES],
        )

    total = _entero_o_none(datos.get("max_runtime_total"))
    episodio = _entero_o_none(datos.get("max_runtime_episode"))

    if total is not None:
        tipos_validos = [t for t in tipos_validos if t == "movie"] or ["movie"]

    if not consulta and not tipos_validos and total is None and episodio is None:
        # Ni tema ni filtros. Solo queda una cosa que buscar: lo que escribió la persona.
        # Es distinto de "el modelo no pidió filtros", y de una query vacía ilegible.
        return _sin_filtros(texto, "Groq no devolvió ni query ni filtros")

    if not consulta:
        LOGGER.info("Groq devolvió filtros sin query; se busca solo por ellos")

    return FiltrosConsulta(
        query=consulta,
        media_types=tipos_validos,
        max_runtime_total=total,
        max_runtime_episode=episodio,
    )


def _entero_o_none(valor: Any) -> int | None:
    """Un entero o None. Un runtime negativo o absurdo se descarta, no se corrige."""
    if valor is None or isinstance(valor, bool):
        return None
    try:
        numero = int(valor)
    except (TypeError, ValueError):
        return None
    # 0 no es una duración, y por encima de 10 000 minutos es ruido de bucle. Descartar
    # es mejor que filtrar por un valor que nadie pidió.
    return numero if 0 < numero <= 10_000 else None


def parsear_filtros(texto: str, *, cliente: Any | None = None) -> FiltrosConsulta:
    """Traduce `texto` a filtros. Nunca lanza: si algo falla, degrada y sigue.

    `cliente` existe para los tests, que pasan un doble y no llegan a la red. Sin él sale
    del singleton de `groq_client`.
    """
    if not hay_cliente():
        return _sin_filtros(texto, "no hay GROQ_API_KEY configurada")

    avisar_si_no_soporta_strict(settings.groq_model)
    activo = cliente if cliente is not None else obtener_cliente()

    try:
        respuesta = activo.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SISTEMA},
                {"role": "user", "content": texto},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": NOMBRE_SCHEMA,
                    "strict": True,
                    "schema": SCHEMA_FILTROS,
                },
            },
        )
    except Exception as error:  # noqa: BLE001
        # 429, timeout, 400 por un modelo sin structured outputs, red caída. Todos
        # aterrizan aquí y todos significan lo mismo: no hay filtros.
        return _sin_filtros(texto, f"Groq falló ({type(error).__name__})")

    try:
        contenido = respuesta.choices[0].message.content
        datos = json.loads(contenido or "")
    except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError):
        return _sin_filtros(texto, "la respuesta de Groq no es JSON utilizable")

    if not isinstance(datos, dict):
        return _sin_filtros(texto, "la respuesta de Groq no es un objeto")

    filtros = _validar(datos, texto)
    LOGGER.info(
        "Filtros para %r: media_types=%s total=%s episodio=%s%s",
        texto[:60],
        filtros.media_types,
        filtros.max_runtime_total,
        filtros.max_runtime_episode,
        " (degradado)" if filtros.degradado else "",
    )
    return filtros


def aplicar_a_search(
    filtros: FiltrosConsulta,
    liked_ids: list[str] | None = None,
    runtime_maximo_total: int | None = None,
    runtime_maximo_episodio: int | None = None,
) -> Any:
    """Convierte los filtros en lo que `app.services.search.buscar` espera.

    Vive aquí y no en el buscador porque es la dirección de la traducción: lo sabe el
    parser, que es quien conoce el significado de sus propios campos. `buscar` solo sabe
    montar un `where`, no que exista `max_runtime_total` ni por qué fuerza `movie`.

    Los runtimes se pasan aparte y no se leen de `filtros` a propósito. El buscador los
    aplica tal cual, sin interpretar: si quien llama decide que `max_runtime_total` con
    series es un error de datos, es problema de quien llama. Así no hay dos sitios que
    saben qué significa una duración.
    """
    from app.services.search import FiltrosBusqueda

    return FiltrosBusqueda(
        media_types=list(filtros.media_types),
        liked_ids=list(liked_ids or []),
        max_runtime_total=runtime_maximo_total,
        max_runtime_episode=runtime_maximo_episodio,
    )


__all__ = [
    "MEDIA_TYPES",
    "SCHEMA_FILTROS",
    "FiltrosConsulta",
    "aplicar_a_search",
    "parsear_filtros",
]