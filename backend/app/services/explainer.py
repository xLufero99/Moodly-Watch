"""Explica cada resultado con Groq, en una sola llamada, o con una plantilla si no puede.

Va **después** de buscar y no antes, por una razón de coste y de sentido: la búsqueda ya
sabe qué ha devuelto, y el LLM solo tiene que justificar lo que ya está decidedo. Al revés
habría que elegir los títulos con el LLM, que es la parte que el índice hace bien y
barato.

**Una sola llamada para los seis resultados**, no seis. Medido contra el endpoint, seis
llamadas por búsqueda agotan el free tier en una; una sola cabe de sobra en el presupuesto
de 8 000 tokens por minuto. La coherencia también mejora: si cada tarjeta se explicara por
su cuenta, dos títulos parecidos acabarían con argumentos distintos, y se notaría.

## La clave es `id`, no la posición

El array va en el mismo orden que los resultados, pero el emparejamiento es por `id`, no
por índice. Con `minItems`/`maxItems` iguales al número de resultados, el modelo no puede
cambiar la cantidad, pero **sí puede reordenar**, y con emparejamiento posicional la
explicación de *A* se pegaría a la tarjeta de *B*. Una explicación de "el tema es el
duelo entre padre e hijo" bajo una película de atracos es un error visible, y uno
silencioso porque el texto es plausible. Con `id` el desajuste se detecta y se corrige
solo.

## La sinopsis no está en los metadatos

Vive al final del `document` del índice, después de `Temas:` (`build_catalog.py:109`), y
los metadatos no la llevan: medido, sus 14 claves son genres, language, media_type,
popularity, popularity_pct, poster_url, rating, rating_pct, runtime_total, source, status,
title, vote_count y year. Por eso `search.buscar` pide `documents` en el `include`: sin eso
el explainer solo tendría el título y los géneros, y con eso no se puede decir por qué
encaja con un estado de ánimo, que es justo lo que se le pide.

Seis documentos son ~590 caracteres cada uno, unos 880 tokens. Entra de sobra.

## Degradación por ítem, no todo o nada

El parser degrada entero porque devuelve **un** objeto: o hay filtros o no hay. Aquí hay
seis textos independientes, así que un fallo se queda en el ítem que falló y los otros cinco
conservan su explicación. Tirar cinco explicaciones buenas porque la sexta vino mal es
justo el fallo que aquí se evita.

Las cuatro salidas, en orden:

  1. Todo Groq: explicaciones por `id`
  2. Groq parcial: las que vinieron por `id`, plantilla en las que falten
  3. Groq inservible (error, sin key, JSON inválido, contenido vacío): plantilla en las seis

La plantilla nombra los géneros, porque son datos reales del título y no inventan nada.
No menciona los filtros: el filtro es una restricción, no un motivo, y "cumple tu filtro
de 90 minutos" no le dice a nadie por qué le va a gustar.

`groq_client.py` es el mismo singleton que usa el parser. Este módulo no crea cliente.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.services.filter_parser import FiltrosConsulta
from app.services.groq_client import (
    avisar_si_no_soporta_strict,
    hay_cliente,
    obtener_cliente,
)

LOGGER = logging.getLogger(__name__)

NOMBRE_SCHEMA = "explicaciones_recomendacion"

# 6 explicaciones de 1-2 frases son ~350 tokens de salida. Se ponen 1400 porque
# `openai/gpt-oss-20b` es un modelo de razonamiento y el presupuesto se lo come antes el
# razonamiento: con un tope corto, `message.content` llega **vacío** sin error. Medido al
# diagnosticar el parser, que no fijaba tope y se quedó sin contenido.
MAX_TOKENS = 1400

# Sinopsis muy largas hacen que el modelo se extienda y, con más razón, se le escape el
# final. El corte va antes de enviar nada, no después de leer la respuesta.
MAX_SINOPSIS = 600

# `temperature=0` porque esto es extracción, no redacción creativa: la misma consulta
# debería dar la misma explicación. La skill de Groq sugiere bajar la temperatura cuando el
# JSON viene mal, que es para JSON mode; aquí el decoding va constreñido por el schema.
TEMPERATURA = 0.0


def schema_explicaciones(cantidad: int) -> dict[str, Any]:
    """El schema de la respuesta, con la cantidad **exacta** de explicaciones.

    `minItems` e `maxItems` se fijan en runtime en vez de dejarlo abierto porque un array
    de tamaño variable es justo lo que hace que el emparejamiento por `id` sea necesario.
    Con el tamaño clavado, si el modelo devuelve otra cosa es porque se equivocó, y eso
    se detecta; si además coincide en cantidad, la reordenación es el único fallo posible
    y el `id` lo cubre.

    Todo en `required` con `additionalProperties: false`, como exige `strict: true`.
    """
    return {
        "type": "object",
        "properties": {
            "explicaciones": {
                "type": "array",
                "minItems": cantidad,
                "maxItems": cantidad,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "El id del resultado explicado. Copiarlo exacto.",
                        },
                        "explicacion": {
                            "type": "string",
                            "description": "Por qué encaja con el estado de ánimo, en español.",
                        },
                    },
                    "required": ["id", "explicacion"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["explicaciones"],
        "additionalProperties": False,
    }


SISTEMA = """\
Eres el que explica recomendaciones de películas, series y anime. Recibes lo que alguien ha
pedido con sus palabras y una lista de títulos que el buscador ya ha elegido. Tu trabajo
es decir, para cada título, por qué le va a encantar a esa persona.

Devuelves un JSON con un campo `explicaciones`: un array con exactamente un objeto por
título, cada uno con el `id` del título (copiado tal cual, es la clave que lo empareja con
su tarjeta) y `explicacion`, el texto.

Reglas:

1. Escribe en español, en segunda persona, dirigiéndote a quien pidió. Dos frases como
   mucho, y basta con una si la idea cabe.

2. Explica por qué encaja con el estado de ánimo o la situación que ha descrito, no qué
   pasa en la película. "Un thriller contemplativo sobre la culpa" sirve; "un thriller de
   1980 con un final inesperado" no, porque eso es la ficha, y la ficha ya está en la
   tarjeta.

3. **No reveles el final ni ningún giro**, aunque la sinopsis los mencione. Si la sinopsis
   cuenta cómo acaba algo, usa solo lo que ocurre *antes*. Quien escribió esa sinopsis la
   escribió porque el formato del catálogo la pedía entera, no para destriparle la película
   a quien busca qué ver.

4. Si la sinopsis menciona algo sexual, violencia explícita o una muerte importante,
   descríbelo de forma neutra y no lo pongas como gancho.

5. No inventes nada. Si de la ficha no se puede sacar un motivo, di algo honesto y corto:
   "es una de las opciones que mejor encajan con lo que buscas".

6. Cada explicación debe poder sostenerse con la ficha de SU título. Nada de mezclar
   argumentos entre títulos.

7. No menciones nunca cómo funciona el buscador. Ni "filtro", ni "puntuación", ni
   "similitud", ni "embeddings". La persona no está usando un motor de búsqueda, está
   eligiendo qué ver esta noche.

Ejemplo:

Consulta: "quiero algo de ciencia ficción pero con tensión"
Filtros: {"duracion_maxima": null, "tipos": ["movie", "tv"]}
Títulos:
  - {"id": "550", "titulo": " Solaris", "media_type": "movie", "year": 1972,
     "generos": ["Ciencia ficción", "Drama"], "sinopsis": "Un sicólogo viaja a una estación
     orbital para evaluar a sus trippingantes habitantes."}

-> {"explicaciones": [{"id": "550", "explicacion": "Ciencia ficción que no busca aventura
   sino la incomodidad de mirar de cerca lo que nos pasa. Si lo que quieres es tensión que
   te incomode un poco, este es el título."}]}
"""


@dataclass(frozen=True)
class Explicacion:
    """La explicación de un resultado, y de dónde salió.

    `generada_por_llm` no va al contrato: es para los tests y para el log. Distingue un
    texto pensado de uno de relleno, que es la diferencia entre "el LLM falló" y "el LLM no
    hacía falta para este".
    """

    id: str
    texto: str
    generada_por_llm: bool = True


def explicar_por_generos(resultado: Any) -> str:
    """La explicación de reserva, con datos del propio título y sin inventar nada.

    Los géneros sí: están en los metadatos del índice, así que son ciertos por
    construcción. Es un texto poor y honesto, que es justo lo que se busca cuando Groq no
    está.
    """
    generos = [g for g in (getattr(resultado, "genres", None) or []) if g]
    if not generos:
        return "Está entre lo que mejor encaja con lo que buscas."
    lista = ", ".join(generos)
    return f"Coincide con tu búsqueda por: {lista}."


def sinopsis_de(document: str | None) -> str:
    """Saca la sinopsis del `document` del índice y la recorta.

    El formato lo pone `PLANTILLA_EMBED_TEXT` (`build_catalog.py:109`):

        "{title}. Tipo: {media_type}. Géneros: {generos}. Temas: {temas}. {overview}"

    Hacen falta **dos** cortes. `Temas: ` es una lista de palabras separadas por comas
    ("ambush, shotgun, machismo") que solo termina en `". "`, así que con un solo
    `partition("Temas: ")` la sinopsis salía precedida de los temas, y el modelo recibía
    "ambush, shotgun, machismo. Eddie convence a tres amigos...". Se descarta esa parte.

    Los temas no se usan porque sin contexto salen como ruido: son palabras clave en
    inglés sueltas, y no son sinopsis. De sinopsis sí hay, está en el documento. Cuando
    falta, mejor sin sinopsis que con una etiqueta.
    """
    if not document:
        return ""
    _, _, cola = document.partition("Temas: ")
    if not cola:
        return ""
    _, _, sinopsis = cola.partition(". ")
    sinopsis = sinopsis.strip()
    if not sinopsis:
        return ""
    if len(sinopsis) > MAX_SINOPSIS:
        sinopsis = sinopsis[:MAX_SINOPSIS].rsplit(" ", 1)[0] + "..."
    return sinopsis


def _filtros_en_llano(filtros: FiltrosConsulta) -> dict[str, Any]:
    """Los filtros como los lee el modelo: minutos, no campos de Python.

    Sin esto el LLM solo ve sus ids y no puede decir "es una película, no una serie, como
    pediste", que es de las cosas más útiles que puede decir cuando alguien ha pedido una
    restricción concreta.
    """
    llano: dict[str, Any] = {}
    if filtros.media_types:
        llano["tipos"] = list(filtros.media_types)
    if filtros.max_runtime_total is not None:
        llano["duracion_maxima_pelicula"] = f"{filtros.max_runtime_total} minutos"
    if filtros.max_runtime_episode is not None:
        llano["duracion_maxima_capitulo"] = f"{filtros.max_runtime_episode} minutos"
    return llano


def _ficha(resultado: Any) -> dict[str, Any]:
    """La ficha que ve el modelo. Titles, genres, year y sinopsis."""
    return {
        "id": resultado.id,
        "titulo": resultado.title,
        "media_type": resultado.media_type,
        "year": resultado.year,
        "generos": list(resultado.genres or []),
        "sinopsis": sinopsis_de(getattr(resultado, "document", None)),
    }


def _por_id(respuestas: Any) -> dict[str, str]:
    """Convierte el array del modelo en un `{id: explicación}`.

    - un id repetido gana el primero: el array está ordenado y el modelo ha copiado mal
    - un texto vacío no se guarda: es peor que no tener y que la plantilla lo sustituya
    """
    limpio: dict[str, str] = {}
    if not isinstance(respuestas, list):
        return limpio
    for entrada in respuestas:
        if not isinstance(entrada, dict):
            continue
        identificador = entrada.get("id")
        texto = entrada.get("explicacion")
        if not isinstance(identificador, str) or not isinstance(texto, str):
            continue
        texto = texto.strip()
        if not texto or identificador in limpio:
            continue
        limpio[identificador] = texto
    return limpio


def _alinear(
    resultados: list[Any], por_id: dict[str, str], *, motivo: str | None = None
) -> list[Explicacion]:
    """Empareja por `id` y rellena con plantilla lo que falte, ítem a ítem.

    Un `id` que el modelo se inventó no se cuela en ninguna tarjeta: se avisa y se pasa. Lo
    que no se hace es un `zip` por posición, que es donde el error se vuelve invisible.
    """
    conocidos = {r.id for r in resultados}
    sobrantes = set(por_id) - conocidos
    if sobrantes:
        LOGGER.warning("Groq explicó ids que no son de esta búsqueda: %s", sorted(sobrantes))

    salidas: list[Explicacion] = []
    for resultado in resultados:
        texto = por_id.get(resultado.id)
        if texto:
            salidas.append(Explicacion(id=resultado.id, texto=texto, generada_por_llm=True))
        else:
            LOGGER.info("Sin explicación de Groq para %s; va la de plantilla", resultado.id)
            salidas.append(
                Explicacion(
                    id=resultado.id,
                    texto=explicar_por_generos(resultado),
                    generada_por_llm=False,
                )
            )

    if motivo:
        LOGGER.warning(motivo)
    return salidas


def _todo_plantilla(resultados: list[Any], motivo: str) -> list[Explicacion]:
    """Groq no dio nada utilizable. Los seis llevan plantilla y se sigue."""
    LOGGER.warning("El explainer se degrada (%s); %d explicaciones por plantilla", motivo, len(resultados))
    return [
        Explicacion(id=r.id, texto=explicar_por_generos(r), generada_por_llm=False)
        for r in resultados
    ]


def explicar_resultados(
    consulta: str,
    filtros: FiltrosConsulta,
    resultados: list[Any],
    *,
    cliente: Any | None = None,
) -> list[Explicacion]:
    """Explica `resultados` en una sola llamada. Nunca lanza: degrada y sigue.

    `resultados` son `ResultadoBusqueda` de `search.py`. Se usan por atributo y no con un
    tipo cerrado para que el test pueda pasar dobles, como hace el resto del proyecto.

    `cliente` es para los tests, que pasan un doble y no llegan a la red. Sin él sale del
    singleton de `groq_client`, el mismo que el parser.
    """
    if not resultados:
        return []

    if not hay_cliente():
        return _todo_plantilla(resultados, "no hay GROQ_API_KEY configurada")

    avisar_si_no_soporta_strict(settings.groq_model)
    activo = cliente if cliente is not None else obtener_cliente()

    fichas = [_ficha(r) for r in resultados]
    mensaje_usuario = json.dumps(
        {
            "consulta": consulta,
            "filtros": _filtros_en_llano(filtros),
            "titulos": fichas,
        },
        ensure_ascii=False,
    )

    try:
        respuesta = activo.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SISTEMA},
                {"role": "user", "content": mensaje_usuario},
            ],
            temperature=TEMPERATURA,
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": NOMBRE_SCHEMA,
                    "strict": True,
                    "schema": schema_explicaciones(len(resultados)),
                },
            },
        )
    except Exception as error:  # noqa: BLE001
        return _todo_plantilla(resultados, f"Groq falló ({type(error).__name__})")

    try:
        contenido = respuesta.choices[0].message.content
        if not contenido or not contenido.strip():
            # El caso de `content` vacío: el modelo razonó y se comió el presupuesto sin
            # llegar a escribir. Con `max_tokens` corto en un modelo de razonamiento es lo
            # que pasa, y no es JSON inválido, es nada.
            return _todo_plantilla(resultados, "Groq devolvió contenido vacío")
        datos = json.loads(contenido)
    except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError):
        return _todo_plantilla(resultados, "la respuesta de Groq no es JSON utilizable")

    if not isinstance(datos, dict) or not isinstance(datos.get("explicaciones"), list):
        return _todo_plantilla(resultados, "la respuesta de Groq no trae el array esperado")

    por_id = _por_id(datos["explicaciones"])
    if not por_id:
        return _todo_plantilla(resultados, "Groq no explicó ningún id válido")

    LOGGER.info(
        "Explicadas %d de %d por Groq%s",
        len(por_id),
        len(resultados),
        "" if len(por_id) == len(resultados) else " (el resto por plantilla)",
    )
    return _alinear(resultados, por_id)


__all__ = [
    "MAX_SINOPSIS",
    "MAX_TOKENS",
    "SISTEMA",
    "Explicacion",
    "explicar_por_generos",
    "explicar_resultados",
    "schema_explicaciones",
    "sinopsis_de",
]