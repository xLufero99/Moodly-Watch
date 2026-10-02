"""Cliente de Groq, singleton, con degradación si no hay red o no hay key.

Existe aparte de `filter_parser.py` por una razón práctica: el cliente es lo único que se
cachea y lo único que se sustituye en las pruebas. Si el parser lo trajera dentro, los
tests del parser tendrían que construir una cadena de indirección para llegar al doble; con
los dos ficheros separados, el test mete un cliente falso y ya.

El cliente va en caché con `lru_cache` porque el SDK de Groq monta su propio pool de
conexiones HTTP. Uno por llamada tira el handshake y las conexiones cada vez. Con la misma
decisión consciente que en `vector_store.py`: esto es **por proceso**, y con varios workers
cada uno tendría su cliente. No es problema hoy porque se corre un worker.

**Si no hay `GROQ_API_KEY`, esto no lanza: degrada.** Un settings vacío no puede romper una
búsqueda. `hay_cliente()` responde a eso sin construir nada.

Lo que no se hace aquí, y por qué: no hay reintentos. La skill sugiere reintentar con otra
temperatura cuando el JSON viene mal, pero eso es para JSON mode, donde el modelo puede
inventar sintaxis. Con `strict: true` el decoding está constreñido y no hay JSON inválido
que reintentar. Y en el free tier el límite son 30 peticiones por minuto: un reintento en
medio de un 429 empeora justo lo que está fallando.
"""

from __future__ import annotations

import functools
import logging

from app.config import settings

LOGGER = logging.getLogger(__name__)

# Modelos que aceptan structured outputs con `strict: true`. Comprobado en la doc de
# Groq: son tres, y el resto devuelve 400 en cuanto se le pasa un `json_schema` estricto.
# `llama-3.3-70b-versatile` no está en la lista, que es justo el valor que tenía la
# configuración por defecto antes.
MODELOS_CON_STRICT = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    }
)


def hay_cliente() -> bool:
    """Si hay con qué hablarle a Groq. No construye nada, solo mira la configuración."""
    return bool(settings.groq_api_key.strip())


def modelo_soporta_strict(modelo: str) -> bool:
    """Si este modelo acepta `strict: true`. Sirve para avisar antes de degradar."""
    return modelo in MODELOS_CON_STRICT


def avisar_si_no_soporta_strict(modelo: str) -> None:
    """Deja constancia si el modelo configurado no puede hacer structured outputs.

    El aviso es obligatorio y va antes de degradar. Si el modelo está mal, los resultados
    llegan sin filtros y sin ninguna pista de por qué, y eso es peor que un error: el
    usuario ve recomendaciones que no ha pedido y no tiene forma de saber que el LLM no
    está haciendo su parte.
    """
    if not modelo_soporta_strict(modelo):
        LOGGER.warning(
            "El modelo %s no soporta structured outputs con strict: true, así que el "
            "parser se va a degradar y las búsquedas saldrán sin filtros. Modelos que sí: %s",
            modelo,
            ", ".join(sorted(MODELOS_CON_STRICT)),
        )


@functools.lru_cache(maxsize=1)
def obtener_cliente() -> object:
    """Devuelve el cliente de Groq, construyéndolo una vez.

    Importa `groq` dentro de la función a propósito: importar el módulo no debe costing
    nada, y los tests que no usan el cliente no lo necesitan instalado.
    """
    from groq import Groq

    LOGGER.info("Cliente de Groq creado con el modelo %s", settings.groq_model)
    return Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)


def limpiar_cache() -> None:
    """Olvida el cliente. Para tests."""
    obtener_cliente.cache_clear()