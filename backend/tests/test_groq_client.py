"""Tests de `app/services/groq_client.py`: el cliente que comparten parser y explainer.

No hay ni un test de esto en otro sitio, y es una pena: los dos módulos que lo usan lo dan
por bueno, así que un fallo aquí se manifiesta como "el parser degrada sin motivo" o "el
explainer tarda 24 segundos", que son dos síntomas muy lejos de la causa.

Lo que cubren:

  * que el cliente se construye una vez y se reutiliza
  * que `max_retries=0`, que es un reintento de más de 20 s de latencia si vuelve a 2
  * que el timeout sale de la configuración
  * que `hay_cliente` solo mira la configuración y no construye nada
  * que un modelo sin structured outputs avisa antes de degradar
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services import groq_client
from app.services.groq_client import (
    MODELOS_CON_STRICT,
    avisar_si_no_soporta_strict,
    hay_cliente,
    limpiar_cache,
    modelo_soporta_strict,
    obtener_cliente,
)


@pytest.fixture(autouse=True)
def _limpiar():
    """Cada test empieza y acaba con el caché del cliente vacío.

    El singleton es del proceso, así que un test que lo llame deja el cliente construido
    para los siguientes, y `monkeypatch` sobre `settings` ya no tendría efecto.
    """
    limpiar_cache()
    yield
    limpiar_cache()


@pytest.fixture
def con_key(monkeypatch):
    monkeypatch.setattr(groq_client.settings, "groq_api_key", "test-key")


def test_el_cliente_no_reintenta(con_key) -> None:
    """`max_retries=0`, y no por gusto sino por medición.

    El SDK de Groq viene con `max_retries=2`. Con eso, el `timeout` de 10 s **no** acota la
    llamada: son tres intentos de hasta 10 s, o sea hasta 30 s. Medido, una llamada del
    explainer tardó **24,4 s**.

    Y en el free tier es peor que lento: reintentar en medio de un 429 consume cuota y hace
    que el 429 sea más probable, que es justo lo contrario de lo que se busca al degradar.
    """
    cliente = obtener_cliente()
    assert cliente.max_retries == 0


def test_el_timeout_sale_de_la_configuracion(con_key) -> None:
    cliente = obtener_cliente()
    assert cliente.timeout == settings.groq_timeout


def test_el_cliente_se_construye_una_sola_vez(con_key) -> None:
    """El SDK tiene su propio pool de conexiones HTTP; uno por llamada tira el handshake.

    Sin esto la búsqueda paga una conexión nueva por request. Y es también lo que permite
    que los tests inyecten un doble sin tener que monkeypatchear el import de `groq`.
    """
    primero = obtener_cliente()
    segundo = obtener_cliente()

    assert primero is segundo
    assert obtener_cliente.cache_info().hits == 1


def test_limpiar_cache_olvida_el_cliente(con_key) -> None:
    primero = obtener_cliente()
    limpiar_cache()
    segundo = obtener_cliente()
    assert primero is not segundo


def test_hay_cliente_no_construye_nada(monkeypatch) -> None:
    """Solo mira la configuración. Es lo que permite llamar sin red y sin key.

    Si construyera, `hay_cliente()` dejaría de ser barato y los tests que no usan Groq
    pagarían una construcción por el solo hecho de preguntar.
    """
    monkeypatch.setattr(groq_client.settings, "groq_api_key", "algo")
    hay_cliente()
    assert obtener_cliente.cache_info().currsize == 0


def test_sin_key_no_hay_cliente(monkeypatch) -> None:
    monkeypatch.setattr(groq_client.settings, "groq_api_key", "")
    assert hay_cliente() is False

    monkeypatch.setattr(groq_client.settings, "groq_api_key", "   ")
    assert hay_cliente() is False


def test_los_modelos_de_structured_outputs() -> None:
    """La lista corta y comprobada. `llama-3.3-70b-versatile` no está, a propósito.

    La skill de Groq lo recomienda para structured outputs y es **falso**: devuelve 400 en
    cuanto se le pasa un `json_schema` estricto. Era el valor por defecto de la
    configuración y hacía que el parser degradara siempre.
    """
    assert modelo_soporta_strict("openai/gpt-oss-20b")
    assert modelo_soporta_strict("openai/gpt-oss-120b")
    assert not modelo_soporta_strict("llama-3.3-70b-versatile")
    assert "llama-3.3-70b-versatile" not in MODELOS_CON_STRICT


def test_un_modelo_invalido_avisa_antes_de_degradar(caplog) -> None:
    """El aviso es obligatorio: sin él, los resultados salen sin filtros y sin explicación
    de por qué, que es peor que un error."""
    avisar_si_no_soporta_strict("llama-3.3-70b-versatile")
    assert any("no soporta structured outputs" in r.message for r in caplog.records)


def test_un_modelo_valido_no_avisa(caplog) -> None:
    avisar_si_no_soporta_strict("openai/gpt-oss-20b")
    assert not caplog.records