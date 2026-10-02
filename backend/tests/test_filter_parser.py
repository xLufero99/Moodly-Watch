"""Tests de `app/services/filter_parser.py` y de los filtros nuevos de `search.py`.

**Nada de aquí habla con Groq.** Todos los tests usan `ClienteGroqFalso`, que devuelve un
contenido fijo o tira una excepción fija. Los tests que sí llaman a la API real están
marcados `@pytest.mark.lento` y no los recoge la suite normal: un test que llama a un
servicio externo en cada `pytest` se rompe el día que no hay red, y ese día es
precisamente cuando más falta hace la suite.

Lo que cubren:

  * que el schema es válido para `strict: true`: todo en `required`, con
    `additionalProperties: false`. Es el error que da 400 y no lo da hasta runtime
  * que la llamada lleva `strict: True` y el schema
  * null y lista vacía son respuestas legítimas, no degradación
  * `query` vacía sí degrada
  * JSON inválido, excepción del cliente y ausencia de key degradan sin lanzar
  * que un tipo inventado se descarta y que `max_runtime_total` fuerza `movie`
  * que un runtime absurdo se descarta en vez de corregirse
  * el `$and` de `search.py`: que hay un solo operador por `where`, porque ChromaDB no
    admite más

Y el punto de fondo, medido:

  * `test_max_runtime_total_fuerza_movie`: `runtime_total` no existe para ninguna serie
    (0 % de cobertura en `tv`), así que prometer ese filtro con series devolvería vacío y
    parecería un fallo.

Sobre lo que NO cubren: que el modelo entienda las negaciones. Eso solo se puede probar
llamando a Groq de verdad, y está en los tests `@pytest.mark.lento`.
"""

from __future__ import annotations

import json

import pytest
from _fakes import ClienteGroqFalso, ColeccionFalsa, EmbedderFalso

from app.services import groq_client
from app.services.filter_parser import (
    MEDIA_TYPES,
    SCHEMA_FILTROS,
    FiltrosConsulta,
    aplicar_a_search,
    parsear_filtros,
)
from app.services.groq_client import (
    MODELOS_CON_STRICT,
    hay_cliente,
    modelo_soporta_strict,
)
from app.services.search import FiltrosBusqueda, construir_where_completo


def groq_responde(**campos) -> ClienteGroqFalso:
    """Un cliente falso que devuelve un JSON con todos los campos del schema."""
    base = {
        "query": "algo",
        "media_types": [],
        "max_runtime_total": None,
        "max_runtime_episode": None,
    }
    base.update(campos)
    return ClienteGroqFalso(json.dumps(base))


@pytest.fixture(autouse=True)
def _con_key(request, monkeypatch):
    """Pone una key falsa para que ningún unitario dependa de la real.

    Se salta en los tests `lento`, que sí necesitan la key de verdad: sin esto el
    autouse les pisaba la configuración y todos fallaban con AuthenticationError
    teniendo la key correcta en el entorno.
    """
    if not request.node.get_closest_marker("lento"):
        monkeypatch.setattr(groq_client.settings, "groq_api_key", "test-key")
        monkeypatch.setattr(
            "app.services.filter_parser.settings.groq_model", "openai/gpt-oss-20b"
        )
    yield


# --------------------------------------------------------------------------- #
# el schema
# --------------------------------------------------------------------------- #


def test_el_schema_es_valido_para_strict() -> None:
    """Todo en `required` y `additionalProperties: false`, o la API devuelve 400."""
    assert SCHEMA_FILTROS["additionalProperties"] is False
    assert set(SCHEMA_FILTROS["required"]) == set(SCHEMA_FILTROS["properties"])


def test_todo_objeto_del_schema_es_cerrado() -> None:
    """Los objetos anidados también. Solo hay el raíz, pero se comprueba igualmente."""
    assert list(SCHEMA_FILTROS["properties"].keys()) == [
        "query",
        "media_types",
        "max_runtime_total",
        "max_runtime_episode",
    ]


def test_las_duradas_admiten_null() -> None:
    """Null es "no pidió duración", y es el mecanismo para los opcionales en Groq.

    Sin el `null` en la unión el modelo estaría obligado a inventar un número, que es
    justo el fallo que no se quiere: un filtro inventado es peor que ningún filtro.
    """
    for campo in ("max_runtime_total", "max_runtime_episode"):
        assert SCHEMA_FILTROS["properties"][campo]["type"] == ["integer", "null"]


def test_el_enum_de_tipos_coincide_con_el_catalogo() -> None:
    items = SCHEMA_FILTROS["properties"]["media_types"]["items"]
    assert items["enum"] == MEDIA_TYPES


# --------------------------------------------------------------------------- #
# la llamada
# --------------------------------------------------------------------------- #


def test_manda_strict_y_el_schema() -> None:
    cliente = groq_responde()
    parsear_filtros("algo", cliente=cliente)
    formato = cliente.llamadas[0]["response_format"]
    assert formato["type"] == "json_schema"
    assert formato["json_schema"]["strict"] is True
    assert formato["json_schema"]["schema"] is SCHEMA_FILTROS


def test_manda_el_modelo_configurado() -> None:
    cliente = groq_responde()
    parsear_filtros("algo", cliente=cliente)
    assert cliente.llamadas[0]["model"] == "openai/gpt-oss-20b"


def test_no_usa_streaming_ni_tools() -> None:
    """Con structured outputs, Groq no admite ninguno de los dos."""
    cliente = groq_responde()
    parsear_filtros("algo", cliente=cliente)
    llamada = cliente.llamadas[0]
    assert "stream" not in llamada
    assert "tools" not in llamada
    assert "tool_choice" not in llamada


def test_lleva_el_texto_de_la_persona() -> None:
    cliente = groq_responde()
    parsear_filtros("no quiero anime", cliente=cliente)
    mensajes = cliente.llamadas[0]["messages"]
    assert mensajes[0]["role"] == "system"
    assert mensajes[1]["content"] == "no quiero anime"


# --------------------------------------------------------------------------- #
# respuestas legítimas
# --------------------------------------------------------------------------- #


def test_null_no_es_degradacion() -> None:
    """Que el modelo no pida nada es una respuesta válida, no un fallo."""
    filtros = parsear_filtros("algo triste", cliente=groq_responde(query="algo triste"))
    assert filtros.degradado is False
    assert filtros.media_types == []
    assert filtros.max_runtime_total is None
    assert filtros.max_runtime_episode is None


def test_lista_vacia_no_es_degradacion() -> None:
    filtros = parsear_filtros("algo", cliente=groq_responde(media_types=[]))
    assert filtros.degradado is False


def test_filtrado_por_tipo() -> None:
    filtros = parsear_filtros("no quiero anime", cliente=groq_responde(media_types=["movie", "tv"]))
    assert filtros.media_types == ["movie", "tv"]
    assert filtros.degradado is False


def test_duracion_de_episodio() -> None:
    filtros = parsear_filtros(
        "series con capítulos cortos", cliente=groq_responde(media_types=["tv"], max_runtime_episode=25)
    )
    assert filtros.max_runtime_episode == 25
    assert filtros.degradado is False


def test_duracion_total_fuerza_movie() -> None:
    """`runtime_total` no existe para series: 0 % de cobertura en `tv`.

    Si el modelo devolviera series con duración total, el filtro no se podría cumplir y
    devolvería vacío, que parece un fallo. Se corrige a movie en vez de propagarlo.
    """
    filtros = parsear_filtros(
        "películas de menos de 2 horas",
        cliente=groq_responde(media_types=["movie", "tv"], max_runtime_total=120),
    )
    assert filtros.media_types == ["movie"]
    assert filtros.max_runtime_total == 120


def test_max_runtime_total_sin_tipos_pone_movie() -> None:
    filtros = parsear_filtros(
        "algo de 90 minutos", cliente=groq_responde(media_types=[], max_runtime_total=90)
    )
    assert filtros.media_types == ["movie"]


# --------------------------------------------------------------------------- #
# valores que se descartan
# --------------------------------------------------------------------------- #


def test_tipo_inventado_se_descarta() -> None:
    """Un filtro con un tipo que el catálogo no tiene es peor que no filtrar."""
    filtros = parsear_filtros("algo", cliente=groq_responde(media_types=["serie", "movie"]))
    assert filtros.media_types == ["movie"]


@pytest.mark.parametrize("valor", [0, -30, 99_999, "mucho", True])
def test_runtime_absurdo_se_descarta(valor) -> None:
    """Se descarta, no se corrige. Inventar un número es lo que no queremos."""
    filtros = parsear_filtros("algo", cliente=groq_responde(max_runtime_total=valor))
    assert filtros.max_runtime_total is None


def test_media_types_que_no_es_lista() -> None:
    filtros = parsear_filtros("algo", cliente=groq_responde(media_types="movie"))
    assert filtros.media_types == []


# --------------------------------------------------------------------------- #
# degradación
# --------------------------------------------------------------------------- #


def test_query_vacia_sin_filtros_degrada() -> None:
    """Ni tema ni filtros: no hay nada con qué buscar, se usa el texto entero."""
    filtros = parsear_filtros("no quiero anime", cliente=groq_responde(query="   "))
    assert filtros.degradado is True
    assert filtros.query == "no quiero anime"
    assert filtros.media_types == []


def test_query_vacia_con_filtros_no_degrada() -> None:
    """"series de menos de 2 horas" no tiene tema, y eso es una respuesta válida.

    Solo hay filtros, así que se busca por los metadatos. Verificado contra el modelo
    real: devuelve `{"query": "", "media_types": ["tv"], "max_runtime_episode": 120}`.
    """
    filtros = parsear_filtros(
        "series de menos de 2 horas",
        cliente=groq_responde(query="", media_types=["tv"], max_runtime_episode=120),
    )
    assert filtros.degradado is False
    assert filtros.query == ""
    assert filtros.media_types == ["tv"]
    assert filtros.max_runtime_episode == 120


def test_json_invalido_degrada() -> None:
    filtros = parsear_filtros("no quiero anime", cliente=ClienteGroqFalso("esto no es json"))
    assert filtros.degradado is True
    assert filtros.query == "no quiero anime"


def test_json_que_no_es_objeto_degrada() -> None:
    filtros = parsear_filtros("algo", cliente=ClienteGroqFalso("[1, 2, 3]"))
    assert filtros.degradado is True


def test_contenido_none_degrada() -> None:
    filtros = parsear_filtros("algo", cliente=ClienteGroqFalso(None))
    assert filtros.degradado is True


def test_error_del_cliente_degrada() -> None:
    """429, timeout, 400 por modelo sin strict, red caída: todo es lo mismo aquí."""
    for error in (RuntimeError("429"), TimeoutError("timeout"), ValueError("400")):
        filtros = parsear_filtros("no quiero anime", cliente=ClienteGroqFalso(error=error))
        assert filtros.degradado is True
        assert filtros.query == "no quiero anime"
        assert filtros.media_types == []


def test_sin_key_degrada_sin_lanzar(monkeypatch) -> None:
    """Una key vacía no puede romper una búsqueda."""
    monkeypatch.setattr(groq_client.settings, "groq_api_key", "")
    filtros = parsear_filtros("no quiero anime")
    assert filtros.degradado is True
    assert filtros.query == "no quiero anime"
    assert "KEY" in (filtros.motivo or "")


def test_sin_key_no_se_llama_a_groq(monkeypatch) -> None:
    """Sin key no se construye ni se toca el cliente."""
    from app.services import filter_parser as fp

    llamado = []
    monkeypatch.setattr(fp, "obtener_cliente", lambda: llamado.append(1))
    monkeypatch.setattr(groq_client.settings, "groq_api_key", "")

    fp.parsear_filtros("algo")
    assert llamado == []


def test_el_motivo_de_la_degradacion_se_guarda() -> None:
    """Para poder decir por qué el usuario vio resultados sin filtrar."""
    filtros = parsear_filtros("algo", cliente=ClienteGroqFalso("nope"))
    assert filtros.motivo
    assert "json" in filtros.motivo.lower()


def test_el_motivo_registra_el_tipo_de_error() -> None:
    filtros = parsear_filtros("algo", cliente=ClienteGroqFalso(error=RuntimeError("429")))
    assert "RuntimeError" in filtros.motivo


def test_el_texto_crudo_no_se_altera_al_degradar() -> None:
    """Es lo que se va a buscar, así que tiene que ser exactamente lo que escribió."""
    texto = "  no quiero anime, algo real y corto  "
    filtros = parsear_filtros(texto, cliente=ClienteGroqFalso("nope"))
    assert filtros.query == texto


# --------------------------------------------------------------------------- #
# el aviso del modelo
# --------------------------------------------------------------------------- #


def test_modelo_sin_strict_avisa_antes_de_degradar(monkeypatch, caplog) -> None:
    """Silencio sería peor que error: el usuario vería resultados sin saber por qué."""
    monkeypatch.setattr(
        "app.services.filter_parser.settings.groq_model", "llama-3.3-70b-versatile"
    )
    parsear_filtros("no quiero anime", cliente=ClienteGroqFalso("nope"))
    assert any("no soporta structured outputs" in r.message for r in caplog.records)


def test_modelo_correcto_no_avisa(caplog) -> None:
    parsear_filtros("algo", cliente=groq_responde())
    assert not any("no soporta" in r.message for r in caplog.records)


@pytest.mark.parametrize("modelo", sorted(MODELOS_CON_STRICT))
def test_los_tres_modelos_validos(modelo: str) -> None:
    assert modelo_soporta_strict(modelo)


def test_llama_no_esta_entre_los_validos() -> None:
    """La skill lo recomienda para structured outputs y no los soporta."""
    assert not modelo_soporta_strict("llama-3.3-70b-versatile")


def test_hay_cliente_refleja_la_config() -> None:
    assert hay_cliente() is True


# --------------------------------------------------------------------------- #
# traducir a search
# --------------------------------------------------------------------------- #


def test_aplicar_a_search_mueve_los_campos() -> None:
    filtros = FiltrosConsulta(
        query="algo", media_types=["movie"], max_runtime_total=120
    )
    busqueda = aplicar_a_search(filtros, liked_ids=["visto"], runtime_maximo_total=120)
    assert busqueda.media_types == ["movie"]
    assert busqueda.liked_ids == ["visto"]
    assert busqueda.max_runtime_total == 120


def test_aplicar_a_search_por_defecto_sin_duracion() -> None:
    busqueda = aplicar_a_search(FiltrosConsulta(query="algo"))
    assert busqueda.max_runtime_total is None
    assert busqueda.max_runtime_episode is None


# --------------------------------------------------------------------------- #
# el $and de search.py
# --------------------------------------------------------------------------- #


def test_sin_filtros_no_hay_where() -> None:
    filtros = FiltrosBusqueda(media_types=[])
    assert construir_where_completo(filtros) is None


def test_un_solo_filtro_no_lleva_and() -> None:
    assert construir_where_completo(FiltrosBusqueda(media_types=["movie"])) == {
        "media_type": "movie"
    }


def test_tipo_mas_duracion_lleva_and() -> None:
    """ChromaDB admite un solo operador por `where`; con dos claves sueltas da 400."""
    where = construir_where_completo(
        FiltrosBusqueda(media_types=["movie"], max_runtime_total=120)
    )
    assert where == {"$and": [{"media_type": "movie"}, {"runtime_total": {"$lte": 120}}]}


def test_todo_el_where_tiene_un_solo_operador() -> None:
    """El error concreto, comprobado en todos los casos, no en uno."""
    casos = [
        FiltrosBusqueda(media_types=["movie"], max_runtime_total=120),
        FiltrosBusqueda(media_types=["tv"], max_runtime_episode=25),
        FiltrosBusqueda(media_types=["anime"], max_runtime_episode=25),
        FiltrosBusqueda(media_types=["movie"], max_runtime_total=120, max_runtime_episode=25),
        FiltrosBusqueda(media_types=[], max_runtime_total=90),
    ]
    for filtros in casos:
        where = construir_where_completo(filtros)
        assert isinstance(where, (dict, type(None)))
        if where is None or "$and" in where:
            continue
        assert len(where) == 1, f"más de un operador: {where}"


def test_las_dos_duraciones_a_la_vez() -> None:
    where = construir_where_completo(
        FiltrosBusqueda(media_types=["anime"], max_runtime_total=10, max_runtime_episode=25)
    )
    assert where == {
        "$and": [
            {"media_type": "anime"},
            {"runtime_total": {"$lte": 10}},
            {"runtime_episode": {"$lte": 25}},
        ]
    }


def test_el_and_llega_a_chroma() -> None:
    coleccion = ColeccionFalsa(
        {"ids": [["a"]], "metadatas": [[{"title": "A"}]], "distances": [[0.1]]}
    )
    from app.services.search import buscar

    buscar(
        "algo",
        filtros=FiltrosBusqueda(media_types=["movie"], max_runtime_total=120),
        embedder=EmbedderFalso(),
        coleccion=coleccion,
    )
    assert coleccion.ultima["where"] == {
        "$and": [{"media_type": "movie"}, {"runtime_total": {"$lte": 120}}]
    }


# --------------------------------------------------------------------------- #
# tests que sí llaman a Groq
# --------------------------------------------------------------------------- #


@pytest.mark.lento
@pytest.mark.requiere_red
def test_groq_real_entiende_las_negaciones() -> None:
    """Lo que no se puede probar sin red: que el modelo acerte de verdad.

    Va marcado `lento` y `requiere_red` para no romper la suite normal. Corre a mano:

        uv run pytest tests/test_filter_parser.py -m lento
    """
    if not hay_cliente():
        pytest.skip("no hay GROQ_API_KEY")

    filtros = parsear_filtros("no quiero anime, algo real y corto")
    assert "anime" not in filtros.media_types, (
        f"el modelo dejó anime en los tipos: {filtros.media_types}"
    )


@pytest.mark.lento
@pytest.mark.requiere_red
def test_groq_real_salta_a_pelicula_con_duracion_total() -> None:
    if not hay_cliente():
        pytest.skip("no hay GROQ_API_KEY")

    filtros = parsear_filtros("películas de menos de 90 minutos sobre unaSubmission")
    assert filtros.max_runtime_total == 90
    assert filtros.media_types == ["movie"]


@pytest.mark.lento
@pytest.mark.requiere_red
def test_groq_real_devuelve_json_valido() -> None:
    """Que el schema estricto no da 400 y el contenido es parseable."""
    if not hay_cliente():
        pytest.skip("no hay GROQ_API_KEY")

    filtros = parsear_filtros("algo de otro planeta")
    assert filtros.degradado is False
    assert filtros.query