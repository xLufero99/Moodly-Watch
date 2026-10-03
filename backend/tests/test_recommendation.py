"""Tests de `app/services/recommendation.py`: la regla de mezcla y el pipeline entero.

**Nada de aquí habla con Groq.** Parser y explainer van con `ClienteGroqFalso`.

El centro son los dos estados de la lista de tipos, que es donde está el bug que este
módulo existe para arreglar:

    None -> nadie pidió nada -> se busca en todo el catálogo
    []   -> los dos se contradicen -> cero resultados, sin llamar a ChromaDB

Si los dos se representaran con `[]`, "no quiero anime" con anime marcado en el selector
devolvería anime otra vez, y sería el mismo fallo que motivó todo el trabajo del parser.

Lo que cubren:

  * los cuatro casos de la mezcla, incluido que la intersección vacía da `[]` y no `None`
  * que la contradicción **no llama a ChromaDB**, que es lo que la hace barata
  * que el pipeline entero monta parser + búsqueda + explainer en ese orden
  * que las explicaciones se pegan a la tarjeta correcta por `id`
  * que `explanation` llega al contrato como texto y nunca como `None`
  * que sin Groq el endpoint responde 200 con las plantillas puestas
  * que el contrato del frontend no cambia: los mismos tests de antes, sin tocar
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _fakes import ClienteGroqFalso, ColeccionFalsa, EmbedderFalso

from app.services.recommendation import (
    TOP_K,
    aplicar_explicaciones,
    mezclar_media_types,
    recomendar,
)
from app.services.search import ResultadoBusqueda

DIR_INDICE_PRUEBA = Path(__file__).resolve().parents[1] / "data" / "index" / "prueba"

# Groq respondiendo algo que no es JSON. Va como constante y no construyendo un doble y
# leyendo su atributo, que sería raro de leer.
NO_ES_JSON = "esto no es un JSON, es una frase"


@pytest.fixture(autouse=True)
def _sin_key(request, monkeypatch):
    """Sin key real en los unitarios, y con la real intacta en los `lento`.

    Los unitarios miden la regla de mezcla, no la calidad de las explicaciones. Que el
    explainer caiga a plantilla es lo correcto aquí y lo que hay que poder comprobar.

    El salto para los `lento` es obligatorio, no cosmético: sin él, este `autouse` les
    quita la key y los tests "contra Groq real" pasan por la vía de la plantilla mientras
    parecen haber probado el modelo. Es exactamente el fallo que ya desapareció una vez en
    `test_filter_parser.py`, y en el que se cayó por usar `return` en vez de `yield`: un
    fixture generador tiene que ceder el control siempre, o pytest lo reporta como
    "did not yield a value".
    """
    if not request.node.get_closest_marker("lento"):
        monkeypatch.setattr("app.services.explainer.hay_cliente", lambda: False)
        monkeypatch.setattr("app.services.filter_parser.hay_cliente", lambda: False)
    yield


def meta_para(identificador: str, **extra) -> dict:
    """Unos metadatos de catálogo, con `genres` como los guarda ChromaDB: texto con barras."""
    base = {
        "title": f"Título {identificador}",
        "media_type": "movie",
        "year": 2000,
        "genres": "Comedia|Drama",
        "poster_url": None,
    }
    base.update(extra)
    return base


def ids(cantidad: int = 6) -> list[str]:
    return [str(n) for n in range(1, cantidad + 1)]


def coleccion_con(los_ids: list[str]):
    """Una colección que devuelve `los_ids` con la forma que espera `buscar`."""
    return ColeccionFalsa(
        {
            "ids": [los_ids],
            "metadatas": [[meta_para(i) for i in los_ids]],
            "distances": [[0.1 * (n + 1) for n in range(len(los_ids))]],
            "documents": [
                [
                    f"{i}. Tipo: movie. Géneros: Comedia. Temas: x. Una sinopsis."
                    for i in los_ids
                ]
            ],
        }
    )


def respuesta_parser(**campos) -> str:
    """El JSON que el parser devuelve, con los cuatro campos del schema."""
    base = {
        "query": "algo",
        "media_types": [],
        "max_runtime_total": None,
        "max_runtime_episode": None,
    }
    base.update(campos)
    return json.dumps(base)


@pytest.fixture
def con_groq(monkeypatch):
    """Convierte los dos `hay_cliente` en True, para los tests que sí simulan el LLM."""
    monkeypatch.setattr("app.services.explainer.hay_cliente", lambda: True)
    monkeypatch.setattr("app.services.filter_parser.hay_cliente", lambda: True)


# --------------------------------------------------------------------------- #
# la mezcla de media_types
# --------------------------------------------------------------------------- #


def test_ninguno_pide_nada_es_none() -> None:
    """`None` y no `[]`: `None` significa "buscar sin filtro"."""
    assert mezclar_media_types(None, None) is None
    assert mezclar_media_types([], []) is None


def test_solo_el_request_pide_manda_el_request() -> None:
    assert mezclar_media_types(["anime"], []) == ["anime"]
    assert mezclar_media_types(["anime"], None) == ["anime"]


def test_solo_el_parser_pide_manda_el_parser() -> None:
    assert mezclar_media_types([], ["movie"]) == ["movie"]
    assert mezclar_media_types(None, ["movie"]) == ["movie"]


def test_los_dos_piden_manda_la_interseccion() -> None:
    """El selector es un filtro y la frase escrita es una exclusión.

    Si pide "anime y tv" en la UI y el texto solo habla de anime, gana anime.
    """
    assert mezclar_media_types(["anime", "tv"], ["anime"]) == ["anime"]
    assert mezclar_media_types(["movie", "tv"], ["tv", "anime"]) == ["tv"]


def test_interseccion_vacia_es_lista_vacia_y_no_none() -> None:
    """El matiz que lo cambia todo.

    `[]` no es "sin filtro": es "cero resultados". Si aquí se devolviera `None`, buscaría
    en todo el catálogo y devolvería anime a quien acaba de escribir "no quiero anime".
    """
    assert mezclar_media_types(["anime"], ["movie"]) == []
    assert mezclar_media_types(["movie"], ["anime"]) == []


def test_interseccion_parcial_no_es_vacia() -> None:
    """Anime contra anime+tv deja anime, no vacío. Una intersección vacía solo cuando no
    hay ni un tipo en común."""
    assert mezclar_media_types(["anime", "tv"], ["tv", "movie"]) == ["tv"]
    assert mezclar_media_types(["anime"], ["anime"]) == ["anime"]


def test_la_interseccion_vacia_avisa(caplog) -> None:
    mezclar_media_types(["anime"], ["movie"])
    assert any("no se cruzan" in r.message for r in caplog.records)


def test_la_mezcla_no_altera_las_entradas() -> None:
    """Que devuelva copias: si devolviera la lista del request, quien la modificara
    después cambiaría un filtro ya aplicado."""
    entrada = ["anime"]
    salida = mezclar_media_types(entrada, [])
    salida.append("tv")
    assert entrada == ["anime"]


# --------------------------------------------------------------------------- #
# la contradicción no busca
# --------------------------------------------------------------------------- #


def test_contradiccion_no_llama_a_chroma(con_groq) -> None:
    """Lo que hace que la intersección vacía sea barata y no solo correcta.

    El caso: anime marcado en el selector y "no quiero anime" en el texto. El parser dice
    `["movie", "tv"]`, la intersección con `["anime"]` es `[]`, y `[]` significa cero
    resultados. Si esto devolviera el catálogo entero, sería el bug que todo el módulo
    existe para no tener.
    """
    coleccion = coleccion_con(ids())
    cuerpo = recomendar(
        "no quiero anime, algo de acción",
        media_types=["anime"],
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        cliente=ClienteGroqFalso(
            respuesta_parser(query="acción", media_types=["movie", "tv"])
        ),
    )
    assert cuerpo == {"results": []}
    assert coleccion.llamadas == []


def test_sin_contradiccion_cuando_el_texto_no_dice_tipos(con_groq) -> None:
    """El parser sin tipos y anime en el selector: manda el selector y se busca anime.

    Es el caso del `test_recommend_filters_by_media_type` del contrato, y es el que obliga
    a que "sin tipos del parser" se guarde como lista vacía y no como `None`.
    """
    coleccion = coleccion_con(ids())
    cuerpo = recomendar(
        "quiero algo de ciencia ficción pero con tensión",
        media_types=["anime"],
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        cliente=ClienteGroqFalso(respuesta_parser(query="ciencia ficción con tensión")),
    )
    assert len(cuerpo["results"]) == 6
    assert coleccion.ultima["where"] == {"media_type": "anime"}


def test_sin_contradiccion_si_el_parser_degrada() -> None:
    """Con el parser degradado no hay tipos del parser, así que manda el request y no hay
    contradicción que detectar."""
    coleccion = coleccion_con(ids())
    cuerpo = recomendar(
        "no quiero anime",
        media_types=["movie"],
        embedder=EmbedderFalso(),
        coleccion=coleccion,
    )
    assert cuerpo["results"]
    assert coleccion.ultima["where"] == {"media_type": "movie"}


# --------------------------------------------------------------------------- #
# el pipeline
# --------------------------------------------------------------------------- #


def test_el_pipeline_pide_seis() -> None:
    """El contrato limita a seis y es lo que el explainer clava en el schema."""
    assert TOP_K == 6


def test_el_pipeline_pega_las_explicaciones_por_id(con_groq) -> None:
    """Cada explicación en su tarjeta, aunque el explainer las devuelva en otro orden."""
    explicaciones = [{"id": i, "explicacion": f"Texto de {i}"} for i in reversed(ids())]
    cliente = ClienteGroqFalso(
        respuestas=[
            respuesta_parser(query="algo", media_types=["movie"]),
            json.dumps({"explicaciones": explicaciones}),
        ]
    )

    cuerpo = recomendar(
        "algo",
        media_types=["movie"],
        embedder=EmbedderFalso(),
        coleccion=coleccion_con(ids()),
        cliente=cliente,
    )

    assert [r["id"] for r in cuerpo["results"]] == ids()
    for item in cuerpo["results"]:
        assert item["explanation"] == f"Texto de {item['id']}"


def test_el_where_lleva_los_tipos_mezclados(con_groq) -> None:
    """La mezcla se aplica antes de buscar, y lo que sale va al `where`."""
    coleccion = coleccion_con(ids())
    recomendar(
        "no quiero anime",
        media_types=["movie", "anime"],
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        cliente=ClienteGroqFalso(respuesta_parser(query="algo", media_types=["movie"])),
    )
    assert coleccion.ultima["where"] == {"media_type": "movie"}


def test_el_pipeline_sin_explicacion_no_deja_none(con_groq) -> None:
    """Si el explainer no trae texto para un resultado, ese lleva plantilla.

    Un `explanation: null` en el contrato pinta un párrafo vacío en el frontend, que es
    peor que un texto poor pero cierto.
    """
    cuerpo = recomendar(
        "algo",
        media_types=["movie"],
        embedder=EmbedderFalso(),
        coleccion=coleccion_con(ids()),
        cliente=ClienteGroqFalso(
            respuestas=[respuesta_parser(), json.dumps({"explicaciones": []})]
        ),
    )
    for item in cuerpo["results"]:
        assert isinstance(item["explanation"], str)
        assert item["explanation"].strip()


def test_explicacion_por_generos_si_el_explainer_falla(con_groq) -> None:
    """El explainer caído no puede dejar tarjetas sin texto."""
    cuerpo = recomendar(
        "algo",
        media_types=["movie"],
        embedder=EmbedderFalso(),
        coleccion=coleccion_con(ids()),
        cliente=ClienteGroqFalso(
            respuestas=[respuesta_parser(), NO_ES_JSON]
        ),
    )
    assert len(cuerpo["results"]) == 6
    for item in cuerpo["results"]:
        assert "Comedia" in item["explanation"]


def test_sin_resultados_no_se_explica(con_groq) -> None:
    """Nada que explicar, y una llamada menos."""
    cliente = ClienteGroqFalso(respuestas=[respuesta_parser()])
    cuerpo = recomendar(
        "algo",
        media_types=["movie"],
        embedder=EmbedderFalso(),
        coleccion=ColeccionFalsa({}),
        cliente=cliente,
    )
    assert cuerpo == {"results": []}
    assert len(cliente.llamadas) == 1, "solo la del parser"


def test_aplicar_explicaciones_no_altera_el_original() -> None:
    """`ResultadoBusqueda` es frozen; si se mutara, el explainer y el buscador se
    contaminarían entre llamadas."""
    original = ResultadoBusqueda(
        id="1",
        title="T",
        media_type="movie",
        year=2000,
        genres=[],
        poster_url=None,
        score=1.0,
    )
    desde_el_explainer = [type("E", (), {"id": "1", "texto": "Explicación"})()]
    resultado = aplicar_explicaciones([original], desde_el_explainer)[0]
    assert resultado.explanation == "Explicación"
    assert original.explanation is None


def test_aplicar_explicaciones_pone_plantilla_a_lo_que_falta() -> None:
    original = ResultadoBusqueda(
        id="2",
        title="T",
        media_type="movie",
        year=2000,
        genres=["Terror"],
        poster_url=None,
        score=1.0,
    )
    desde_el_explainer = [type("E", (), {"id": "otro", "texto": "X"})()]
    resultado = aplicar_explicaciones([original], desde_el_explainer)[0]
    assert "Terror" in resultado.explanation


# --------------------------------------------------------------------------- #
# integración contra el índice de prueba
# --------------------------------------------------------------------------- #

comprobacion = pytest.mark.skipif(
    not DIR_INDICE_PRUEBA.exists(),
    reason="requiere data/index/prueba (se construye con scripts.build_index --limit 200)",
)


@comprobacion
def test_recomendar_contra_el_indice_de_prueba() -> None:
    """El camino entero con índice real, embeddings reales y Groq degrading a plantilla."""
    cuerpo = recomendar(
        "algo de otro planeta",
        media_types=["movie"],
        dir_indice=DIR_INDICE_PRUEBA,
    )
    assert cuerpo["results"]
    for item in cuerpo["results"]:
        assert set(item) == {
            "id",
            "title",
            "media_type",
            "year",
            "genres",
            "poster_url",
            "score",
            "explanation",
        }
        assert item["media_type"] == "movie"
        assert isinstance(item["explanation"], str)
        assert item["explanation"].strip()


@comprobacion
def test_recomendar_trae_sinopsis_al_explainer() -> None:
    """Con índice real llega el `document`, así que la sinopsis no está vacía.

    Es el motivo del `include` con `documents` en `search.py`: sin él, el explainer solo
    tendría título y géneros.
    """
    from app.services.explainer import sinopsis_de
    from app.services.search import buscar

    resultados = buscar("algo", top_k=1, dir_indice=DIR_INDICE_PRUEBA)
    assert resultados, "el índice de prueba no devolvió nada"
    assert resultados[0].document is not None
    assert sinopsis_de(resultados[0].document)


# --------------------------------------------------------------------------- #
# lentos: el pipeline completo contra Groq de verdad
# --------------------------------------------------------------------------- #


@pytest.mark.lento
@pytest.mark.requiere_red
def test_pipeline_completo_contra_groq_real() -> None:
    """Parser, búsqueda y explainer contra el modelo de verdad y el índice completo.

    Es el test que demuestra que las tres piezas se hablan: si el parser degradara, el
    explainer lo diría; si el explainer degradara, las explicaciones serían de plantilla.
    """
    from app.services.vector_store import ruta_indice_por_defecto

    cuerpo = recomendar(
        "quiero algo de ciencia ficción pero con tensión",
        media_types=["movie", "tv", "anime"],
        dir_indice=ruta_indice_por_defecto(),
    )
    assert cuerpo["results"]
    assert all(
        isinstance(r["explanation"], str) and r["explanation"].strip()
        for r in cuerpo["results"]
    )
    print("\n  Explicaciones reales:")
    for item in cuerpo["results"]:
        print(f"    [{item['media_type']}] {item['title']}: {item['explanation']}")


@pytest.mark.lento
@pytest.mark.requiere_red
def test_negacion_real_de_extremo_a_extremo() -> None:
    """La prueba de que todo este trabajo sirve para algo.

    "no quiero anime" con el parser real y el índice real no debe devolver anime. Es la
    razón por la que existe el filtro: un vector cercano no es un vector opuesto.
    """
    from app.services.vector_store import ruta_indice_por_defecto

    cuerpo = recomendar(
        "no quiero anime, algo real y con tensión",
        media_types=["movie", "tv", "anime"],
        dir_indice=ruta_indice_por_defecto(),
    )
    assert cuerpo["results"], "no devolvió nada"
    assert all(item["media_type"] != "anime" for item in cuerpo["results"])