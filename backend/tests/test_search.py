"""Tests de `app/services/search.py` y de `app/services/vector_store.py`.

Nada de aquí baja el modelo de verdad: los unitarios usan `EmbedderFalso` de `_fakes` y
`ColeccionFalsa`, que es un doble que devuelve lo que se le pone. No hay red ni pesos.

Los de integración sí abren ChromaDB de verdad, pero sobre `data/index/prueba` y con el
embedder falso. Ojo con lo que eso permite y lo que no: **el ranking no significa nada
**. Ese índice se construyó con vectores e5 reales y aquí se consulta con vectores
inventados de 4 componentes, así que el orden que devuelva es ruido. Por eso los tests de
integración afirman forma, metadatos y filtrado, y nunca "el primero es X". Probar
ranking con esto daría una falsa sensación de cobertura.

Lo que cubren:

  * `genres` de texto con "|" a lista, y lista vacía cuando falta la clave
  * que `genres` nunca sale None, porque `ResultCard.jsx:66` hace `genres.map()`
  * `poster_url` None se propaga sin reventar
  * `year` ausente sale None, no una excepción y no un 0
  * el reescalado del score: rango, orden, y los casos sin nada que reescalar
  * que el reescalado no invierte el orden de las distancias
  * el `where` de media_types con uno, dos y tres tipos
  * que `media_types` se filtra antes de elegir el top, no después
  * el descarte de `liked_ids` y la sobrepetición que lo hace posible
  * que se devuelve menos de top_k antes que rellenar con títulos vistos
  * que el `distance` crudo sobrevive en el resultado pero no viaja al contrato
  * que la salida valida contra el schema real

Y una cosa comprobada midiendo, que motivó el reescalado:

  * `test_el_score_crudo_no_distinguiria_nada`: con las distancias reales medidas
    (0.1368 y 0.1435), el `score` crudo da 0.8632 y 0.8565, y el frontend redondea a
    entero: los dos salen a "86%". Ese es el motivo de que el score sea relativo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _fakes import ColeccionFalsa, EmbedderFalso

from app.models.schemas import Recommendation
from app.services.search import (
    SCORE_MAXIMO,
    SCORE_MINIMO,
    FiltrosBusqueda,
    buscar,
    buscar_como_contrato,
    construir_resultado,
    construir_where,
    parsear_generos,
    reescalar_score,
)
from app.services.vector_store import (
    IndiceNoDisponibleError,
    abrir_coleccion,
    comprobar_modelo,
    leer_index_info,
)

DIR_INDICE_PRUEBA = Path(__file__).resolve().parents[1] / "data" / "index" / "prueba"

# Dimensión que tiene el índice real. No es la del `EmbedderFalso` de 4 componentes:
# ChromaDB rechaza el embedding si no coincide con el de la colección, así que contra un
# índice de verdad hay que dar un vector de 384. Sigue siendo un vector inventado, así
# que el ranking que devuelva no significa nada (afirmamos forma, no orden).
DIMENSION_INDICE = 384


def embedder_real() -> EmbedderFalso:
    return EmbedderFalso(dimension=DIMENSION_INDICE)


def respuesta_falsa(
    ids: list[str],
    metadatas: list[dict],
    distancias: list[float],
) -> dict:
    """Construye el dict que devuelve `query()` de ChromaDB."""
    return {"ids": [ids], "metadatas": [metadatas], "distances": [distancias]}


def meta(**extras) -> dict:
    base = {"media_type": "movie", "title": "Algo", "genres": "Drama"}
    base.update(extras)
    return base


# --------------------------------------------------------------------------- #
# genres
# --------------------------------------------------------------------------- #


def test_genres_se_parten_por_barra() -> None:
    assert parsear_generos({"genres": "Comedia|Crimen|Thriller"}) == [
        "Comedia",
        "Crimen",
        "Thriller",
    ]


def test_genres_ausentes_da_lista_vacia() -> None:
    assert parsear_generos({}) == []


@pytest.mark.parametrize("bruto", [None, "", "|", "||", "  "])
def test_genres_vacios_dan_lista_vacia(bruto: str | None) -> None:
    assert parsear_generos({"genres": bruto}) == []


def test_genres_nunca_devuelve_none() -> None:
    """`ResultCard.jsx:66` hace `genres.map()`; un None aquí rompe el frontend."""
    for meta in ({}, {"genres": None}, {"genres": ""}, {"genres": 42}):
        assert parsear_generos(meta) is not None
        assert isinstance(parsear_generos(meta), list)


def test_genres_recorta_espacios() -> None:
    assert parsear_generos({"genres": "Drama | Crimen "}) == ["Drama", "Crimen"]


def test_genres_que_no_es_texto_se_ignora() -> None:
    """ChromaDB nunca devuelve eso, pero un `None` aquí petaría el frontend."""
    assert parsear_generos({"genres": ["Drama", "Crimen"]}) == []


# --------------------------------------------------------------------------- #
# where
# --------------------------------------------------------------------------- #


def test_where_de_un_solo_tipo() -> None:
    assert construir_where(["movie"]) == {"media_type": "movie"}


def test_where_de_varios_tipos() -> None:
    assert construir_where(["movie", "anime"]) == {
        "media_type": {"$in": ["movie", "anime"]}
    }


def test_where_de_los_tres_tipos() -> None:
    where = construir_where(["movie", "tv", "anime"])
    assert where == {"media_type": {"$in": ["movie", "tv", "anime"]}}


def test_where_vacio_es_none() -> None:
    """`where=None` y `where={}` no son lo mismo en ChromaDB."""
    assert construir_where([]) is None


def test_el_where_viene_a_chroma() -> None:
    coleccion = ColeccionFalsa(respuesta_falsa([], [], []))
    buscar(
        "algo",
        filtros=FiltrosBusqueda(media_types=["movie", "anime"]),
        embedder=EmbedderFalso(),
        coleccion=coleccion,
    )
    assert coleccion.ultima["where"] == {"media_type": {"$in": ["movie", "anime"]}}


# --------------------------------------------------------------------------- #
# reescalado del score
# --------------------------------------------------------------------------- #


def test_score_crudo_no_distinguiria_nada() -> None:
    """Las distancias reales medidas. Este test es el que justifica el reescalado."""
    # Con el score crudo (1 - d) el frontend, que hace Math.round(score * 100), pinta
    # 86% en los dos. Con el reescalado salen 100% y 50%.
    assert round((1 - 0.1368) * 100) == round((1 - 0.1435) * 100) == 86
    assert reescalar_score([0.1368, 0.1435]) == [SCORE_MAXIMO, SCORE_MINIMO]


def test_reescalado_da_el_rango_acordado() -> None:
    scores = reescalar_score([0.10, 0.20, 0.30])
    assert scores[0] == pytest.approx(SCORE_MAXIMO)
    assert scores[-1] == pytest.approx(SCORE_MINIMO)
    assert all(SCORE_MINIMO <= s <= SCORE_MAXIMO for s in scores)


def test_reescalado_conserva_el_orden() -> None:
    """El reescalado no puede cambiar qué título va primero."""
    originales = [0.10, 0.15, 0.90]
    assert reescalar_score(originales) == sorted(
        reescalar_score(originales), reverse=True
    )


def test_reescalado_de_un_solo_resultado() -> None:
    assert reescalar_score([0.42]) == [SCORE_MAXIMO]


def test_reescalado_de_lista_vacia() -> None:
    assert reescalar_score([]) == []


def test_reescalado_si_todas_son_iguales_no_divide_por_cero() -> None:
    """Con dos resultados idénticos no hay amplitud; dar 1.0 a los dos es lo menos raro."""
    scores = reescalar_score([0.50, 0.50])
    assert scores == [SCORE_MAXIMO, SCORE_MAXIMO]


# --------------------------------------------------------------------------- #
# construcción de un resultado
# --------------------------------------------------------------------------- #


def test_resultado_trae_los_campos_del_contrato() -> None:
    r = construir_resultado(
        "tmdb-movie-1",
        meta(title="Interstellar", year=2014, genres="Aventura|Drama", poster_url="https://x/y.jpg"),
        0.13,
        1.0,
    )
    assert r.id == "tmdb-movie-1"
    assert r.title == "Interstellar"
    assert r.media_type == "movie"
    assert r.year == 2014
    assert r.genres == ["Aventura", "Drama"]
    assert r.poster_url == "https://x/y.jpg"


def test_poster_url_none_se_propaga() -> None:
    """El mock del frontend ya trae `poster_url: null`, así que es un caso real."""
    r = construir_resultado("x", meta(poster_url=None), 0.1, 1.0)
    assert r.poster_url is None


def test_poster_url_ausente_no_rompe() -> None:
    assert construir_resultado("x", {}, 0.1, 1.0).poster_url is None


def test_year_ausente_es_none_y_no_cero() -> None:
    """`build_index` omite la clave cuando el año es null, en vez de poner un 0."""
    assert construir_resultado("x", {"year": None}, 0.1, 1.0).year is None
    assert construir_resultado("x", {}, 0.1, 1.0).year is None


def test_metadatos_a_ninguno_no_revienta() -> None:
    r = construir_resultado("x", None, 0.1, 1.0)
    assert r.title == "(sin título)"
    assert r.genres == []


def test_al_contrato_no_lleva_la_distance() -> None:
    """El `distance` es interno: se conserva para depurar, pero no sale en la API."""
    r = construir_resultado("x", meta(), 0.1234, 0.9)
    assert r.distance == pytest.approx(0.1234)
    assert "distance" not in r.al_contrato()
    assert r.al_contrato()["explanation"] is None


# --------------------------------------------------------------------------- #
# buscar()
# --------------------------------------------------------------------------- #


def test_buscar_devuelve_los_distancias_en_orden() -> None:
    coleccion = ColeccionFalsa(
        respuesta_falsa(
            ["a", "b", "c"],
            [meta(title="A"), meta(title="B"), meta(title="C")],
            [0.10, 0.20, 0.30],
        )
    )
    resultados = buscar("algo", embedder=EmbedderFalso(), coleccion=coleccion, top_k=3)
    assert [r.title for r in resultados] == ["A", "B", "C"]
    assert [r.score for r in resultados] == [SCORE_MAXIMO, 0.75, SCORE_MINIMO]


def test_buscar_pide_los_embeddings_de_la_consulta() -> None:
    coleccion = ColeccionFalsa(respuesta_falsa(["a"], [meta()], [0.1]))
    buscar("mi frase", embedder=EmbedderFalso(), coleccion=coleccion)
    assert len(coleccion.ultima["query_embeddings"]) == 1


def test_buscar_sin_resultados_devuelve_lista_vacia() -> None:
    coleccion = ColeccionFalsa({"ids": [[]], "metadatas": [[]], "distances": [[]]})
    assert buscar("nada", embedder=EmbedderFalso(), coleccion=coleccion) == []


def test_buscar_con_respuesta_vacia_no_revienta() -> None:
    """Chroma puede devolver un dict sin las claves si no hay nada."""
    assert buscar("nada", embedder=EmbedderFalso(), coleccion=ColeccionFalsa()) == []


def test_top_k_cero_o_negativo() -> None:
    assert buscar("x", embedder=EmbedderFalso(), coleccion=ColeccionFalsa(), top_k=0) == []


def test_el_filtro_actua_antes_de_elegir_el_top() -> None:
    """El punto: `where` va a ChromaDB, no a un podado en Python.

    Si el filtro fuera posterior, un top-5 con 3 anime devolvería 2, no 5. Al ir en la
    consulta, Chroma entrega los n_results ya válidos.
    """
    coleccion = ColeccionFalsa(respuesta_falsa(["a", "b", "c"], [meta(), meta(), meta()], [0.1, 0.2, 0.3]))
    resultados = buscar(
        "algo",
        filtros=FiltrosBusqueda(media_types=["tv"]),
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        top_k=5,
    )
    assert coleccion.ultima["n_results"] == 5
    assert len(resultados) == 3


# --------------------------------------------------------------------------- #
# liked_ids
# --------------------------------------------------------------------------- #


def test_liked_ids_se_descartan() -> None:
    coleccion = ColeccionFalsa(
        respuesta_falsa(
            ["visto-1", "nuevo-1", "nuevo-2"],
            [meta(title="Visto"), meta(title="Nuevo 1"), meta(title="Nuevo 2")],
            [0.10, 0.20, 0.30],
        )
    )
    resultados = buscar(
        "algo",
        filtros=FiltrosBusqueda(liked_ids=["visto-1"]),
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        top_k=2,
    )
    assert [r.id for r in resultados] == ["nuevo-1", "nuevo-2"]


def test_liked_ids_piden_mas_para_poder_podar() -> None:
    """Sin sobrepetición no hay de dónde descartar."""
    coleccion = ColeccionFalsa(respuesta_falsa(["a"], [meta()], [0.1]))
    buscar(
        "algo",
        filtros=FiltrosBusqueda(liked_ids=["x"]),
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        top_k=5,
    )
    assert coleccion.ultima["n_results"] == 10


def test_liked_ids_sin_sobrepeticion_tiene_tope() -> None:
    """top_k grande no debe pedir miles de vectores."""
    coleccion = ColeccionFalsa(respuesta_falsa(["a"], [meta()], [0.1]))
    buscar(
        "algo",
        filtros=FiltrosBusqueda(liked_ids=["x"]),
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        top_k=1000,
    )
    assert coleccion.ultima["n_results"] == 50


def test_sin_liked_ids_no_sobrepide() -> None:
    coleccion = ColeccionFalsa(respuesta_falsa(["a"], [meta()], [0.1]))
    buscar("algo", embedder=EmbedderFalso(), coleccion=coleccion, top_k=5)
    assert coleccion.ultima["n_results"] == 5


def test_el_reescalado_ignora_los_descartados() -> None:
    """Si se reescalara antes de podar, el mejor real no llegaría a SCORE_MAXIMO."""
    coleccion = ColeccionFalsa(
        respuesta_falsa(
            ["visto", "a", "b"],
            [meta(title="Visto"), meta(title="A"), meta(title="B")],
            [0.01, 0.10, 0.30],
        )
    )
    resultados = buscar(
        "algo",
        filtros=FiltrosBusqueda(liked_ids=["visto"]),
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        top_k=2,
    )
    assert resultados[0].score == pytest.approx(SCORE_MAXIMO)


def test_devuelve_menos_antes_que_rellenar() -> None:
    """Con todo el top ya visto, se devuelve vacío. Rellenar sería peor."""
    coleccion = ColeccionFalsa(
        respuesta_falsa(
            ["a", "b"],
            [meta(title="A"), meta(title="B")],
            [0.10, 0.20],
        )
    )
    resultados = buscar(
        "algo",
        filtros=FiltrosBusqueda(liked_ids=["a", "b"]),
        embedder=EmbedderFalso(),
        coleccion=coleccion,
        top_k=2,
    )
    assert resultados == []


def test_nunca_devuelve_mas_de_top_k() -> None:
    coleccion = ColeccionFalsa(
        respuesta_falsa(
            ["a", "b", "c", "d"],
            [meta(title=t) for t in "ABCD"],
            [0.10, 0.20, 0.30, 0.40],
        )
    )
    assert len(buscar("algo", embedder=EmbedderFalso(), coleccion=coleccion, top_k=2)) == 2


# --------------------------------------------------------------------------- #
# forma del contrato
# --------------------------------------------------------------------------- #


def test_la_salida_valida_contra_el_schema() -> None:
    """Si el índice cambia de forma, revienta aquí y no en el navegador."""
    coleccion = ColeccionFalsa(
        respuesta_falsa(
            ["tmdb-movie-1", "mal-anime-2"],
            [
                meta(title="A", year=2001, genres="Drama", poster_url="https://x.jpg"),
                meta(media_type="anime", title="B", year=None, genres="", poster_url=None),
            ],
            [0.10, 0.20],
        )
    )
    respuesta = buscar_como_contrato(
        "algo", top_k=2, embedder=EmbedderFalso(), coleccion=coleccion
    )
    validados = [Recommendation.model_validate(r) for r in respuesta["results"]]
    assert [v.id for v in validados] == ["tmdb-movie-1", "mal-anime-2"]
    assert validados[1].poster_url is None
    assert validados[1].year is None
    assert all(v.explanation is None for v in validados)


def test_explanation_ya_no_es_obligatoria() -> None:
    """El schema se relajó para que la búsqueda viva sin Groq."""
    r = Recommendation.model_validate(
        {
            "id": "x",
            "title": "Algo",
            "media_type": "movie",
            "score": 0.9,
        }
    )
    assert r.explanation is None


def test_buscar_como_contrato_pasa_los_filtros() -> None:
    coleccion = ColeccionFalsa(respuesta_falsa(["a"], [meta()], [0.1]))
    buscar_como_contrato(
        "algo",
        media_types=["anime"],
        liked_ids=["visto"],
        top_k=3,
        embedder=EmbedderFalso(),
        coleccion=coleccion,
    )
    assert coleccion.ultima["where"] == {"media_type": "anime"}
    assert coleccion.ultima["n_results"] == 6


# --------------------------------------------------------------------------- #
# integración contra el índice real de prueba
# --------------------------------------------------------------------------- #

comprobacion = pytest.mark.skipif(
    not DIR_INDICE_PRUEBA.exists(),
    reason="requiere data/index/prueba (se construye con scripts.build_index --limit 200)",
)


@comprobacion
def test_indice_de_prueba_se_abre() -> None:
    coleccion = abrir_coleccion(DIR_INDICE_PRUEBA)
    assert coleccion.count() > 0


@comprobacion
def test_index_info_dice_cuantos_hay() -> None:
    if not DIR_INDICE_PRUEBA.exists():
        pytest.skip("no hay índice de prueba")
    info = leer_index_info(DIR_INDICE_PRUEBA)
    assert info is not None
    assert info["documentos"] > 0
    assert info["modelo"] == "intfloat/multilingual-e5-small"


def test_indice_inexistente_da_error_propio() -> None:
    with pytest.raises(IndiceNoDisponibleError):
        abrir_coleccion(Path("/tmp/no-existe-este-indice-moodle"))


def test_comprobar_modelo_avisa_si_no_coincide(caplog) -> None:
    comprobar_modelo({"modelo": "otro/modelo"}, EmbedderFalso())
    assert any("no son comparables" in r.message for r in caplog.records)


def test_comprobar_modelo_no_dice_nada_si_no_hay_info(caplog) -> None:
    comprobar_modelo(None, EmbedderFalso())
    assert not caplog.records


@comprobacion
def test_consulta_contra_el_indice_de_prueba_devuelve_forma_valida() -> None:
    """Afirma forma, no ranking.

    El índice de prueba guarda vectores e5 de verdad y aquí se le pasa un vector falso de
    4 componentes, así que el orden que salga es ruido. Lo único que afirma esto es que
    el camino completo funciona y que lo que sale cumple el contrato.
    """
    resultados = buscar(
        "algo de otro planeta",
        filtros=FiltrosBusqueda(media_types=["movie"]),
        top_k=3,
        embedder=embedder_real(),
        dir_indice=DIR_INDICE_PRUEBA,
    )
    assert len(resultados) <= 3
    for r in resultados:
        Recommendation.model_validate(r.al_contrato())
        assert r.media_type == "movie"
        assert isinstance(r.genres, list)
        assert SCORE_MINIMO <= r.score <= SCORE_MAXIMO


@comprobacion
def test_el_filtro_reduce_contra_el_indice_de_prueba() -> None:
    """Aquí el filtro sí es verificable: `media_type` viene de los metadatos reales."""
    todos = buscar("algo", top_k=10, embedder=embedder_real(), dir_indice=DIR_INDICE_PRUEBA)
    solo_peliculas = buscar(
        "algo",
        filtros=FiltrosBusqueda(media_types=["movie"]),
        top_k=10,
        embedder=embedder_real(),
        dir_indice=DIR_INDICE_PRUEBA,
    )
    assert all(r.media_type == "movie" for r in solo_peliculas)
    assert len(solo_peliculas) <= len(todos)


@comprobacion
def test_liked_ids_contra_el_indice_de_prueba() -> None:
    todos = buscar("algo", top_k=5, embedder=embedder_real(), dir_indice=DIR_INDICE_PRUEBA)
    if not todos:
        pytest.skip("el índice de prueba no devolvió nada")
    vistos = [todos[0].id]
    restantes = buscar(
        "algo",
        filtros=FiltrosBusqueda(liked_ids=vistos),
        top_k=5,
        embedder=embedder_real(),
        dir_indice=DIR_INDICE_PRUEBA,
    )
    assert todos[0].id not in {r.id for r in restantes}