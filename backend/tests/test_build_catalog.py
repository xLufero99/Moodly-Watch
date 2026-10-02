"""Tests de scripts/build_catalog.py: sin red, con JSON mínimos en tmp_path."""

import json

import pandas as pd
import pyarrow as pa
import pytest

from scripts import build_catalog
from scripts.build_catalog import Contexto

MAL_PIE = "[Written by MAL Rewrite]"


# --------------------------------------------------------------------------- fixtures


def detalle_pelicula(pelicula_id: int = 1, **overrides) -> dict:
    """Detalle de movie de TMDB con los campos que lee el script."""
    base = {
        "id": pelicula_id,
        "adult": False,
        "status": "Released",
        "title": "Matrix",
        "original_title": "The Matrix",
        "release_date": "1999-03-30",
        "overview": "Un hacker descubre la realidad.",
        "genres": [{"id": 28, "name": "Ciencia ficción"}, {"id": 878, "name": "Acción"}],
        "keywords": {"keywords": [{"id": 1, "name": "distopía"}]},
        "original_language": "en",
        "poster_path": "/matrix.jpg",
        "runtime": 136,
        "vote_average": 8.2,
        "vote_count": 24000,
        "popularity": 78.5,
    }
    return {**base, **overrides}


def detalle_serie(serie_id: int = 10, **overrides) -> dict:
    """Detalle de tv de TMDB. Ojo: usa `name`, no `title`, y no trae `runtime`."""
    base = {
        "id": serie_id,
        "adult": False,
        "status": "Ended",
        "name": "The Last of Us",
        "original_name": "The Last of Us",
        "first_air_date": "2023-01-15",
        "overview": "Joel y Ellie cruzan el país.",
        "genres": [{"id": 18, "name": "Sci-Fi & Fantasy"}],
        "keywords": {"results": [{"id": 1, "name": "infeccion"}]},
        "original_language": "en",
        "poster_path": "/tlou.jpg",
        "episode_run_time": [50],
        "number_of_episodes": 9,
        "number_of_seasons": 1,
        "vote_average": 8.6,
        "vote_count": 5200,
        "popularity": 90.1,
    }
    return {**base, **overrides}


def item_anime(anime_id: int = 5114, **overrides) -> dict:
    """Item de anime.json, ya aplanado por fetch_mal.py."""
    base = {
        "id": anime_id,
        "title": "Fullmetal Alchemist: Brotherhood",
        "title_english": "Fullmetal Alchemist: Brotherhood",
        "title_japanese": "鋼の錬金術師",
        "media_type": "tv",
        "num_episodes": 64,
        "status": "finished_airing",
        "duration_minutes": 24,
        "rating": "r",
        "mean": 9.2,
        "num_scoring_users": 900000,
        "popularity": 12,
        "synopsis": "Dos hermanos buscan elphisector.",
        "year": 2009,
        "start_date": "2009-04-05",
        # Ojo: en anime.json los géneros son cadenas planas, no {"id", "name"}
        # como en TMDB.
        "genres": ["Adventure", "Action"],
        "poster_url": "https://cdn.example/hagaren.jpg",
    }
    return {**base, **overrides}


def escribir(directorio, nombre: str, payload) -> None:
    directorio.mkdir(parents=True, exist_ok=True)
    (directorio / nombre).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def arbol_completo(tmp_path):
    """Escribe un movie, un tv y un anime.json y devuelve las tres rutas."""
    tmdb = tmp_path / "tmdb"
    escribir(tmdb / "movie", "1.json", detalle_pelicula())
    escribir(tmdb / "tv", "10.json", detalle_serie())
    mal = tmp_path / "mal" / "anime.json"
    escribir(mal.parent, "anime.json", [item_anime()])
    return tmdb, mal


def construir(tmp_path):
    """Arranca un catálogo de tres filas y devuelve (dataframe, contexto)."""
    tmdb, mal = arbol_completo(tmp_path)
    ctx = Contexto()
    filas = (
        build_catalog.leer_tmdb(tmdb, "movie", build_catalog.fila_de_pelicula, ctx)
        + build_catalog.leer_tmdb(tmdb, "tv", build_catalog.fila_de_serie, ctx)
        + build_catalog.leer_mal(mal, ctx)
    )
    return build_catalog.construir_dataframe(filas), ctx


# ------------------------------------------------------------------------------ géneros


def test_generos_pelicula_mapean_al_vocabulario():
    ctx = Contexto()
    fila = build_catalog.fila_de_pelicula(detalle_pelicula(), ctx)
    # TMDB ya trae los nombres en español y caen 1:1 en el vocabulario canónico.
    assert fila["genres"] == ["Ciencia ficción", "Acción"]
    assert "distopía" in fila["keywords"]


@pytest.mark.parametrize(
    ("nombre_fuente", "esperado"),
    [
        ("Sci-Fi & Fantasy", ["Ciencia ficción", "Fantasía"]),
        ("Action & Adventure", ["Acción", "Aventura"]),
        ("War & Politics", ["Bélica"]),
    ],
)
def test_generos_combinados_de_serie_se_expanden(nombre_fuente, esperado):
    """Las series de TMDB traen estas etiquetas combinadas y hay que abrirlas."""
    detalle = detalle_serie(genres=[{"id": 1, "name": nombre_fuente}])
    fila = build_catalog.fila_de_serie(detalle, Contexto())
    assert fila["genres"] == esperado
    assert nombre_fuente not in fila["keywords"], "el género se coló en keywords"


@pytest.mark.parametrize("nombre_fuente", ["Kids", "Soap", "Reality", "News", "Talk"])
def test_generos_de_serie_sin_equivalente_van_a_keywords(nombre_fuente):
    detalle = detalle_serie(genres=[{"id": 1, "name": nombre_fuente}])
    fila = build_catalog.fila_de_serie(detalle, Contexto())
    assert fila["genres"] == []
    assert nombre_fuente in fila["keywords"]


def test_genero_pelicula_sin_equivalente_cae_a_keywords():
    """`Película de TV` no tiene sitio en el vocabulario."""
    detalle = detalle_pelicula(genres=[{"id": 10759, "name": "Película de TV"}])
    ctx = Contexto()
    fila = build_catalog.fila_de_pelicula(detalle, ctx)
    assert fila["genres"] == []
    assert "Película de TV" in fila["keywords"]
    assert ctx.generos_no_mapeados["Película de TV"] == 1


@pytest.mark.parametrize(
    ("nombre_mal", "esperado"),
    [
        ("Action", ["Acción"]),
        ("Sci-Fi", ["Ciencia ficción"]),
        ("Military", ["Bélica"]),
        ("Slice of Life", ["Slice of Life"]),
        ("Team Sports", ["Deportes"]),
    ],
)
def test_generos_mal_mapean_los_equivalentes_obvios(nombre_mal, esperado):
    item = item_anime(genres=[nombre_mal])
    fila = build_catalog.fila_de_anime(item, Contexto())
    assert fila["genres"] == esperado


@pytest.mark.parametrize("nombre_mal", ["Shounen", "Mecha", "Isekai", "Ecchi"])
def test_generos_mal_que_no_cuadran_caen_a_keywords(nombre_mal):
    """Sin equivalente obvio va a keywords, no a un género forzado."""
    item = item_anime(genres=[nombre_mal])
    fila = build_catalog.fila_de_anime(item, Contexto())
    assert fila["genres"] == []
    assert fila["keywords"] == [nombre_mal]


def test_todo_genero_canonico_se_mapea_en_tmdb():
    """El vocabulario canónico no puede tener etiquetas que TMDB no sepa mapear."""
    for nombre in build_catalog.GENEROS_CANONICOS:
        assert build_catalog.MAPA_TMDB_PELICULA.get(nombre) is not None, nombre
        assert build_catalog.MAPA_TMDB_SERIE.get(nombre) is not None, nombre


def test_los_tres_combinados_de_serie_estan_en_el_mapa():
    """Si TMDB añade otro combinado, este test avisa en vez de ir a keywords."""
    mapa = build_catalog.MAPA_TMDB_SERIE
    assert mapa["Sci-Fi & Fantasy"] == ["Ciencia ficción", "Fantasía"]
    assert mapa["Action & Adventure"] == ["Acción", "Aventura"]
    assert mapa["War & Politics"] == ["Bélica"]


# ------------------------------------------------------------------------------ runtime


def test_runtime_pelicula_usa_la_duracion_total():
    fila = build_catalog.fila_de_pelicula(detalle_pelicula(runtime=136), Contexto())
    assert fila["runtime"] == 136


def test_runtime_serie_usa_el_primer_valor_de_episode_run_time():
    detalle = detalle_serie(episode_run_time=[45, 60])
    fila = build_catalog.fila_de_serie(detalle, Contexto())
    assert fila["runtime"] == 45


def test_runtime_serie_cae_a_last_episode_to_air():
    """525 de las 1163 series reales vienen sin episode_run_time."""
    detalle = detalle_serie(episode_run_time=[])
    detalle["last_episode_to_air"] = {"id": 1, "runtime": 58}
    fila = build_catalog.fila_de_serie(detalle, Contexto())
    assert fila["runtime"] == 58


def test_runtime_serie_cae_a_next_episode_to_air():
    detalle = detalle_serie(episode_run_time=[])
    detalle["next_episode_to_air"] = {"id": 1, "runtime": 22}
    fila = build_catalog.fila_de_serie(detalle, Contexto())
    assert fila["runtime"] == 22


def test_runtime_serie_null_si_no_hay_ningun_campo_de_duracion():
    detalle = detalle_serie(episode_run_time=[])
    assert build_catalog.fila_de_serie(detalle, Contexto())["runtime"] is None


def test_runtime_serie_ignora_un_runtime_a_cero():
    detalle = detalle_serie(episode_run_time=[0], last_episode_to_air={"runtime": 0})
    assert build_catalog.fila_de_serie(detalle, Contexto())["runtime"] is None


def test_runtime_anime_usa_duration_minutes():
    fila = build_catalog.fila_de_anime(item_anime(duration_minutes=24), Contexto())
    assert fila["runtime"] == 24


def test_runtime_anime_null_si_duracion_es_cero():
    fila = build_catalog.fila_de_anime(item_anime(duration_minutes=0), Contexto())
    assert fila["runtime"] is None


# --------------------------------------------------------------------------- exclusiones


@pytest.mark.parametrize(
    ("campo", "valor", "motivo"),
    [
        ("adult", True, "tmdb_adulto"),
        ("status", "In Production", "tmdb_no_estrenado"),
        ("status", "Planned", "tmdb_no_estrenado"),
        ("status", "Post Production", "tmdb_no_estrenado"),
        ("overview", "", "tmdb_sinopsis_vacia"),
        ("overview", "   ", "tmdb_sinopsis_vacia"),
    ],
)
def test_se_excluyen_peliculas(campo, valor, motivo):
    ctx = Contexto()
    detalle = detalle_pelicula(**{campo: valor})
    assert build_catalog.fila_de_pelicula(detalle, ctx) is None
    assert ctx.descartes[motivo] == 1


def test_se_excluyen_series(caplog):
    ctx = Contexto()
    assert build_catalog.fila_de_serie(detalle_serie(adult=True), ctx) is None
    assert build_catalog.fila_de_serie(detalle_serie(status="Planned"), ctx) is None
    assert ctx.descartes["tmdb_adulto"] == 1
    assert ctx.descartes["tmdb_no_estrenado"] == 1


def test_cancelada_no_se_excluye():
    """Una serie cancelada ya existentió: se queda, con status `other`."""
    fila = build_catalog.fila_de_serie(detalle_serie(status="Canceled"), Contexto())
    assert fila["status"] == "other"


def test_se_excluye_anime_rx():
    ctx = Contexto()
    assert build_catalog.fila_de_anime(item_anime(rating="rx"), ctx) is None
    assert ctx.descartes["mal_adulto"] == 1


def test_se_excluye_anime_sin_airstrear():
    ctx = Contexto()
    detalle = item_anime(status="not_yet_aired")
    assert build_catalog.fila_de_anime(detalle, ctx) is None
    assert ctx.descartes["mal_no_estrenado"] == 1


@pytest.mark.parametrize(
    ("estado_tmdb", "esperado"),
    [
        ("Released", "ended"),
        ("Ended", "ended"),
        ("Returning Series", "ongoing"),
        ("Canceled", "other"),
    ],
)
def test_estados_tmdb(estado_tmdb, esperado):
    detalle = detalle_pelicula(status=estado_tmdb)
    assert build_catalog.fila_de_pelicula(detalle, Contexto())["status"] == esperado


@pytest.mark.parametrize(
    ("estado_mal", "esperado"),
    [
        ("finished_airing", "ended"),
        ("currently_airing", "ongoing"),
    ],
)
def test_estados_mal(estado_mal, esperado):
    fila = build_catalog.fila_de_anime(item_anime(status=estado_mal), Contexto())
    assert fila["status"] == esperado


# ------------------------------------------------- duplicados con MAL (animación ja)


def test_animacion_japonesa_se_excluye():
    """Con `ja` y Animación, MAL ya lo cubre."""
    ctx = Contexto()
    detalle = detalle_pelicula(
        original_language="ja", genres=[{"id": 16, "name": "Animación"}]
    )
    assert build_catalog.fila_de_pelicula(detalle, ctx) is None
    assert ctx.descartes["tmdb_animacion_japonesa"] == 1
    assert ctx.ejemplos_ja == ["Matrix"]


def test_animacion_japonesa_de_serie_se_excluye():
    ctx = Contexto()
    detalle = detalle_serie(
        original_language="ja",
        genres=[{"id": 16, "name": "Animación"}, {"id": 1, "name": "Drama"}],
    )
    assert build_catalog.fila_de_serie(detalle, ctx) is None
    assert ctx.descartes["tmdb_animacion_japonesa"] == 1


def test_japones_live_action_se_conserva():
    """24 `ja` reales no son animación (Godzilla, Harakiri, Battle Royale) y MAL no
    los cubre, así que la regla se estrecha por el género y no pierden el sitio."""
    ctx = Contexto()
    detalle = detalle_pelicula(
        id=104,
        title="Harakiri",
        original_language="ja",
        genres=[{"id": 80, "name": "Drama"}],
    )
    fila = build_catalog.fila_de_pelicula(detalle, ctx)
    assert fila is not None
    assert fila["id"] == "tmdb-movie-104"
    assert fila["genres"] == ["Drama"]
    assert ctx.descartes["tmdb_animacion_japonesa"] == 0


def test_no_japones_animado_se_conserva():
    """El filtro necesita las dos cosas: idioma y género."""
    ctx = Contexto()
    detalle = detalle_pelicula(
        original_language="en", genres=[{"id": 16, "name": "Animación"}]
    )
    assert build_catalog.fila_de_pelicula(detalle, ctx) is not None
    detalle = detalle_pelicula(
        original_language="ja", genres=[{"id": 18, "name": "Drama"}]
    )
    assert build_catalog.fila_de_pelicula(detalle, ctx) is not None
    assert ctx.descartes["tmdb_animacion_japonesa"] == 0


# -------------------------------------------------------------------------- keywords


def test_keywords_pelicula_vienen_de_keywords_keywords():
    detalle = detalle_pelicula(
        keywords={"keywords": [{"id": 1, "name": "distopía"}, {"id": 2, "name": "realidad"}]}
    )
    fila = build_catalog.fila_de_pelicula(detalle, Contexto())
    assert fila["keywords"] == ["distopía", "realidad"]


def test_keywords_serie_vienen_de_keywords_results():
    """En tv la lista cuelga de `results`, no de `keywords` como en movie."""
    detalle = detalle_serie(
        keywords={"results": [{"id": 1, "name": "infeccion"}, {"id": 2, "name": "pandemia"}]}
    )
    fila = build_catalog.fila_de_serie(detalle, Contexto())
    assert fila["keywords"] == ["infeccion", "pandemia"]


def test_generos_mal_vienen_como_cadenas_en_anime_json():
    """Regression: anime.json no guarda {"id", "name"} como TMDB. Si se leen con el
    lector de TMDB, los 3467 anime salen con `genres` vacía y no se nota."""
    item = item_anime(genres=["Action", "Shounen"])
    ctx = Contexto()
    fila = build_catalog.fila_de_anime(item, ctx)
    assert fila["genres"] == ["Acción"]
    assert fila["keywords"] == ["Shounen"]
    assert ctx.generos_mal_inesperados == 0


def test_generos_mal_en_formato_inesperado_se_cuentan():
    """Si mañana cambian de forma, que salga un warning en vez de perderlos."""
    item = item_anime(genres=[{"id": 1, "name": "Action"}])
    ctx = Contexto()
    fila = build_catalog.fila_de_anime(item, ctx)
    assert fila["genres"] == []
    assert ctx.generos_mal_inesperados == 1


def test_keywords_anime_son_los_generos_no_mapeados():
    item = item_anime(
        genres=["Action", "Shounen"]
    )
    fila = build_catalog.fila_de_anime(item, Contexto())
    assert fila["genres"] == ["Acción"]
    assert fila["keywords"] == ["Shounen"]


def test_una_fila_sin_keywords_no_revienta(tmp_path):
    tmdb, mal = arbol_completo(tmp_path)
    escribir(tmdb / "movie", "1.json", detalle_pelicula(keywords={"keywords": []}))
    escribir(tmdb / "tv", "10.json", detalle_serie(keywords={}))
    escribir(mal.parent, "anime.json", [item_anime(genres=["Shounen"])])
    salida = tmp_path / "processed" / "catalog.parquet"
    assert build_catalog.run(tmdb, mal, salida) == build_catalog.EXIT_OK
    df = pd.read_parquet(salida)
    sin_keywords = df["keywords"].map(len) == 0
    assert int(sin_keywords.sum()) == 2
    assert df.loc[df["source"] == "mal", "keywords"].iloc[0] == ["Shounen"]


def test_resumen_informa_del_porcentaje_sin_keywords(tmp_path):
    tmdb, mal = arbol_completo(tmp_path)
    escribir(tmdb / "movie", "1.json", detalle_pelicula(keywords={"keywords": []}))
    escribir(tmdb / "tv", "10.json", detalle_serie(keywords={"results": []}))
    escribir(mal.parent, "anime.json", [item_anime(genres=["Action"])])
    ctx = Contexto()
    filas = (
        build_catalog.leer_tmdb(tmdb, "movie", build_catalog.fila_de_pelicula, ctx)
        + build_catalog.leer_tmdb(tmdb, "tv", build_catalog.fila_de_serie, ctx)
        + build_catalog.leer_mal(mal, ctx)
    )
    texto = build_catalog.resumen(build_catalog.construir_dataframe(filas), ctx)
    assert "sin keywords: 3 (100.0%)" in texto


# ------------------------------------------------------------------------- sinopsis MAL


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("Una historia.\n\n[Written by MAL Rewrite]", "Una historia."),
        ("Una historia.\n\n(Source: AniDB)", "Una historia."),
        ("Una historia.\n\n(Source: MangaHelpers, edited)", "Una historia."),
        (
            "Una historia.\n\n[Written by MAL Rewrite]\n\n(Source: Official website)",
            "Una historia.",
        ),
        ("  Una   historia\n  con  saltos ", "Una historia con saltos"),
    ],
)
def test_limpiar_sinopsis_mal(entrada, esperado):
    assert build_catalog.limpiar_sinopsis_mal(entrada) == esperado


def test_overview_viene_limpio_de_mal():
    item = item_anime(synopsis=f"Dos hermanos.\n\n{MAL_PIE}\n\n(Source: AniDB)")
    fila = build_catalog.fila_de_anime(item, Contexto())
    assert fila["overview"] == "Dos hermanos."
    assert MAL_PIE not in fila["embed_text"]


def test_overview_de_tmdb_se_normaliza():
    detalle = detalle_pelicula(overview="Primera línea\n\n   Segunda   línea")
    fila = build_catalog.fila_de_pelicula(detalle, Contexto())
    assert fila["overview"] == "Primera línea Segunda línea"


# ------------------------------------------------------------------------------- ids


def test_ids_usan_el_prefijo_de_cada_fuente(tmp_path):
    df, _ = construir(tmp_path)
    assert list(df["id"]) == ["tmdb-movie-1", "tmdb-tv-10", "mal-anime-5114"]


def test_ids_duplicados_fallan_con_el_detalle():
    filas = [
        {"id": "tmdb-movie-1"},
        {"id": "tmdb-movie-1"},
        {"id": "tmdb-tv-2"},
    ]
    with pytest.raises(ValueError, match="tmdb-movie-1 x2"):
        build_catalog.comprobar_ids(filas)


def test_run_aborta_con_ids_duplicados(tmp_path):
    tmdb, mal = arbol_completo(tmp_path)
    # Dos archivos con el mismo id de película.
    escribir(tmdb / "movie", "1.json", detalle_pelicula())
    escribir(tmdb / "movie", "1-bis.json", detalle_pelicula())
    salida = tmp_path / "catalog.parquet"
    assert build_catalog.run(tmdb, mal, salida) == build_catalog.EXIT_PROBLEMA
    assert not salida.exists()


# ------------------------------------------------------------------- media_type


@pytest.mark.parametrize("tipo_mal", ["tv", "movie", "ova", "ona"])
def test_todo_anime_sale_como_anime(tipo_mal):
    """El contrato con el frontend es movie|tv|anime. Lo que diga MAL se va a
    `mal_type`, que es informativo y no toca el embed_text."""
    fila = build_catalog.fila_de_anime(item_anime(media_type=tipo_mal), Contexto())
    assert fila["media_type"] == "anime"
    assert fila["mal_type"] == tipo_mal
    assert "Tipo: anime." in fila["embed_text"]
    assert f"Tipo: {tipo_mal}" not in fila["embed_text"]


@pytest.mark.parametrize(
    ("constructor", "detalle"),
    [
        (build_catalog.fila_de_pelicula, detalle_pelicula()),
        (build_catalog.fila_de_serie, detalle_serie()),
    ],
)
def test_solo_los_anime_de_mal_llevan_mal_type(constructor, detalle):
    """`mal_type` es None en TMDB: esa columna no significa nada ahí."""
    assert constructor(detalle, Contexto())["mal_type"] is None


def test_ningun_titulo_de_tmdb_es_anime(tmp_path):
    tmdb, _mal = arbol_completo(tmp_path)
    ctx = Contexto()
    filas = build_catalog.leer_tmdb(
        tmdb, "movie", build_catalog.fila_de_pelicula, ctx
    ) + build_catalog.leer_tmdb(tmdb, "tv", build_catalog.fila_de_serie, ctx)
    df = build_catalog.construir_dataframe(filas)
    assert set(df["media_type"]) == {"movie", "tv"}
    assert "anime" not in set(df["media_type"])
    assert df["mal_type"].isna().all()


def test_media_type_anime_llega_al_parquet(tmp_path):
    tmdb, mal = arbol_completo(tmp_path)
    escribir(
        mal.parent,
        "anime.json",
        [item_anime(media_type="ova"), item_anime(id=9, media_type="tv")],
    )
    salida = tmp_path / "processed" / "catalog.parquet"
    assert build_catalog.run(tmdb, mal, salida) == build_catalog.EXIT_OK
    df = pd.read_parquet(salida)
    anime = df[df["source"] == "mal"]
    assert list(anime["media_type"]) == ["anime", "anime"]
    assert sorted(anime["mal_type"]) == ["ova", "tv"]
    assert anime["embed_text"].str.contains("Tipo: anime.").all()
    assert df[df["source"] == "tmdb"]["mal_type"].isna().all()


def test_resumen_cuenta_anime_como_media_type(tmp_path):
    df, ctx = construir(tmp_path)
    assert "'anime': 1" in build_catalog.resumen(df, ctx)


# ------------------------------------------------------------------------ embed_text


def test_embed_text_sigue_la_plantilla():
    detalle = detalle_pelicula(
        title="Matrix",
        genres=[{"id": 28, "name": "Ciencia ficción"}],
        keywords={"keywords": [{"id": 1, "name": "distopía"}]},
        overview="Un hacker descubre la realidad.",
    )
    fila = build_catalog.fila_de_pelicula(detalle, Contexto())
    assert fila["embed_text"] == (
        "Matrix. Tipo: movie. Géneros: Ciencia ficción. Temas: distopía. "
        "Un hacker descubre la realidad."
    )


def test_embed_text_recorta_a_quince_keywords(tmp_path):
    detalle = detalle_pelicula(
        keywords={"keywords": [{"id": i, "name": f"tema{i:02d}"} for i in range(30)]}
    )
    fila = build_catalog.fila_de_pelicula(detalle, Contexto())
    # La columna keywords guarda los 30; solo embed_text recorta.
    assert len(fila["keywords"]) == 30
    assert "tema14" in fila["embed_text"]
    assert "tema15" not in fila["embed_text"]
    assert "tema29" not in fila["embed_text"]


def test_embed_text_no_deja_espacios_raros():
    fila = build_catalog.fila_de_anime(item_anime(), Contexto())
    assert "  " not in fila["embed_text"]
    assert fila["embed_text"] == fila["embed_text"].strip()


# ---------------------------------------------------------------------- parquet/schema


def test_parquet_tiene_las_veintidos_columnas(tmp_path):
    df, _ = construir(tmp_path)
    assert list(df.columns) == list(build_catalog.ORDEN_COLUMNAS)
    assert len(df) == 3


def test_parquet_se_relee_con_los_tipos_del_esquema(tmp_path):
    df, _ = construir(tmp_path)
    salida = tmp_path / "processed" / "catalog.parquet"
    build_catalog.escribir_parquet_atomico(df, salida)

    schema = pa.parquet.read_schema(salida)
    assert schema.names == list(build_catalog.ORDEN_COLUMNAS)
    assert schema.field("genres").type == pa.list_(pa.string())
    assert schema.field("year").type == pa.int64()
    assert schema.field("rating").type == pa.float64()
    assert not schema.field("id").nullable
    assert not schema.field("embed_text").nullable

    leido = pd.read_parquet(salida)
    assert isinstance(leido["year"].dtype, pd.Int64Dtype)
    assert isinstance(leido["rating"].dtype, pd.Float64Dtype)


def test_run_escribe_el_parquet_completo(tmp_path):
    tmdb, mal = arbol_completo(tmp_path)
    salida = tmp_path / "processed" / "catalog.parquet"
    assert build_catalog.run(tmdb, mal, salida) == build_catalog.EXIT_OK
    assert salida.exists()
    assert not salida.with_name("catalog.parquet.tmp").exists()

    df = pd.read_parquet(salida)
    assert len(df) == 3
    assert set(df["source"]) == {"tmdb", "mal"}
    assert set(df["media_type"]) == {"movie", "tv", "anime"}
    pelicula = df[df["id"] == "tmdb-movie-1"].iloc[0]
    serie = df[df["id"] == "tmdb-tv-10"].iloc[0]
    anime = df[df["id"] == "mal-anime-5114"].iloc[0]

    assert pelicula["year"] == 1999
    assert pelicula["overview_lang"] == "es"
    assert pelicula["poster_url"] == "https://image.tmdb.org/t/p/w500/matrix.jpg"
    assert pd.isna(pelicula["episodes"]) and pd.isna(pelicula["seasons"])
    assert pd.isna(pelicula["age_rating"])

    # La serie saca el título de `name`/`original_name` y `first_air_date`.
    assert serie["title"] == "The Last of Us"
    assert serie["original_title"] == "The Last of Us"
    assert serie["year"] == 2023
    assert serie["episodes"] == 9
    assert serie["seasons"] == 1

    assert anime["age_rating"] == "r"
    assert anime["overview_lang"] == "en"
    assert anime["language"] == "ja"
    assert pd.isna(anime["seasons"])
    assert anime["vote_count"] == 900000


def test_run_falla_si_una_fuente_no_entrega_filas(tmp_path):
    tmdb, mal = arbol_completo(tmp_path)
    escribir(mal.parent, "anime.json", [])
    salida = tmp_path / "catalog.parquet"
    assert build_catalog.run(tmdb, mal, salida) == build_catalog.EXIT_PROBLEMA
    assert not salida.exists()


def test_run_falla_si_no_hay_nada_que_leer(tmp_path):
    mal = tmp_path / "mal" / "anime.json"
    escribir(mal.parent, "anime.json", [item_anime()])
    salida = tmp_path / "catalog.parquet"
    assert build_catalog.run(tmp_path / "nada", mal, salida) == build_catalog.EXIT_PROBLEMA


# ------------------------------------------------------------------------- otros campos


def test_anio_desde_fecha():
    assert build_catalog.anio_desde_fecha("1999-03-30") == 1999
    assert build_catalog.anio_desde_fecha("2009-04-05T15:30:00+00:00") == 2009
    assert build_catalog.anio_desde_fecha(None) is None
    assert build_catalog.anio_desde_fecha("") is None


def test_anio_de_mal_usa_year():
    assert build_catalog.anio_desde_fecha(2009) == 2009
    assert build_catalog.anio_desde_fecha(None) is None