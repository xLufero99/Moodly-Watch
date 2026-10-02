"""Tests de scripts/build_index.py y del prefijo que comparte con search_demo.py.

Nada de aquí baja el modelo de verdad. Para las pruebas del índice se usa
`EmbedderFalso`, que devuelve vectores deterministas, y ChromaDB de verdad sobre un
directorio temporal: así se comprueba que lo que produce el script es algo que ChromaDB
acepta de verdad, y no solo que el dict tiene las claves correctas. Lo que no se prueba
es la calidad del modelo, que para eso no sirve un vector inventado.

Los vectores se sacan de `hashlib` y no del `hash()` de Python a propósito: `hash()` de
cadenas lleva una sal por proceso, así que dos ejecuciones distintas darían vectores
distintos y el test no sería reproducible ni en local ni en CI.

Lo que cubren:
  * que `None` y `NaN` no llegan a los metadatos (ChromaDB los rechaza, ver test_note)
  * que `genres` sale como texto con `|`, y que una lista vacía no se cuela
  * que los números de numpy se bajan a tipos nativos antes de dárselos a ChromaDB
  * que los percentiles van dentro de cada `source` y salen en 0-1
  * que los empates en `vote_count` dan el mismo `popularity_pct`
  * que el prefijo de documento y el de consulta llegan al modelo, y no se duplican
  * que reconstruir no deja duplicados
  * que `index_info.json` lleva los campos que prometía

Y un par de hechos comprobados contra la versión instalada, dejados en los tests donde
importan:

  * `test_chroma_rechaza_none_de_verdad`: el validador de Python de ChromaDB admite
    `None`, pero el `add` falla con `Cannot convert Python object to MetadataValue`,
    porque la comprobación que manda es la de la capa de Rust. Por eso los nulos se
    omiten en vez de mandarse.
  * `test_los_numeros_numpy_se_bajan_a_nativos`: `isinstance(np.int64(5), int)` es
    `False` en Python, así que los enteros de numpy se rechazan si se pasan sin
    convertir.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _fakes import DIMENSION, EmbedderFalso, ModeloFalso, vector_para

from app.services.embedder import (
    PREFIJO_CONSULTA,
    PREFIJO_DOCUMENTO,
    Embedder,
    consulta_para_buscar,
    documento_para_indexar,
)
from scripts.build_index import (
    EXIT_OK,
    anadir_percentiles,
    construir_coleccion,
    limitar,
    main,
    metadatos_de_fila,
    run,
    volcar_en_coleccion,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def fila(
    id_: str,
    *,
    source: str = "TMDB",
    media_type: str = "movie",
    rating: float = 7.0,
    vote_count: int = 1000,
    genres=("Comedia", "Crimen"),
    poster_url: str | None = "https://example.org/p.jpg",
    year: int = 2020,
    mal_type: str | None = None,
) -> dict:
    return {
        "id": id_,
        "source": source,
        "media_type": media_type,
        "mal_type": mal_type,
        "title": f"Título {id_}",
        "year": year,
        "rating": rating,
        "vote_count": vote_count,
        "popularity": 10.0,
        "runtime_total": 100,
        "runtime_episode": None,
        "episodes": None,
        "seasons": None,
        "status": "Ended",
        "language": "es",
        "poster_url": poster_url,
        "genres": genres,
        "embed_text": f"Texto de {id_}.",
    }


def catalogo(n: int = 6) -> pd.DataFrame:
    df = pd.DataFrame([fila(f"id-{i:03d}") for i in range(n)])
    return df


@pytest.fixture
def parquet_pequeno(tmp_path: Path) -> Path:
    ruta = tmp_path / "catalog.parquet"
    catalogo().to_parquet(ruta, index=False)
    return ruta


# --------------------------------------------------------------------------- #
# Metadatos
# --------------------------------------------------------------------------- #


def test_none_no_aparece_en_los_metadatos():
    meta = metadatos_de_fila(fila("a", poster_url=None, mal_type=None))
    assert "poster_url" not in meta
    assert "mal_type" not in meta


def test_nan_no_aparece_en_los_metadatos():
    registro = fila("a")
    registro["rating"] = float("nan")
    meta = metadatos_de_fila(registro)
    assert "rating" not in meta


def test_pd_na_no_aparece_en_los_metadatos():
    registro = fila("a")
    registro["poster_url"] = pd.NA
    assert "poster_url" not in metadatos_de_fila(registro)


def test_los_nulos_no_llegan_a_chroma_de_verdad(tmp_path):
    """El `add` real acepta lo que produce el script: ni un `None` se cuela."""
    import chromadb

    registro = fila("ok")
    registro["poster_url"] = None
    registro["mal_type"] = None
    registro["rating"] = float("nan")
    registro["year"] = np.int64(2020)  # numpy, que es justo lo que se rechaza

    coleccion = chromadb.PersistentClient(path=str(tmp_path / "chroma")).create_collection(
        "acepta"
    )
    coleccion.add(
        ids=["ok"],
        documents=["Texto."],
        embeddings=[vector_para("Texto.")],
        metadatas=[metadatos_de_fila(registro)],
    )
    guardado = coleccion.get("ok")["metadatas"][0]
    assert guardado["year"] == 2020
    assert isinstance(guardado["year"], int)
    assert "poster_url" not in guardado


def test_chroma_rechaza_none_de_verdad(tmp_path):
    """Hecho comprobado contra chromadb 1.5.9, no leído: por eso se omiten los nulos."""
    import chromadb

    coleccion = chromadb.PersistentClient(path=str(tmp_path / "chroma")).create_collection(
        "rechazo"
    )
    # chromadb sube esto como TypeError: "argument 'metadatas': Cannot convert Python
    # object to MetadataValue".
    with pytest.raises(TypeError, match="MetadataValue"):
        coleccion.add(
            ids=["x"],
            documents=["Texto."],
            embeddings=[vector_para("Texto.")],
            metadatas=[{"year": None}],
        )


def test_generos_se_unen_con_barra():
    meta = metadatos_de_fila(fila("a", genres=("Terror", "Suspense")))
    assert meta["genres"] == "Terror|Suspense"


def test_generos_vienen_de_numpy_y_aun_así_se_unen():
    meta = metadatos_de_fila(fila("a", genres=np.array(["Drama", "Historia"])))
    assert meta["genres"] == "Drama|Historia"


def test_generos_vacios_se_omiten():
    """Una lista vacía la rechaza ChromaDB; no puede guardarse ni como texto vacío."""
    meta = metadatos_de_fila(fila("a", genres=np.array([], dtype=object)))
    assert "genres" not in meta


def test_los_numeros_numpy_se_bajan_a_nativos():
    registro = fila("a")
    registro["year"] = np.int64(2020)
    registro["vote_count"] = np.int32(500)
    registro["rating"] = np.float64(7.5)
    meta = metadatos_de_fila(registro)
    assert meta["year"] == 2020 and type(meta["year"]) is int
    assert meta["vote_count"] == 500 and type(meta["vote_count"]) is int
    assert meta["rating"] == 7.5 and type(meta["rating"]) is float


def test_todos_los_campos_de_metadatos_estan():
    # Se pasa por anadir_percentiles porque rating_pct y popularity_pct no vienen del
    # parquet: los calcula el script. Esta es la fila tal como llega a ChromaDB.
    df = anadir_percentiles(pd.DataFrame([fila("a"), fila("b"), fila("c")]))
    meta = metadatos_de_fila(df.iloc[0].to_dict())
    for campo in (
        "media_type", "source", "title", "year", "rating", "vote_count",
        "popularity", "runtime_total", "status", "language", "poster_url", "genres",
        "rating_pct", "popularity_pct",
    ):
        assert campo in meta, campo


def test_una_pelicula_no_trae_los_campos_de_serie():
    """LaRuntime va partida en dos columnas y una fila nunca lleva las dos."""
    meta = metadatos_de_fila(fila("a", media_type="movie"))
    assert meta["runtime_total"] == 100
    assert "runtime_episode" not in meta
    assert "episodes" not in meta


def test_una_serie_trae_el_episodio_y_no_el_total():
    registro = fila("s", media_type="tv")
    registro["runtime_total"] = None
    registro["runtime_episode"] = 45
    registro["episodes"] = 8
    registro["seasons"] = 2
    meta = metadatos_de_fila(registro)
    assert meta["runtime_episode"] == 45
    assert meta["episodes"] == 8
    assert "runtime_total" not in meta


# --------------------------------------------------------------------------- #
# Percentiles
# --------------------------------------------------------------------------- #


def df_de_percentiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["TMDB"] * 3 + ["MAL"] * 3,
            "rating": [9.0, 8.0, 7.0, 6.0, 5.0, 9.5],
            "vote_count": [100, 200, 300, 400, 500, 600],
        }
    )


def test_los_percentiles_salen_en_cero_y_uno():
    df = anadir_percentiles(df_de_percentiles())
    for columna in ("rating_pct", "popularity_pct"):
        assert df[columna].between(0.0, 1.0).all()
        assert df[columna].min() > 0.0
        assert df[columna].max() == 1.0


def test_los_percentiles_se_calculan_dentro_de_cada_source():
    df = anadir_percentiles(df_de_percentiles())
    # El 6.0 de MAL es el segundo más alto de MAL, pero solo el cuarto de seis en el
    # catálogo entero. Si fuera global saldría 0.5; dentro de la fuente sale 2/3.
    fila_mal = df[(df["source"] == "MAL") & (df["rating"] == 6.0)].iloc[0]
    assert fila_mal["rating_pct"] == pytest.approx(2 / 3)


def test_los_empates_en_vote_count_dan_el_mismo_popularity_pct():
    df = pd.DataFrame(
        {
            "source": ["TMDB"] * 4,
            "rating": [7.0, 8.0, 9.0, 9.5],
            "vote_count": [100, 500, 500, 900],
        }
    )
    df = anadir_percentiles(df)
    empatados = df[df["vote_count"] == 500]
    assert len(empatados) == 2
    assert empatados["popularity_pct"].iloc[0] == empatados["popularity_pct"].iloc[1]
    # Rango medio de los dos empatados: (2 + 3) / 2 = 2.5 de 4 filas -> 0.625
    assert empatados["popularity_pct"].iloc[0] == pytest.approx(0.625)


def test_los_empates_no_se_reparten_como_con_first():
    """`first` los separaría en 0.5 y 0.75, que es justo lo que no queremos."""
    df = pd.DataFrame(
        {
            "source": ["TMDB"] * 4,
            "rating": [7.0, 8.0, 9.0, 9.5],
            "vote_count": [100, 500, 500, 900],
        }
    )
    con_first = df["vote_count"].rank(pct=True, method="first")
    con_average = anadir_percentiles(df)["popularity_pct"]
    assert con_first.nunique() > con_average.nunique()
    assert con_average.nunique() == 3


# --------------------------------------------------------------------------- #
# Prefijos
# --------------------------------------------------------------------------- #


def test_el_prefijo_de_documento_llega_al_modelo():
    modelo = ModeloFalso()
    Embedder(modelo, "falso/stub").incrustar_documentos(["Una sinopsis."])
    assert modelo.recibidos[0] == [PREFIJO_DOCUMENTO + "Una sinopsis."]


def test_el_prefijo_de_consulta_llega_al_modelo():
    modelo = ModeloFalso()
    Embedder(modelo, "falso/stub").incrustar_consultas(["Qué busco"])
    assert modelo.recibidos[0] == [PREFIJO_CONSULTA + "Qué busco"]


def test_documento_y_consulta_llevan_prefijos_distintos():
    """No es descuido: E5 es asimétrico y mezclarlos degrada sin dar error."""
    assert PREFIJO_DOCUMENTO != PREFIJO_CONSULTA
    assert documento_para_indexar("x").startswith(PREFIJO_DOCUMENTO)
    assert consulta_para_buscar("x").startswith(PREFIJO_CONSULTA)


def test_el_prefijo_no_se_duplica():
    assert documento_para_indexar(documento_para_indexar("x")) == PREFIJO_DOCUMENTO + "x"
    assert consulta_para_buscar(consulta_para_buscar("x")) == PREFIJO_CONSULTA + "x"


def test_el_documento_guardado_no_lleva_el_prefijo_dentro(tmp_path):
    """El `passage: ` va al modelo, no al `document` de ChromaDB."""
    df = anadir_percentiles(catalogo(3))
    filas = []
    for registro in df.to_dict("records"):
        fila_ = dict(registro)
        fila_["_embedding"] = vector_para(fila_["embed_text"])
        filas.append(fila_)
    coleccion = construir_coleccion(tmp_path / "index")
    volcar_en_coleccion(coleccion, filas)
    guardados = coleccion.get()["documents"]
    assert guardados
    assert all(not d.startswith(PREFIJO_DOCUMENTO) for d in guardados)
    assert all(d.startswith("Texto de ") for d in guardados)


# --------------------------------------------------------------------------- #
# Índice
# --------------------------------------------------------------------------- #


def test_run_devuelve_ok_y_escribe_los_dos_ficheros(parquet_pequeno, tmp_path):
    dir_indice = tmp_path / "index"
    codigo = run(
        parquet=parquet_pequeno,
        dir_indice=dir_indice,
        embedder=EmbedderFalso(),
    )
    assert codigo == EXIT_OK
    assert (dir_indice / "index_info.json").exists()
    assert (dir_indice / "chroma.sqlite3").exists()


def test_reconstruir_no_deja_duplicados(parquet_pequeno, tmp_path):
    dir_indice = tmp_path / "index"
    for _ in range(2):
        codigo = run(
            parquet=parquet_pequeno,
            dir_indice=dir_indice,
            embedder=EmbedderFalso(),
        )
        assert codigo == EXIT_OK

    import chromadb

    coleccion = chromadb.PersistentClient(path=str(dir_indice)).get_collection("catalog")
    assert coleccion.count() == 6
    assert len(coleccion.get()["ids"]) == 6


def test_los_documentos_del_indice_llevan_los_metadatos_esperados(parquet_pequeno, tmp_path):
    import chromadb

    dir_indice = tmp_path / "index"
    run(parquet=parquet_pequeno, dir_indice=dir_indice, embedder=EmbedderFalso())
    coleccion = chromadb.PersistentClient(path=str(dir_indice)).get_collection("catalog")
    meta = coleccion.get("id-000")["metadatas"][0]

    assert meta["title"] == "Título id-000"
    assert meta["media_type"] == "movie"
    assert meta["genres"] == "Comedia|Crimen"
    assert "mal_type" not in meta  # era None en la fila
    assert 0.0 < meta["rating_pct"] <= 1.0


def test_index_info_lleva_los_campos_prometidos(parquet_pequeno, tmp_path):
    dir_indice = tmp_path / "index"
    run(parquet=parquet_pequeno, dir_indice=dir_indice, embedder=EmbedderFalso())
    info = json.loads((dir_indice / "index_info.json").read_text(encoding="utf-8"))

    assert info["modelo"] == EmbedderFalso().nombre
    assert info["dimension"] == DIMENSION
    assert info["prefijo_documento"] == PREFIJO_DOCUMENTO
    assert info["prefijo_consulta"] == PREFIJO_CONSULTA
    assert info["documentos"] == 6
    assert info["metrica"] == "cosine"
    assert len(info["parquet_sha256"]) == 64
    assert info["generado"].endswith("+00:00")
    assert info["docs_por_segundo"] > 0


def test_index_info_cambia_si_cambia_el_parquet(tmp_path):
    ruta = tmp_path / "catalog.parquet"
    catalogo(6).to_parquet(ruta, index=False)
    run(parquet=ruta, dir_indice=tmp_path / "a", embedder=EmbedderFalso())
    primera = json.loads((tmp_path / "a" / "index_info.json").read_text(encoding="utf-8"))

    catalogo(7).to_parquet(ruta, index=False)
    run(parquet=ruta, dir_indice=tmp_path / "b", embedder=EmbedderFalso())
    segunda = json.loads((tmp_path / "b" / "index_info.json").read_text(encoding="utf-8"))

    assert primera["parquet_sha256"] != segunda["parquet_sha256"]


def test_sin_parquet_devuelve_problema(tmp_path):
    codigo = run(
        parquet=tmp_path / "no-existe.parquet",
        dir_indice=tmp_path / "index",
        embedder=EmbedderFalso(),
    )
    assert codigo != EXIT_OK


# --------------------------------------------------------------------------- #
# --limit
# --------------------------------------------------------------------------- #


def test_limitar_reparte_entre_los_tres_media_types():
    df = pd.concat(
        [
            pd.DataFrame({"id": [f"m{i}" for i in range(50)], "media_type": ["movie"] * 50}),
            pd.DataFrame({"id": [f"a{i}" for i in range(35)], "media_type": ["anime"] * 35}),
            pd.DataFrame({"id": [f"t{i}" for i in range(15)], "media_type": ["tv"] * 15}),
        ],
        ignore_index=True,
    )
    recorte = limitar(df, 60)
    reparto = recorte["media_type"].value_counts().to_dict()
    assert len(recorte) == 60
    assert set(reparto) == {"movie", "anime", "tv"}
    # Proporcional, no equitativo: las películas pesan más que las series.
    assert reparto["movie"] > reparto["tv"]
    assert reparto["movie"] == pytest.approx(60 * 50 / 100, abs=1)


def test_limitar_devuelve_exactamente_n():
    df = pd.DataFrame(
        {
            "id": [f"x{i}" for i in range(100)],
            "media_type": ["movie"] * 50 + ["anime"] * 35 + ["tv"] * 15,
        }
    )
    for limite in (1, 7, 13, 99, 100):
        recorte = limitar(df, limite)
        assert len(recorte) == limite, limite
        assert recorte["id"].nunique() == limite, limite
        # Con 3 filas o más tiene que haber hueco para los tres tipos. Con menos es
        # imposible, no un fallo: el reparto proporcional no puede inventar filas.
        if limite >= 3:
            assert recorte["media_type"].nunique() == 3, limite


def test_limitar_es_determinista():
    df = pd.DataFrame(
        {
            "id": [f"x{i}" for i in range(10)],
            "media_type": ["movie"] * 5 + ["anime"] * 3 + ["tv"] * 2,
        }
    )
    assert limitar(df, 6)["id"].tolist() == limitar(df, 6)["id"].tolist()


def test_limit_mayor_que_el_catalogo_no_recorta():
    df = catalogo(6)
    assert len(limitar(df, 100)) == 6
    assert len(limitar(df, None)) == 6


def test_main_rechaza_limit_no_positivo():
    assert main(["--limit", "0"]) != EXIT_OK
    assert main(["--limit", "-5"]) != EXIT_OK


def test_main_rechaza_top_no_positivo():
    from scripts.search_demo import main as main_busqueda

    assert main_busqueda(["algo", "--top", "0"]) != EXIT_OK