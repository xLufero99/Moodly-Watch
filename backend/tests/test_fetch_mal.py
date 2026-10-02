"""Tests de scripts/fetch_mal.py sin tocar la red: todo va por httpx.MockTransport."""

import json
import logging
from collections import Counter

import httpx
import pytest

from scripts import fetch_mal
from scripts.fetch_mal import MalAuthError, Status, parse_args

CLAVE = "clave-de-prueba-123"


@pytest.fixture(autouse=True)
def sin_pausas(monkeypatch):
    """Sin esperas: los reintentos y la pausa entre peticiones no interesan aquí."""
    monkeypatch.setattr(fetch_mal, "REQUEST_PAUSE_S", 0.0)
    monkeypatch.setattr(fetch_mal, "_sleep", lambda _seconds: None)


def pagina(items: list[dict], next_url: str | None = None) -> dict:
    """Respuesta de /anime/ranking. Sin `next` en `paging` es la última página."""
    paging = {"previous": "https://api.myanimelist.net/v2/anime/ranking?offset=0"}
    if next_url is not None:
        paging["next"] = next_url
    return {"paging": paging, "data": items}


def node(anime_id: int, **overrides) -> dict:
    """El `node` de un ítem del ranking, con los campos que nos interesan."""
    base = {
        "id": anime_id,
        "title": f"Anime {anime_id}",
        "main_picture": {
            "medium": "https://cdn.example/1.jpg",
            "large": "https://cdn.example/1l.jpg",
        },
        "alternative_titles": {
            "synonyms": ["Otro nombre"],
            "en": f"Anime {anime_id} EN",
            "ja": f"アニメ{anime_id}",
        },
        "start_date": "2023-09-29",
        "synopsis": "Una historia.",
        "mean": 9.25,
        "rank": anime_id,
        "popularity": 97,
        "num_list_users": 150,
        "num_scoring_users": 5000,
        "genres": [{"id": 2, "name": "Adventure"}, {"id": 8, "name": "Drama"}],
        "media_type": "tv",
        "status": "finished_airing",
        "num_episodes": 28,
        "start_season": {"year": 2023, "season": "fall"},
        "average_episode_duration": 1470,
        "rating": "pg_13",
        "studios": [{"id": 11, "name": "Madhouse"}],
    }
    base.update(overrides)
    return base


def item(anime_id: int, rank: int | None = None, **overrides) -> dict:
    """Un ítem completo: `node` + `ranking`, que es como lo devuelve MAL."""
    return {
        "node": node(anime_id, **overrides),
        "ranking": {"rank": anime_id if rank is None else rank},
    }


class Contador:
    """Handler con contador, para afirmar cuántas peticiones hicieron falta."""

    def __init__(self, respuestas: list[httpx.Response]) -> None:
        self.respuestas = respuestas
        self.calls = 0
        self.headers: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.headers.append(request.headers.get("X-MAL-CLIENT-ID"))
        return self.respuestas[min(self.calls, len(self.respuestas)) - 1]


class Paginador:
    """Handler que responde la página que le pide cada offset."""

    def __init__(self, pages: dict[int, dict], status: int = 200) -> None:
        self.pages = pages
        self.status = status
        self.calls: list[dict[str, str]] = []
        self.headers: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/anime/ranking"
        params = dict(request.url.params)
        self.calls.append(params)
        self.headers.append(request.headers.get("X-MAL-CLIENT-ID"))
        offset = int(params["offset"])
        if offset not in self.pages:
            return httpx.Response(self.status, json={"error": "not_found"})
        return httpx.Response(200, json=self.pages[offset])


def make_client(handler, client_id: str = CLAVE) -> httpx.Client:
    return fetch_mal.build_client(client_id, transport=httpx.MockTransport(handler))


def make_args(**overrides):
    argv = []
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return parse_args(argv)


def guardar_pagina(directory, page: int, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"page_{page}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


# --- (a) reanudable: una página ya guardada no se vuelve a pedir ------------


def test_pagina_ya_guardada_no_se_vuelve_a_pedir(tmp_path):
    directory = tmp_path / "pages"
    guardar_pagina(directory, 1, pagina([item(1)]))
    handler = Contador([httpx.Response(500, json={"error": "down"})])

    with make_client(handler) as client:
        estado, payload = fetch_mal.fetch_page(client, 1, directory)

    assert estado is Status.SKIPPED
    assert handler.calls == 0
    assert payload["data"][0]["node"]["id"] == 1


def test_reanudar_no_repide_ninguna_pagina(tmp_path):
    directory = tmp_path / "pages"
    siguiente = "https://api.myanimelist.net/v2/anime/ranking?offset=50&ranking_type=all&limit=50"
    guardar_pagina(directory, 1, pagina([item(1)], next_url=siguiente))
    guardar_pagina(directory, 2, pagina([item(2)]))
    handler = Paginador({})
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_mal.fetch_pages(client, directory, 100, counters)

    assert handler.calls == []
    assert counters[Status.SKIPPED] == 2
    assert counters[Status.SAVED] == 0


def test_pagina_corrupta_se_vuelve_a_pedir(tmp_path, caplog):
    directory = tmp_path / "pages"
    directory.mkdir(parents=True)
    (directory / "page_1.json").write_text("{roto", encoding="utf-8")
    handler = Paginador({0: pagina([item(1)])})

    with make_client(handler) as client, caplog.at_level(logging.WARNING):
        estado, payload = fetch_mal.fetch_page(client, 1, directory)

    assert estado is Status.SAVED
    assert handler.calls[0]["offset"] == "0"
    assert "corrupta" in caplog.text
    assert payload["data"][0]["node"]["id"] == 1


# --- (b) 429 con Retry-After -------------------------------------------------


def test_429_con_retry_after_se_reintenta_y_luego_funciona(
    tmp_path, monkeypatch, caplog
):
    esperas: list[float] = []
    monkeypatch.setattr(fetch_mal, "_sleep", esperas.append)
    handler = Contador(
        [
            httpx.Response(429, headers={"retry-after": "3"}, json={"error": "rate"}),
            httpx.Response(200, json=pagina([item(1)])),
        ]
    )

    with make_client(handler) as client, caplog.at_level(logging.WARNING):
        estado, payload = fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert estado is Status.SAVED
    assert handler.calls == 2
    assert payload["data"][0]["node"]["id"] == 1
    assert "429" in caplog.text


def test_429_sin_retry_after_usa_backoff_exponencial(tmp_path, monkeypatch):
    esperas: list[float] = []
    monkeypatch.setattr(fetch_mal, "_sleep", esperas.append)
    handler = Contador(
        [
            httpx.Response(429, json={"error": "rate"}),
            httpx.Response(429, json={"error": "rate"}),
            httpx.Response(200, json=pagina([item(1)])),
        ]
    )

    with make_client(handler) as client:
        estado, _ = fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert estado is Status.SAVED
    assert handler.calls == 3
    # 1 s y luego 2 s, el backoff por defecto de tenacity.
    assert esperas == [1.0, 2.0]


# --- (c) el aplanado deduplica, convierte segundos y filtra ----------------


def test_aplana_deduplica_por_id():
    pages = [
        pagina(
            [
                item(1),
                item(2, media_type="special"),
                item(3, num_scoring_users=10),
                item(4, synopsis="   "),
                {"ranking": {"rank": 5}},
            ]
        ),
        pagina([item(1, title="Repetido")]),
    ]

    lista, stats = fetch_mal.flatten(pages, fetch_mal.DEFAULT_MIN_SCORING_USERS)

    assert [entrada["id"] for entrada in lista] == [1]
    assert lista[0]["title"] == "Anime 1"
    assert stats["duplicados"] == 1
    assert stats["fuera_de_tipo"] == 1
    assert stats["pocos_votos"] == 1
    assert stats["sin_sinopsis"] == 1
    assert stats["sin_node"] == 1
    assert stats["vistos"] == 6


def test_aplana_extrae_los_campos_pedidos():
    lista, _ = fetch_mal.flatten([pagina([item(1)])], 0)

    assert lista == [
        {
            "id": 1,
            "title": "Anime 1",
            "title_english": "Anime 1 EN",
            "title_japanese": "アニメ1",
            "media_type": "tv",
            "num_episodes": 28,
            "status": "finished_airing",
            "duration_minutes": 24,
            "rating": "pg_13",
            "mean": 9.25,
            "rank": 1,
            "popularity": 97,
            "num_scoring_users": 5000,
            "synopsis": "Una historia.",
            "year": 2023,
            "season": "fall",
            "start_date": "2023-09-29",
            "genres": ["Adventure", "Drama"],
            "studios": ["Madhouse"],
            "poster_url": "https://cdn.example/1l.jpg",
        }
    ]


@pytest.mark.parametrize(
    ("segundos", "minutos"),
    [(1470, 24), (1440, 24), (90, 1), (0, 0), (None, 0), (3600, 60)],
)
def test_segundos_a_minutos_truncando(segundos, minutos):
    lista, _ = fetch_mal.flatten(
        [pagina([item(1, average_episode_duration=segundos)])], 0
    )

    assert lista[0]["duration_minutes"] == minutos


def test_poster_cae_a_medium_si_no_hay_large():
    lista, _ = fetch_mal.flatten(
        [pagina([item(1, main_picture={"medium": "https://cdn.example/1.jpg"})])], 0
    )

    assert lista[0]["poster_url"] == "https://cdn.example/1.jpg"


def test_poster_null_si_no_hay_imagen():
    lista, _ = fetch_mal.flatten([pagina([item(1, main_picture={})])], 0)

    assert lista[0]["poster_url"] is None


def test_ano_cae_a_start_date_si_no_hay_start_season():
    lista, _ = fetch_mal.flatten(
        [pagina([item(1, start_season=None, start_date="1997-10-20")])], 0
    )

    assert lista[0]["year"] == 1997
    assert lista[0]["season"] is None


def test_ano_null_si_no_hay_start_season_ni_start_date():
    lista, _ = fetch_mal.flatten(
        [pagina([item(1, start_season=None, start_date=None)])], 0
    )

    assert lista[0]["year"] is None


def test_rank_sale_de_ranking_si_el_node_no_lo_trae():
    lista, _ = fetch_mal.flatten([pagina([item(1, rank=None)])], 0)

    assert lista[0]["rank"] == 1


def test_min_scoring_users_sube_el_umbral():
    pages = [pagina([item(1, num_scoring_users=999), item(2, num_scoring_users=1000)])]

    lista, _ = fetch_mal.flatten(pages, 1000)

    assert [entrada["id"] for entrada in lista] == [2]


@pytest.mark.parametrize(
    ("media_type", "permitido"),
    [
        ("tv", True),
        ("TV", True),
        ("movie", True),
        ("ova", True),
        ("ona", True),
        ("special", False),
        ("music", False),
        ("unknown", False),
        (None, False),
    ],
)
def test_filtro_de_media_type_ignora_mayusculas(media_type, permitido):
    assert fetch_mal.es_tipo_permitido(media_type) is permitido


def test_flatten_ignora_entradas_basura():
    lista, stats = fetch_mal.flatten([{"data": ["texto", None]}], 0)

    assert lista == []
    assert stats["vistos"] == 0


# --- (d) la paginación se detiene sin paging.next --------------------------


def test_paginacion_se_detiene_sin_paging_next(tmp_path):
    directory = tmp_path / "pages"
    handler = Paginador({0: pagina([item(1)])})
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_mal.fetch_pages(client, directory, 100, counters)

    assert len(handler.calls) == 1
    assert counters[Status.SAVED] == 1
    assert not (directory / "page_2.json").exists()


def test_paginacion_sigue_con_paging_next(tmp_path):
    directory = tmp_path / "pages"
    siguiente = "https://api.myanimelist.net/v2/anime/ranking?offset=50&ranking_type=all&limit=50"
    handler = Paginador(
        {
            0: pagina([item(1)], next_url=siguiente),
            fetch_mal.page_offset(2): pagina([item(2)]),
        }
    )
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_mal.fetch_pages(client, directory, 100, counters)

    assert [int(call["offset"]) for call in handler.calls] == [0, 50]
    assert counters[Status.SAVED] == 2


def test_paginacion_se_detiene_al_llegar_al_tope_de_items(tmp_path):
    directory = tmp_path / "pages"
    siguiente = "https://api.myanimelist.net/v2/anime/ranking?offset=50"
    handler = Paginador({0: pagina([item(1)], next_url=siguiente)})
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_mal.fetch_pages(client, directory, 1, counters)

    assert len(handler.calls) == 1


def test_paginacion_se_detiene_con_pagina_vacia(tmp_path):
    directory = tmp_path / "pages"
    handler = Paginador({0: pagina([item(1)], next_url="https://x/next")})
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_mal.fetch_pages(client, directory, 500, counters)

    # La segunda página vino vacía: no se escribe y se para.
    assert [int(call["offset"]) for call in handler.calls] == [0, 50]
    assert counters[Status.SAVED] == 1
    assert counters[Status.EMPTY] == 1
    assert not (directory / "page_2.json").exists()


# --- (e) 401 (y el 400 de client id) aborta sin reintentar ------------------


def test_401_aborta_de_inmediato(tmp_path, caplog):
    handler = Contador([httpx.Response(401, json={"error": "unauthorized"})])

    with (
        make_client(handler) as client,
        caplog.at_level(logging.ERROR),
        pytest.raises(MalAuthError) as error,
    ):
        fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert handler.calls == 1
    assert "MAL_CLIENT_ID" in str(error.value)
    assert CLAVE not in str(error.value)
    assert CLAVE not in caplog.text
    assert not (tmp_path / "pages" / "page_1.json").exists()


def test_400_invalid_client_id_aborta_sin_reintentar(tmp_path, caplog):
    """MAL no da 401 con una client id mala, da 400 'Invalid client id'."""
    handler = Contador(
        [
            httpx.Response(
                400, json={"error": "bad_request", "message": "Invalid client id"}
            )
        ]
    )

    with (
        make_client(handler) as client,
        caplog.at_level(logging.ERROR),
        pytest.raises(MalAuthError),
    ):
        fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert handler.calls == 1
    assert CLAVE not in caplog.text


def test_400_que_no_es_de_credencial_no_aborta(tmp_path):
    """Un 400 de otra cosa (limit inválido, field desconocido) no es de credencial."""
    handler = Contador([httpx.Response(400, json={"error": "limit"})])

    with make_client(handler) as client:
        estado, _ = fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert estado is Status.FAILED
    assert handler.calls == 1


def test_run_con_credencial_rechazada_sale_2(tmp_path, caplog):
    handler = Contador([httpx.Response(401, json={"error": "unauthorized"})])

    with caplog.at_level(logging.ERROR):
        codigo = fetch_mal.run(
            make_args(limit=50),
            raw_dir=tmp_path,
            client_id=CLAVE,
            transport=httpx.MockTransport(handler),
        )

    assert codigo == fetch_mal.EXIT_CONFIG
    assert CLAVE not in caplog.text
    assert not (tmp_path / "anime.json").exists()


def test_run_sin_credencial_sale_2(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        codigo = fetch_mal.run(make_args(), raw_dir=tmp_path, client_id="")

    assert codigo == fetch_mal.EXIT_CONFIG
    assert "MAL_CLIENT_ID" in caplog.text
    assert not tmp_path.joinpath("pages").exists()


def test_5xx_persistentes_cuentan_como_failed(tmp_path):
    handler = Contador([httpx.Response(503, json={"error": "down"})])

    with make_client(handler) as client:
        estado, _ = fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert estado is Status.FAILED
    assert handler.calls == fetch_mal.MAX_ATTEMPTS
    assert not (tmp_path / "pages" / "page_1.json").exists()


def test_404_se_salta_sin_escribir_archivo(tmp_path, caplog):
    directory = tmp_path / "pages"
    handler = Paginador({}, status=404)
    counters: Counter = Counter()

    with make_client(handler) as client, caplog.at_level(logging.WARNING):
        fetch_mal.fetch_pages(client, directory, 100, counters)

    assert counters[Status.MISSING] == 1
    assert not (directory / "page_1.json").exists()
    assert "404" in caplog.text


# --- (f) la client id no se filtra ni a los logs ni a los archivos ----------


def test_logs_nunca_exponen_la_client_id(tmp_path, caplog, monkeypatch):
    """httpx loggea la URL completa en INFO, y los headers solo en TRACE."""
    cliente_httpx = logging.getLogger("httpx")
    monkeypatch.setattr(cliente_httpx, "level", logging.DEBUG)

    fetch_mal.silence_http_client_logs()
    assert cliente_httpx.level == logging.WARNING

    handler = Paginador({0: pagina([item(1)])})
    with make_client(handler) as client, caplog.at_level(logging.DEBUG):
        fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert CLAVE not in caplog.text


def test_la_client_id_no_aparece_en_los_json_guardados(tmp_path):
    handler = Paginador({0: pagina([item(1)])})

    with make_client(handler) as client:
        fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert CLAVE not in json.dumps(handler.calls[0])
    assert CLAVE not in (tmp_path / "pages" / "page_1.json").read_text(encoding="utf-8")
    assert handler.headers == [CLAVE]


def test_la_client_id_no_aparece_en_el_error_de_status(tmp_path, caplog):
    handler = Contador([httpx.Response(400, json={"error": "limit"})])

    with make_client(handler) as client, caplog.at_level(logging.WARNING):
        fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    assert "MAL respondió 400 en /anime/ranking" in caplog.text
    assert CLAVE not in caplog.text


# --- run completo, aplanado y códigos de salida ----------------------------


def test_run_descarga_y_aplana(tmp_path):
    handler = Paginador(
        {
            0: pagina(
                [item(1), item(2, media_type="music")],
                next_url="https://api.myanimelist.net/v2/anime/ranking?offset=50",
            ),
            50: pagina([item(3, num_scoring_users=5), item(4)]),
        }
    )

    codigo = fetch_mal.run(
        make_args(limit=50),
        raw_dir=tmp_path,
        client_id=CLAVE,
        transport=httpx.MockTransport(handler),
    )

    assert codigo == fetch_mal.EXIT_OK
    assert [int(call["offset"]) for call in handler.calls] == [0, 50]
    anime_json = json.loads((tmp_path / "anime.json").read_text(encoding="utf-8"))
    assert [entrada["id"] for entrada in anime_json] == [1, 4]


def test_run_reanudable_genera_el_mismo_anime(tmp_path):
    args = make_args(limit=50)
    handler = Paginador(
        {
            0: pagina(
                [item(1)],
                next_url="https://api.myanimelist.net/v2/anime/ranking?offset=50",
            ),
            50: pagina([item(2)]),
        }
    )
    transporte = httpx.MockTransport(handler)

    primero = fetch_mal.run(
        args, raw_dir=tmp_path, client_id=CLAVE, transport=transporte
    )
    assert primero == fetch_mal.EXIT_OK
    antes = (tmp_path / "anime.json").read_text(encoding="utf-8")

    segundo = fetch_mal.run(
        args, raw_dir=tmp_path, client_id=CLAVE, transport=transporte
    )

    assert segundo == fetch_mal.EXIT_OK
    assert len(handler.calls) == 2
    assert (tmp_path / "anime.json").read_text(encoding="utf-8") == antes


def test_run_sin_filtros_no_pisa_anime_json(tmp_path, caplog):
    handler = Paginador({0: pagina([item(1, num_scoring_users=1)])})
    (tmp_path / "anime.json").write_text("[]", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        codigo = fetch_mal.run(
            make_args(limit=50, min_scoring_users=1000),
            raw_dir=tmp_path,
            client_id=CLAVE,
            transport=httpx.MockTransport(handler),
        )

    assert codigo == fetch_mal.EXIT_PROBLEMA
    assert (tmp_path / "anime.json").read_text(encoding="utf-8") == "[]"
    assert "Ningún anime" in caplog.text


@pytest.mark.parametrize(
    "flag", [("max_items", 0), ("min_scoring_users", -1), ("limit", 0)]
)
def test_run_con_flags_invalidos_sale_2(tmp_path, flag, caplog):
    with caplog.at_level(logging.ERROR):
        codigo = fetch_mal.run(
            make_args(**{flag[0]: flag[1]}),
            raw_dir=tmp_path,
            client_id=CLAVE,
            transport=None,
        )

    assert codigo == fetch_mal.EXIT_CONFIG
    assert not tmp_path.joinpath("pages").exists()


def test_run_sin_paginas_sale_1(tmp_path, caplog):
    handler = Paginador({}, status=404)

    with caplog.at_level(logging.ERROR):
        codigo = fetch_mal.run(
            make_args(limit=50),
            raw_dir=tmp_path,
            client_id=CLAVE,
            transport=httpx.MockTransport(handler),
        )

    assert codigo == fetch_mal.EXIT_PROBLEMA
    assert not (tmp_path / "anime.json").exists()


# --- detalles de la CLI, la petición y la escritura atómica -----------------


def test_limit_se_redondea_a_paginas_completas():
    assert fetch_mal.items_de_paginas(50) == 50
    assert fetch_mal.items_de_paginas(1) == 50
    assert fetch_mal.items_de_paginas(51) == 100


def test_page_limit_esta_dentro_del_maximo_de_mal():
    assert fetch_mal.PAGE_LIMIT <= 500


def test_offset_calculado_por_pagina():
    assert fetch_mal.page_offset(1) == 0
    assert fetch_mal.page_offset(2) == fetch_mal.PAGE_LIMIT
    assert fetch_mal.page_offset(4) == 3 * fetch_mal.PAGE_LIMIT


def test_la_peticion_pide_los_19_campos_y_no_la_url(tmp_path):
    handler = Paginador({0: pagina([item(1)])})

    with make_client(handler) as client:
        fetch_mal.fetch_page(client, 1, tmp_path / "pages")

    params = handler.calls[0]
    assert params["ranking_type"] == "all"
    assert params["limit"] == str(fetch_mal.PAGE_LIMIT)
    assert params["fields"].split(",") == list(fetch_mal.FIELDS)
    assert len(fetch_mal.FIELDS) == 19
    assert CLAVE not in json.dumps(params)


def test_read_pages_ordena_por_numero(tmp_path):
    directory = tmp_path / "pages"
    for page in (1, 2, 10, 11):
        guardar_pagina(directory, page, pagina([item(page)]))
    directory.joinpath("page_99.json").write_text("{roto", encoding="utf-8")

    paginas = fetch_mal.read_pages(directory)

    assert [entrada["data"][0]["node"]["id"] for entrada in paginas] == [1, 2, 10, 11]


def test_escritura_atomica_no_deja_temporales(tmp_path):
    path = tmp_path / "pages" / "page_1.json"
    fetch_mal.write_json_atomic(path, {"title": "Ação"})

    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "Ação"
    assert [p.name for p in path.parent.iterdir()] == ["page_1.json"]


def test_settings_tiene_mal_client_id():
    assert isinstance(fetch_mal.settings.mal_client_id, str)


def test_silence_http_client_logs():
    fetch_mal.silence_http_client_logs()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
