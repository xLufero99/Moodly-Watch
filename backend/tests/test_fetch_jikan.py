"""Tests de scripts/fetch_jikan.py sin tocar la red: todo va por httpx.MockTransport."""

import json
import logging
from collections import Counter

import httpx
import pytest

from scripts import fetch_jikan
from scripts.fetch_jikan import Status, parse_args

MISSING = object()


@pytest.fixture(autouse=True)
def sin_pausas(monkeypatch):
    """Sin esperas: los reintentos y la pausa entre peticiones no interesan aquí."""
    monkeypatch.setattr(fetch_jikan, "REQUEST_PAUSE_S", 0.0)
    monkeypatch.setattr(fetch_jikan, "_sleep", lambda _seconds: None)


def pagina(items: list[dict], has_next: bool = False) -> dict:
    return {"pagination": {"has_next_page": has_next, "current_page": 1}, "data": items}


def anime(mal_id: int, **overrides) -> dict:
    """Un ítem de /top/anime con los campos que nos interesan."""
    entrada = {
        "mal_id": mal_id,
        "title": f"Anime {mal_id}",
        "title_english": f"Anime {mal_id} EN",
        "title_japanese": f"アニメ{mal_id}",
        "type": "TV",
        "episodes": 24,
        "status": "Finished Airing",
        "duration": "24 min per ep",
        "rating": "PG-13",
        "score": 8.5,
        "scored_by": 5000,
        "popularity": 42,
        "synopsis": "Una historia.",
        "year": 1998,
        "season": "spring",
        "aired": {"from": "1998-04-03T00:00:00+00:00", "to": None},
        "genres": [{"mal_id": 1, "name": "Action"}, {"mal_id": 2, "name": "Drama"}],
        "themes": [{"mal_id": 29, "name": "Isekai"}],
        "demographics": [],
        "studios": [{"mal_id": 14, "name": "Sunrise"}],
        "images": {
            "jpg": {
                "image_url": "https://cdn.example/1.jpg",
                "large_image_url": "https://cdn.example/1l.jpg",
            },
            "webp": {"image_url": "https://cdn.example/1.webp"},
        },
        "url": "https://myanimelist.net/anime/1",
        "background": None,
    }
    entrada.update(overrides)
    return entrada


class Contador:
    """Handler con contador, para afirmar cuántas peticiones hicieron falta."""

    def __init__(self, respuestas: list[httpx.Response]) -> None:
        self.respuestas = respuestas
        self.calls = 0
        self.params: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.params.append(dict(request.url.params))
        return self.respuestas[min(self.calls, len(self.respuestas)) - 1]


class Paginador:
    """Handler que responde una página distinta por cada número pedido."""

    def __init__(self, pages: dict[int, dict], status: int = 200) -> None:
        self.pages = pages
        self.status = status
        self.calls: list[int] = []
        self.params: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/top/anime")
        self.params.append(dict(request.url.params))
        page = int(request.url.params["page"])
        self.calls.append(page)
        if page not in self.pages:
            return httpx.Response(self.status, json={"status": 404})
        return httpx.Response(200, json=self.pages[page])


def make_client(handler) -> httpx.Client:
    return fetch_jikan.build_client(transport=httpx.MockTransport(handler))


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
    guardar_pagina(directory, 1, pagina([anime(1)], has_next=False))
    handler = Contador([httpx.Response(500, json={"message": "nope"})])

    with make_client(handler) as client:
        estado, payload = fetch_jikan.fetch_page(client, 1, directory)

    assert estado is Status.SKIPPED
    assert handler.calls == 0
    assert [item["mal_id"] for item in payload["data"]] == [1]


def test_reanudar_no_repide_ninguna_pagina(tmp_path):
    directory = tmp_path / "pages"
    guardar_pagina(directory, 1, pagina([anime(1)], has_next=True))
    guardar_pagina(directory, 2, pagina([anime(2)], has_next=False))
    handler = Paginador({})
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_jikan.fetch_pages(client, directory, 100, counters)

    assert handler.calls == []
    assert counters[Status.SKIPPED] == 2
    assert counters[Status.SAVED] == 0


def test_pagina_corrupta_se_vuelve_a_pedir(tmp_path, caplog):
    directory = tmp_path / "pages"
    directory.mkdir(parents=True)
    (directory / "page_1.json").write_text("{roto", encoding="utf-8")
    handler = Paginador({1: pagina([anime(1)], has_next=False)})

    with make_client(handler) as client, caplog.at_level(logging.WARNING):
        estado, payload = fetch_jikan.fetch_page(client, 1, directory)

    assert estado is Status.SAVED
    assert handler.calls == [1]
    assert "corrupta" in caplog.text
    assert payload["data"][0]["mal_id"] == 1


# --- (b) 429 con Retry-After -------------------------------------------------


def test_429_con_retry_after_se_reintenta_y_luego_funciona(
    tmp_path, monkeypatch, caplog
):
    esperas: list[float] = []
    monkeypatch.setattr(fetch_jikan, "_sleep", esperas.append)
    handler = Contador(
        [
            httpx.Response(429, headers={"retry-after": "3"}, json={"message": "slow"}),
            httpx.Response(200, json=pagina([anime(1)], has_next=False)),
        ]
    )

    with make_client(handler) as client, caplog.at_level(logging.WARNING):
        estado, payload = fetch_jikan.fetch_page(client, 1, tmp_path / "pages")

    assert estado is Status.SAVED
    assert handler.calls == 2
    assert payload["data"][0]["mal_id"] == 1
    assert "429" in caplog.text


def test_429_sin_retry_after_usa_backoff_exponencial(tmp_path, monkeypatch):
    esperas: list[float] = []
    monkeypatch.setattr(fetch_jikan, "_sleep", esperas.append)
    handler = Contador(
        [
            httpx.Response(429, json={"message": "slow"}),
            httpx.Response(429, json={"message": "slow"}),
            httpx.Response(200, json=pagina([anime(1)], has_next=False)),
        ]
    )

    with make_client(handler) as client:
        estado, _ = fetch_jikan.fetch_page(client, 1, tmp_path / "pages")

    assert estado is Status.SAVED
    assert handler.calls == 3
    # 1 s y luego 2 s, el backoff por defecto de tenacity.
    assert esperas == [1.0, 2.0]


# --- (c) el aplanado deduplica y filtra -------------------------------------


def test_aplana_deduplica_por_mal_id():
    pages = [
        pagina(
            [
                anime(1),
                anime(2, type="Special"),
                anime(3, scored_by=10),
                anime(4, synopsis="   "),
            ]
            + [{"title": "Sin mal_id"}]
        ),
        pagina([anime(1, title="Repetido")]),
    ]

    lista, stats = fetch_jikan.flatten(pages, fetch_jikan.DEFAULT_MIN_SCORED_BY)

    assert [item["mal_id"] for item in lista] == [1]
    assert lista[0]["title"] == "Anime 1"
    assert stats["duplicados"] == 1
    assert stats["fuera_de_tipo"] == 1
    assert stats["pocos_votos"] == 1
    assert stats["sin_sinopsis"] == 1
    assert stats["sin_id"] == 1
    assert stats["vistos"] == 6


def test_aplana_extrae_los_campos_pedidos():
    lista, _ = fetch_jikan.flatten([pagina([anime(1)])], 0)

    assert lista == [
        {
            "mal_id": 1,
            "title": "Anime 1",
            "title_english": "Anime 1 EN",
            "title_japanese": "アニメ1",
            "type": "TV",
            "episodes": 24,
            "status": "Finished Airing",
            "duration": "24 min per ep",
            "rating": "PG-13",
            "score": 8.5,
            "scored_by": 5000,
            "popularity": 42,
            "synopsis": "Una historia.",
            "year": 1998,
            "season": "spring",
            "aired_from": "1998-04-03T00:00:00+00:00",
            "genres": ["Action", "Drama"],
            "themes": ["Isekai"],
            "demographics": [],
            "studios": ["Sunrise"],
            "poster_url": "https://cdn.example/1l.jpg",
        }
    ]


def test_min_scored_by_baja_el_umbral():
    pages = [pagina([anime(1, scored_by=999), anime(2, scored_by=1000)])]

    lista, _ = fetch_jikan.flatten(pages, 1000)

    assert [item["mal_id"] for item in lista] == [2]


@pytest.mark.parametrize(
    ("tipo", "permitido"),
    [
        ("TV", True),
        ("tv", True),
        ("Movie", True),
        ("ONA", True),
        ("OVA", True),
        ("Special", False),
        (None, False),
        ("", False),
    ],
)
def test_filtro_de_tipo_ignora_mayusculas(tipo, permitido):
    assert fetch_jikan.es_tipo_permitido(tipo) is permitido


def test_no_se_manda_type_en_la_peticion(tmp_path):
    handler = Paginador({1: pagina([anime(1)], has_next=False)})

    with make_client(handler) as client:
        fetch_jikan.fetch_page(client, 1, tmp_path / "pages")

    assert handler.params == [{"page": "1", "limit": str(fetch_jikan.PAGE_LIMIT)}]


# --- (d) la paginación se detiene con has_next_page=false -------------------


def test_paginacion_se_detiene_con_has_next_page_false(tmp_path):
    directory = tmp_path / "pages"
    handler = Paginador(
        {1: pagina([anime(1)], has_next=True), 2: pagina([anime(2)], has_next=False)}
    )
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_jikan.fetch_pages(client, directory, 100, counters)

    assert handler.calls == [1, 2]
    assert counters[Status.SAVED] == 2
    assert not (directory / "page_3.json").exists()


def test_paginacion_se_detiene_al_llegar_al_tope_de_items(tmp_path):
    directory = tmp_path / "pages"
    handler = Paginador(
        {
            1: pagina([anime(1), anime(2)], has_next=True),
            2: pagina([anime(3), anime(4)], has_next=True),
        }
    )
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_jikan.fetch_pages(client, directory, 3, counters)

    assert handler.calls == [1, 2]
    assert not (directory / "page_3.json").exists()


def test_paginacion_se_detiene_con_pagina_vacia(tmp_path):
    directory = tmp_path / "pages"
    handler = Paginador({1: pagina([], has_next=True)})
    counters: Counter = Counter()

    with make_client(handler) as client:
        fetch_jikan.fetch_pages(client, directory, 100, counters)

    assert handler.calls == [1]
    assert counters[Status.EMPTY] == 1
    assert not (directory / "page_1.json").exists()


# --- (e) 404 se salta sin romper --------------------------------------------


def test_404_se_salta_sin_escribir_archivo(tmp_path, caplog):
    directory = tmp_path / "pages"
    handler = Paginador({}, status=404)
    counters: Counter = Counter()

    with make_client(handler) as client, caplog.at_level(logging.WARNING):
        fetch_jikan.fetch_pages(client, directory, 100, counters)

    assert handler.calls == [1]
    assert counters[Status.MISSING] == 1
    assert not (directory / "page_1.json").exists()
    assert "404" in caplog.text


def test_otro_4xx_aborta_con_mensaje_claro(tmp_path, caplog):
    handler = Contador([httpx.Response(400, json={"status": 400, "message": "bad"})])

    with make_client(handler) as client, caplog.at_level(logging.WARNING):
        estado, payload = fetch_jikan.fetch_page(client, 1, tmp_path / "pages")

    assert estado is Status.FAILED
    assert payload == {}
    assert handler.calls == 1
    assert "400" in caplog.text


def test_5xx_persistentes_cuentan_como_failed(tmp_path):
    handler = Contador([httpx.Response(503, json={"message": "down"})])

    with make_client(handler) as client:
        estado, _ = fetch_jikan.fetch_page(client, 1, tmp_path / "pages")

    assert estado is Status.FAILED
    assert handler.calls == fetch_jikan.MAX_ATTEMPTS
    assert not (tmp_path / "pages" / "page_1.json").exists()


# --- run completo, aplanado y códigos de salida ------------------------------


def test_run_descarga_y_aplana(tmp_path):
    handler = Paginador(
        {
            1: pagina([anime(1), anime(2, type="Music")], has_next=True),
            2: pagina([anime(3, scored_by=5), anime(4)], has_next=False),
        }
    )

    codigo = fetch_jikan.run(
        make_args(limit=50), raw_dir=tmp_path, transport=httpx.MockTransport(handler)
    )

    assert codigo == fetch_jikan.EXIT_OK
    assert handler.calls == [1, 2]
    anime_json = json.loads((tmp_path / "anime.json").read_text(encoding="utf-8"))
    assert [item["mal_id"] for item in anime_json] == [1, 4]


def test_run_reanudable_genera_el_mismo_anime(tmp_path):
    args = make_args(limit=50)
    handler = Paginador(
        {1: pagina([anime(1)], has_next=True), 2: pagina([anime(2)], has_next=False)}
    )
    transporte = httpx.MockTransport(handler)

    primero = fetch_jikan.run(args, raw_dir=tmp_path, transport=transporte)
    assert primero == fetch_jikan.EXIT_OK
    antes = (tmp_path / "anime.json").read_text(encoding="utf-8")

    segundo = fetch_jikan.run(args, raw_dir=tmp_path, transport=transporte)

    assert segundo == fetch_jikan.EXIT_OK
    assert handler.calls == [1, 2]
    assert (tmp_path / "anime.json").read_text(encoding="utf-8") == antes


def test_run_sin_filtros_deja_anime_json_vacio_y_no_pisa(tmp_path, caplog):
    handler = Paginador({1: pagina([anime(1, scored_by=1)], has_next=False)})
    (tmp_path / "anime.json").write_text("[]", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        codigo = fetch_jikan.run(
            make_args(limit=50, min_scored_by=1000),
            raw_dir=tmp_path,
            transport=httpx.MockTransport(handler),
        )

    assert codigo == fetch_jikan.EXIT_PROBLEMA
    assert (tmp_path / "anime.json").read_text(encoding="utf-8") == "[]"
    assert "Ningún anime" in caplog.text


@pytest.mark.parametrize(
    "flag",
    [("max_items", 0), ("min_scored_by", -1), ("limit", 0)],
)
def test_run_con_flags_invalidos_sale_2(tmp_path, flag, caplog):
    with caplog.at_level(logging.ERROR):
        codigo = fetch_jikan.run(
            make_args(**{flag[0]: flag[1]}), raw_dir=tmp_path, transport=None
        )

    assert codigo == fetch_jikan.EXIT_CONFIG
    assert not tmp_path.joinpath("pages").exists()


def test_run_sin_paginas_sale_1(tmp_path, caplog):
    handler = Paginador({}, status=404)

    with caplog.at_level(logging.ERROR):
        codigo = fetch_jikan.run(
            make_args(limit=50),
            raw_dir=tmp_path,
            transport=httpx.MockTransport(handler),
        )

    assert codigo == fetch_jikan.EXIT_PROBLEMA
    assert not (tmp_path / "anime.json").exists()


# --- detalles de la CLI y escritura atómica ---------------------------------


def test_limit_se_redondea_a_paginas_completas():
    assert fetch_jikan.items_de_paginas(50) == 50
    assert fetch_jikan.items_de_paginas(1) == 25
    assert fetch_jikan.items_de_paginas(26) == 50


def test_page_limit_es_el_maximo_de_jikan():
    assert fetch_jikan.PAGE_LIMIT == 25


def test_read_pages_ordena_por_numero(tmp_path):
    directory = tmp_path / "pages"
    for page in (1, 2, 10, 11):
        guardar_pagina(directory, page, pagina([anime(page)]))
    directory.joinpath("page_99.json").write_text("{roto", encoding="utf-8")

    paginas = fetch_jikan.read_pages(directory)

    assert [item["data"][0]["mal_id"] for item in paginas] == [1, 2, 10, 11]


def test_flatten_ignora_entradas_basura():
    pagina_basura = {"pagination": {"has_next_page": False}, "data": ["texto", None]}

    lista, stats = fetch_jikan.flatten([pagina_basura], 0)

    assert lista == []
    assert stats["vistos"] == 0


def test_escritura_atomica_no_deja_temporales(tmp_path):
    path = tmp_path / "pages" / "page_1.json"
    fetch_jikan.write_json_atomic(path, {"title": "Ação"})

    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "Ação"
    assert [p.name for p in path.parent.iterdir()] == ["page_1.json"]


def test_silence_http_client_logs():
    fetch_jikan.silence_http_client_logs()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
