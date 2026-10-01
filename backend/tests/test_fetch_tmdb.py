"""Tests de scripts/fetch_tmdb.py sin tocar la red: todo va por httpx.MockTransport."""

import json
import logging

import httpx
import pytest

from scripts import fetch_tmdb
from scripts.fetch_tmdb import AuthError, Status, parse_args

TEST_KEY = "clave-de-prueba-123"
MISSING = object()


@pytest.fixture(autouse=True)
def sin_pausas(monkeypatch):
    """Sin esperas: los reintentos y las pausas entre peticiones no interestan aquí."""
    monkeypatch.setattr(fetch_tmdb, "REQUEST_PAUSE_S", 0.0)
    monkeypatch.setattr(fetch_tmdb, "_sleep", lambda _seconds: None)


class FakeTmdb:
    """Transport de mentira: responde por (path, language) y guarda las llamadas.

    El cliente tiene `base_url` con el /3 de la API, así que las rutas se
    comparan sin ese prefijo.
    """

    def __init__(self, routes: dict[tuple[str, str | None], object]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/3")
        language = request.url.params.get("language")
        self.calls.append((path, language))
        assert request.url.params.get("api_key") == TEST_KEY
        payload = self.routes.get((path, language), self.routes.get((path, None), MISSING))
        if payload is MISSING:
            return httpx.Response(404, json={"status_code": 34, "status_message": "not found"})
        return httpx.Response(200, json=payload)

    def languages(self, path: str) -> list[str | None]:
        return [lang for called, lang in self.calls if called == path]


def make_client(fake) -> httpx.Client:
    return fetch_tmdb.build_client(TEST_KEY, transport=httpx.MockTransport(fake))


def make_args(**overrides):
    argv = []
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return parse_args(argv)


def read_item(raw_dir, media_type: str, item_id: int) -> dict:
    path = raw_dir / media_type / f"{item_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


class Contador:
    """Handler con contador, para afirmar cuántas peticiones hicieron falta."""

    def __init__(self, respuestas: list[httpx.Response]) -> None:
        self.respuestas = respuestas
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.respuestas[min(self.calls, len(self.respuestas)) - 1]


# --- (a) fallback de idioma -------------------------------------------------


def test_overview_vacio_dispara_fallback_a_en_us(tmp_path):
    fake = FakeTmdb(
        {
            ("/movie/1", "es-ES"): {"id": 1, "title": "Sin resumen", "overview": "  "},
            ("/movie/1", "en-US"): {"id": 1, "title": "Sin resumen", "overview": "A quiet drama."},
        }
    )

    with make_client(fake) as client:
        estado = fetch_tmdb.fetch_detail(client, "movie", 1, tmp_path)

    assert estado is Status.SAVED
    guardado = read_item(tmp_path, "movie", 1)
    assert guardado["overview"] == "A quiet drama."
    assert guardado["overview_lang"] == "en"
    assert fake.languages("/movie/1") == ["es-ES", "en-US"]


def test_overview_en_es_no_dispera_fallback(tmp_path):
    fake = FakeTmdb({("/movie/2", "es-ES"): {"id": 2, "title": "Con resumen", "overview": "Uno."}})

    with make_client(fake) as client:
        estado = fetch_tmdb.fetch_detail(client, "movie", 2, tmp_path)

    assert estado is Status.SAVED
    guardado = read_item(tmp_path, "movie", 2)
    assert guardado["overview"] == "Uno."
    assert "overview_lang" not in guardado
    assert fake.languages("/movie/2") == ["es-ES"]


def test_fallback_tambien_pide_keywords(tmp_path):
    fake = FakeTmdb(
        {
            ("/tv/3", "es-ES"): {"id": 3, "title": "Sin resumen", "overview": ""},
            ("/tv/3", "en-US"): {"id": 3, "title": "Sin resumen", "overview": "Slow burn."},
        }
    )
    pedidos: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        respuesta = fake(request)
        pedidos.append(request.url.params.get("append_to_response"))
        return respuesta

    with fetch_tmdb.build_client(TEST_KEY, transport=httpx.MockTransport(handler)) as client:
        fetch_tmdb.fetch_detail(client, "tv", 3, tmp_path)

    assert pedidos == ["keywords", "keywords"]
    assert read_item(tmp_path, "tv", 3)["overview_lang"] == "en"


# --- (b) reanudable ---------------------------------------------------------


def test_item_ya_guardado_no_se_vuelve_a_pedir(tmp_path):
    path = tmp_path / "movie" / "5.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"id": 5}', encoding="utf-8")
    fake = FakeTmdb({})

    with make_client(fake) as client:
        estado = fetch_tmdb.fetch_detail(client, "movie", 5, tmp_path)

    assert estado is Status.SKIPPED
    assert fake.calls == []
    assert path.read_text(encoding="utf-8") == '{"id": 5}'


# --- (c) 404 se salta sin romper -------------------------------------------


def test_404_se_salta_sin_escribir_archivo(tmp_path, caplog):
    fake = FakeTmdb({})
    caplog.set_level(logging.INFO)

    with make_client(fake) as client, caplog.at_level(logging.WARNING):
        estado = fetch_tmdb.fetch_detail(client, "movie", 404, tmp_path)

    assert estado is Status.MISSING
    assert not (tmp_path / "movie" / "404.json").exists()
    assert len(fake.calls) == 1
    assert TEST_KEY not in caplog.text


def test_run_continua_tras_un_404(tmp_path, caplog):
    fake = FakeTmdb(
        {
            ("/genre/movie/list", "es-ES"): {"genres": [{"id": 1, "name": "Acción"}]},
            ("/genre/tv/list", "es-ES"): {"genres": [{"id": 2, "name": "Drama"}]},
            ("/discover/movie", "es-ES"): {
                "page": 1,
                "total_pages": 1,
                "results": [{"id": 1}, {"id": 2}],
            },
            ("/discover/tv", "es-ES"): {"page": 1, "total_pages": 1, "results": [{"id": 7}]},
            ("/movie/1", "es-ES"): {"id": 1, "overview": "Uno."},
            ("/tv/7", "es-ES"): {"id": 7, "overview": "Siete."},
        }
    )
    caplog.set_level(logging.INFO)

    with caplog.at_level(logging.WARNING):
        codigo = fetch_tmdb.run(
            make_args(),
            raw_dir=tmp_path,
            api_key=TEST_KEY,
            transport=httpx.MockTransport(fake),
        )

    assert codigo == fetch_tmdb.EXIT_OK
    assert read_item(tmp_path, "movie", 1)["id"] == 1
    assert read_item(tmp_path, "tv", 7)["id"] == 7
    assert not (tmp_path / "movie" / "2.json").exists()
    assert TEST_KEY not in caplog.text


# --- 401 aborta, 5xx se reintenta ------------------------------------------


def test_logs_nunca_exponen_la_clave(tmp_path, caplog, monkeypatch):
    """httpx loggea la URL completa en INFO, y la URL lleva la api_key."""
    cliente_httpx = logging.getLogger("httpx")
    monkeypatch.setattr(cliente_httpx, "level", logging.INFO)

    fetch_tmdb.silence_http_client_logs()
    assert cliente_httpx.level == logging.WARNING

    fake = FakeTmdb({("/movie/1", "es-ES"): {"id": 1, "overview": "Uno."}})
    with make_client(fake) as client, caplog.at_level(logging.INFO):
        fetch_tmdb.fetch_detail(client, "movie", 1, tmp_path)

    assert TEST_KEY not in caplog.text


def test_401_aborta_de_inmediato(tmp_path, caplog):
    handler = Contador([httpx.Response(401, json={"status_message": "Invalid API key"})])

    with (
        fetch_tmdb.build_client(TEST_KEY, transport=httpx.MockTransport(handler)) as client,
        caplog.at_level(logging.ERROR),
        pytest.raises(AuthError) as error,
    ):
        fetch_tmdb.fetch_detail(client, "movie", 1, tmp_path)

    assert handler.calls == 1
    assert "TMDB_API_KEY" in str(error.value)
    assert TEST_KEY not in str(error.value)
    assert TEST_KEY not in caplog.text
    assert not (tmp_path / "movie").exists()


def test_503_se_reintenta_y_luego_funciona(tmp_path, caplog):
    handler = Contador(
        [
            httpx.Response(503, json={"status_message": "Service unavailable"}),
            httpx.Response(200, json={"id": 9, "overview": "Al final funciona."}),
        ]
    )

    with (
        fetch_tmdb.build_client(TEST_KEY, transport=httpx.MockTransport(handler)) as client,
        caplog.at_level(logging.WARNING),
    ):
        estado = fetch_tmdb.fetch_detail(client, "movie", 9, tmp_path)

    assert estado is Status.SAVED
    assert handler.calls == 2
    assert TEST_KEY not in caplog.text
    assert "503" in caplog.text


def test_errores_persistentes_cuentan_como_failed(tmp_path):
    handler = Contador([httpx.Response(503, json={"status_message": "Service unavailable"})])

    with fetch_tmdb.build_client(TEST_KEY, transport=httpx.MockTransport(handler)) as client:
        estado = fetch_tmdb.fetch_detail(client, "movie", 11, tmp_path)

    assert estado is Status.FAILED
    assert handler.calls == fetch_tmdb.MAX_ATTEMPTS
    assert not (tmp_path / "movie" / "11.json").exists()


# --- códigos de salida y reanudación del run completo -----------------------


def test_run_sin_credencial_sale_2(tmp_path):
    assert fetch_tmdb.run(make_args(), raw_dir=tmp_path, api_key="") == (
        fetch_tmdb.EXIT_SIN_CREDENCIAL
    )


def test_run_reanudable_no_repite_una_segunda_vez(tmp_path):
    fake = FakeTmdb(
        {
            ("/genre/movie/list", "es-ES"): {"genres": []},
            ("/genre/tv/list", "es-ES"): {"genres": []},
            ("/discover/movie", "es-ES"): {"page": 1, "total_pages": 1, "results": [{"id": 1}]},
            ("/discover/tv", "es-ES"): {"page": 1, "total_pages": 1, "results": [{"id": 7}]},
            ("/movie/1", "es-ES"): {"id": 1, "overview": "Uno."},
            ("/tv/7", "es-ES"): {"id": 7, "overview": "Siete."},
        }
    )
    args = make_args()
    transporte = httpx.MockTransport(fake)

    primero = fetch_tmdb.run(args, raw_dir=tmp_path, api_key=TEST_KEY, transport=transporte)
    assert primero == fetch_tmdb.EXIT_OK
    assert ("/movie/1", "es-ES") in fake.calls
    assert ("/tv/7", "es-ES") in fake.calls

    # Los ids se regeneran, los ítems ya guardados no se vuelven a pedir.
    segundo = fetch_tmdb.run(args, raw_dir=tmp_path, api_key=TEST_KEY, transport=transporte)
    assert segundo == fetch_tmdb.EXIT_OK
    assert fake.calls.count(("/movie/1", "es-ES")) == 1
    assert fake.calls.count(("/tv/7", "es-ES")) == 1
    assert fake.calls.count(("/discover/movie", "es-ES")) == 2


def test_limit_reparte_entre_peliculas_y_series(tmp_path):
    fake = FakeTmdb(
        {
            ("/genre/movie/list", "es-ES"): {"genres": []},
            ("/genre/tv/list", "es-ES"): {"genres": []},
            ("/discover/movie", "es-ES"): {
                "page": 1,
                "total_pages": 1,
                "results": [{"id": n} for n in (1, 2, 3)],
            },
            ("/discover/tv", "es-ES"): {
                "page": 1,
                "total_pages": 1,
                "results": [{"id": n} for n in (7, 8, 9)],
            },
        }
    )
    detalles: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/3")
        if path.startswith(("/movie/", "/tv/")):
            detalles.append(path)
            return httpx.Response(200, json={"id": 1, "overview": "x"})
        return fake(request)

    codigo = fetch_tmdb.run(
        make_args(limit=4),
        raw_dir=tmp_path,
        api_key=TEST_KEY,
        transport=httpx.MockTransport(handler),
    )

    assert codigo == fetch_tmdb.EXIT_OK
    assert detalles == ["/movie/1", "/tv/7", "/movie/2", "/tv/8"]
    ids = json.loads((tmp_path / "ids_movie.json").read_text(encoding="utf-8"))
    assert ids["count"] <= 4
    assert ids["min_votes"] == fetch_tmdb.DEFAULT_MIN_VOTES


# --- escritura atómica ------------------------------------------------------


def test_escritura_atomica_no_deja_temporales(tmp_path):
    path = tmp_path / "movie" / "1.json"
    fetch_tmdb.write_json_atomic(path, {"id": 1, "title": "Ação"})

    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "Ação"
    assert [p.name for p in tmp_path.joinpath("movie").iterdir()] == ["1.json"]


def test_escritura_atomica_reemplaza_sin_dejar_el_anterior(tmp_path):
    path = tmp_path / "movie" / "1.json"
    fetch_tmdb.write_json_atomic(path, {"id": 1, "title": "Antes"})
    fetch_tmdb.write_json_atomic(path, {"id": 1, "title": "Después"})

    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "Después"
    assert not (tmp_path / "movie" / "1.json.tmp").exists()
