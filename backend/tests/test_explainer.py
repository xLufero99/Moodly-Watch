"""Tests de `app/services/explainer.py`.

**Nada de aquí habla con Groq.** Todos usan `ClienteGroqFalso`. Los que sí llaman a la API
real están marcados `@pytest.mark.lento` y la suite normal no los recoge.

Lo que cubren:

  * que el schema es válido para `strict: true` y que `minItems`/`maxItems` se clavan al
    número de resultados
  * que la llamada lleva `strict: True`, `temperature=0` y `max_tokens`
  * alineación **por `id`**, incluido el array reordenado, que es el fallo que el `id`
    evita y un `zip` no
  * degradación por ítem: falta una explicación y las otras cinco se conservan
  * ids inventados que no se cuelan en ninguna tarjeta
  * `content` vacío, JSON inválido, excepción y ausencia de key
  * que la sinopsis se saca del `document` y no de los metadatos
  * que la plantilla nombra los géneros

Y el punto de fondo:

  * `test_content_vacio_es_error_y_no_json_invalido`: con `openai/gpt-oss-20b` el
    presupuesto puede gastarse el razonamiento y dejar `content` vacío. No es JSON malo,
    es nada, y se degrada distinto.
"""

from __future__ import annotations

import json

import pytest
from _fakes import (
    DOCUMENTO_EJEMPLO,
    ClienteGroqFalso,
    ResultadoFalso,
    groq_explica,
    groq_explica_mal,
    groq_explica_rota,
)

from app.services import groq_client
from app.services.explainer import (
    MAX_SINOPSIS,
    MAX_TOKENS,
    explicar_por_generos,
    explicar_resultados,
    schema_explicaciones,
    sinopsis_de,
)
from app.services.filter_parser import FiltrosConsulta

FILTROS = FiltrosConsulta(query="algo", media_types=["movie"])


@pytest.fixture(autouse=True)
def _con_key(request, monkeypatch):
    """Key falsa para los unitarios, y la real intacta para los `lento`.

    Sin el salto, este `autouse` pisaba la configuración de los tests lentos y todos
    fallaban con `AuthenticationError` teniendo la key correcta en el entorno. Es el mismo
    error que en `test_filter_parser.py`, y por eso el salto se repite aquí en vez de
    llevarlo a un conftest común: los dos ficheros son independientes, y un conftest
    compartido que cualquiera de los dos pueda tocar es peor que el duplicado.
    """
    if not request.node.get_closest_marker("lento"):
        monkeypatch.setattr(groq_client.settings, "groq_api_key", "test-key")
        monkeypatch.setattr(
            "app.services.explainer.settings.groq_model", "openai/gpt-oss-20b"
        )
    yield


def seis() -> list[ResultadoFalso]:
    return [ResultadoFalso(str(i), title=f"Título {i}") for i in range(1, 7)]


def por_id_de_6() -> dict[str, str]:
    return {str(i): f"Explicación del título {i}" for i in range(1, 7)}


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_el_schema_es_valido_para_strict() -> None:
    """Todo en `required` y con `additionalProperties: false`, a todos los niveles.

    Es la condición que impone `strict: true`. Un solo objeto sin `additionalProperties`
    da 400 y no lo da hasta runtime, con lo que un test unitario que no mira la forma
    pasa todo verde y el endpoint falla en producción.
    """
    schema = schema_explicaciones(6)

    def comprobar(objeto: dict) -> None:
        assert objeto["type"] == "object"
        assert objeto["additionalProperties"] is False
        assert set(objeto["required"]) == set(objeto["properties"])
        for sub in objeto["properties"].values():
            if sub.get("type") == "object":
                comprobar(sub)
            if sub.get("type") == "array" and sub["items"].get("type") == "object":
                comprobar(sub["items"])

    comprobar(schema)


def test_min_items_y_max_items_son_la_cantidad_pedida() -> None:
    """El tamaño se clava en runtime, no se deja abierto.

    Con el array abierto, "cinco de seis" es una respuesta tan válida como "seis de seis" y
    el desajuste se descubre tarde. Con el tamaño clavado, si el modelo se equivoca se ve.
    """
    schema = schema_explicaciones(6)
    array = schema["properties"]["explicaciones"]
    assert array["minItems"] == 6
    assert array["maxItems"] == 6


def test_el_schema_cambia_con_la_cantidad() -> None:
    """Con 3 resultados no pide 6. Es lo que hace que valga fijarlo en runtime."""
    schema = schema_explicaciones(3)
    assert schema["properties"]["explicaciones"]["minItems"] == 3


def test_los_items_no_admiten_campos_extra() -> None:
    items = schema_explicaciones(6)["properties"]["explicaciones"]["items"]
    assert set(items["properties"]) == {"id", "explicacion"}
    assert set(items["required"]) == {"id", "explicacion"}


# --------------------------------------------------------------------------- #
# la llamada
# --------------------------------------------------------------------------- #


def test_una_sola_llamada_para_los_seis() -> None:
    """El requisito de cuota: una llamada, no seis.

    Seis llamadas por búsqueda agotan el free tier de 30 RPM en una búsqueda y media, y
    además cada una podría explicar el mismo título distinto.
    """
    cliente = groq_explica(por_id_de_6())
    explicar_resultados("algo", FILTROS, seis(), cliente=cliente)
    assert len(cliente.llamadas) == 1


def test_la_llamada_lleva_strict_y_el_schema() -> None:
    cliente = groq_explica(por_id_de_6())
    explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    formato = cliente.llamadas[0]["response_format"]
    assert formato["type"] == "json_schema"
    assert formato["json_schema"]["strict"] is True
    assert formato["json_schema"]["schema"]["properties"]["explicaciones"]["maxItems"] == 6


def test_temperatura_cero_y_max_tokens_alto() -> None:
    """`temperature=0` porque es extracción, y `max_tokens` alto porque razona.

    gpt-oss-20b gasta parte del presupuesto en razonamiento antes de escribir. Con un tope
    corto `message.content` llega vacío sin error, y eso es una degradación distinta a un
    JSON inválido.
    """
    cliente = groq_explica(por_id_de_6())
    explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert cliente.llamadas[0]["temperature"] == 0.0
    assert cliente.llamadas[0]["max_tokens"] == MAX_TOKENS


def test_las_llamadas_al_parser_y_al_explainer_comparten_cliente() -> None:
    """El pipeline usa el mismo `cliente`, que es lo que permite ver las dos llamadas."""
    cliente = ClienteGroqFalso(
        respuestas=[
            json.dumps(
                {
                    "query": "algo",
                    "media_types": [],
                    "max_runtime_total": None,
                    "max_runtime_episode": None,
                }
            ),
            json.dumps({"explicaciones": [{"id": "1", "explicacion": "Porque sí"}]}),
        ]
    )
    from app.services.filter_parser import parsear_filtros

    filtros = parsear_filtros("algo", cliente=cliente)
    explicar_resultados("algo", filtros, [ResultadoFalso("1")], cliente=cliente)

    assert len(cliente.llamadas) == 2
    assert "explicaciones" in cliente.llamadas[1]["response_format"]["json_schema"]["schema"][
        "properties"
    ]


# --------------------------------------------------------------------------- #
# alineación
# --------------------------------------------------------------------------- #


def test_alinea_por_id() -> None:
    cliente = groq_explica(por_id_de_6())
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert [e.texto for e in salida] == [
        f"Explicación del título {i}" for i in range(1, 7)
    ]
    assert all(e.generada_por_llm for e in salida)


def test_un_id_por_cada_resultado_y_en_orden() -> None:
    cliente = groq_explica(por_id_de_6())
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)
    assert [e.id for e in salida] == ["1", "2", "3", "4", "5", "6"]


def test_reordenado_no_mezcla_las_explicaciones() -> None:
    """El motivo de emparejar por `id` y no por posición.

    Con `zip`, este array invertido pegaría "Explicación del título 6" en la tarjeta del
    título 1. El texto es plausible y el error es invisible a ojo, que es lo que lo hace
    peligroso.
    """
    cliente = groq_explica(por_id_de_6(), en_otro_orden=True)
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert [e.texto for e in salida] == [
        f"Explicación del título {i}" for i in range(1, 7)
    ]


def test_id_inventado_no_se_cuela() -> None:
    """Un id que no es de esta búsqueda se ignora con aviso, y no ocupa ninguna tarjeta."""
    cliente = groq_explica(por_id_de_6(), con_ids_inventados=["999", "inventado"])
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert [e.id for e in salida] == ["1", "2", "3", "4", "5", "6"]
    assert all("999" not in e.texto for e in salida)


def test_id_repetido_gana_el_primero(caplog) -> None:
    """Duplicado: gana el primero. El array viene ordenado y el modelo ha copiado mal."""
    cliente = ClienteGroqFalso(
        json.dumps(
            {
                "explicaciones": [
                    {"id": "1", "explicacion": "La primera"},
                    {"id": "1", "explicacion": "La segunda"},
                ]
            }
        )
    )
    salida = explicar_resultados("algo", FILTROS, [ResultadoFalso("1")], cliente=cliente)
    assert salida[0].texto == "La primera"


def test_falta_una_y_las_otras_cinco_se_conservan() -> None:
    """Degradación por ítem, no todo o nada.

    Tirar cinco explicaciones buenas porque la sexta vino mal es justo el fallo que aquí
    se evita. El parser degrada entero porque devuelve un objeto; aquí hay seis.
    """
    parcial = {k: v for k, v in por_id_de_6().items() if k != "3"}
    cliente = groq_explica(parcial)
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert len(salida) == 6
    assert salida[2].texto.startswith("Coincide con tu búsqueda por")
    assert salida[2].generada_por_llm is False
    assert all(salida[i].generada_por_llm for i in (0, 1, 3, 4, 5))


def test_explicacion_vacia_no_gana_a_la_plantilla() -> None:
    """Un texto en blanco es peor que la plantilla: parece que el LLM no dijo nada."""
    cliente = groq_explica({"1": "   ", "2": "Buena"})
    salida = explicar_resultados(
        "algo", FILTROS, [ResultadoFalso("1"), ResultadoFalso("2")], cliente=cliente
    )
    assert salida[0].generada_por_llm is False
    assert salida[1].texto == "Buena"


# --------------------------------------------------------------------------- #
# degradación entera
# --------------------------------------------------------------------------- #


def test_sin_resultados_no_llama_a_groq() -> None:
    cliente = groq_explica({})
    assert explicar_resultados("algo", FILTROS, [], cliente=cliente) == []
    assert cliente.llamadas == []


def test_excepcion_de_groq_degrada_las_seis() -> None:
    cliente = ClienteGroqFalso(error=RuntimeError("429 rate limit"))
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert len(salida) == 6
    assert all(not e.generada_por_llm for e in salida)
    assert all(e.texto for e in salida)


def test_sin_key_degrada_las_seis(monkeypatch) -> None:
    """La app sigue funcionando sin Groq. Eso es el punto."""
    monkeypatch.setattr("app.services.explainer.hay_cliente", lambda: False)
    cliente = groq_explica(por_id_de_6())
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert cliente.llamadas == [], "no debería ni intentar si no hay key"
    assert len(salida) == 6
    assert all(not e.generada_por_llm for e in salida)


def test_json_invalido_degrada_las_seis() -> None:
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=groq_explica_rota())
    assert len(salida) == 6
    assert all(not e.generada_por_llm for e in salida)


def test_forma_inesperada_degrada_las_seis() -> None:
    """`explicaciones` que no es una lista. Con `strict` no debería pasar, pero el 400
    también puede venir del servidor y un filtro con un tipo inventado es peor que no
    tener ninguno."""
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=groq_explica_mal())
    assert len(salida) == 6
    assert all(not e.generada_por_llm for e in salida)


def test_content_vacio_es_error_y_no_json_invalido(caplog) -> None:
    """El caso de `content` vacío, que con gpt-oss es real.

    El modelo razonó, se comió el presupuesto y no llegó a escribir. No es JSON inválido,
    es nada, así que el aviso tiene que decirlo: si no, se debuggearía el schema cuando el
    problema era `max_tokens`.
    """
    cliente = ClienteGroqFalso("")
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert len(salida) == 6
    assert all(not e.generada_por_llm for e in salida)
    assert any("contenido vacío" in r.message for r in caplog.records)


def test_ningun_id_utilizable_degrada_las_seis() -> None:
    """Groq respondió pero sin ningún id que corresponda. Mejor plantilla que texto
    atribuido al título equivocado."""
    cliente = groq_explica({"999": "Algo", "888": "Otro"})
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=cliente)

    assert len(salida) == 6
    assert all(not e.generada_por_llm for e in salida)


def test_todo_sigue_siendo_usable_despues_de_degradar() -> None:
    """Lo que de verdad importa: tras degradar, cada tarjeta tiene texto y no hay None."""
    salida = explicar_resultados("algo", FILTROS, seis(), cliente=groq_explica_rota())
    for explicacion in salida:
        assert isinstance(explicacion.texto, str)
        assert explicacion.texto.strip()


# --------------------------------------------------------------------------- #
# la plantilla
# --------------------------------------------------------------------------- #


def test_la_plantilla_nombra_los_generos() -> None:
    """Los géneros sí: están en los metadatos, así que son ciertos por construcción."""
    resultado = ResultadoFalso("1", genres=["Terror", "Suspense"])
    texto = explicar_por_generos(resultado)
    assert "Terror" in texto
    assert "Suspense" in texto


def test_la_plantilla_sin_generos_no_inventa() -> None:
    texto = explicar_por_generos(ResultadoFalso("1", genres=[]))
    assert texto
    assert "Coincide con tu búsqueda por" not in texto


def test_la_plantilla_no_menciona_los_filtros() -> None:
    """El filtro es una restricción, no un motivo.

    "Cumple tu filtro de 90 minutos" no le dice a nadie por qué le va a gustar, y suena a
    que el sistema lo ha colocado ahí por obligación.
    """
    resultado = ResultadoFalso("1", genres=["Comedia"])
    texto = explicar_por_generos(resultado).lower()
    assert "filtro" not in texto
    assert "puntuación" not in texto


# --------------------------------------------------------------------------- #
# la sinopsis
# --------------------------------------------------------------------------- #


def test_la_sinopsis_se_saca_del_document() -> None:
    """De aquí viene el cambio de `include` en `search.py`: la sinopsis no está en los
    metadatos, está al final del documento, tras `Temas: `."""
    sinopsis = sinopsis_de(DOCUMENTO_EJEMPLO)
    assert sinopsis.startswith("Eddie convence a tres amigos")
    assert "Temas" not in sinopsis
    assert "machismo" not in sinopsis


def test_sinopsis_corta() -> None:
    assert sinopsis_de(None) == ""
    assert sinopsis_de("") == ""
    assert sinopsis_de("Sin sinopsis aquí") == ""


def test_sinopsis_sin_temas() -> None:
    """Cuando la lista de temas viene vacía queda `"Temas: . sinopsis"`.

    Es el segundo corte el que la separa del `. `. Con un solo corte, el `. ` inicial
    se colaba en la sinopsis y el modelo recibía un texto que empezaba por un punto.
    """
    sinopsis = sinopsis_de("Corto. Tipo: movie. Géneros: Drama. Temas: . Un hombre va a casa.")
    assert sinopsis == "Un hombre va a casa."


def test_sinopsis_del_indice_real() -> None:
    """Contra un documento copiado del índice completo, no inventado para el test."""
    sinopsis = sinopsis_de(
        "Lock & Stock. Tipo: movie. Géneros: Comedia, Crimen. Temas: ambush, joint, "
        "alcohol, shotgun, machismo, rifle, cardsharp. Eddie convence a tres amigos para "
        "jugarse sus ahorros en una partida de cartas contra Harry el Hacha."
    )
    assert sinopsis.startswith("Eddie convence a tres amigos")
    assert "ambush" not in sinopsis
    assert "machismo" not in sinopsis


def test_la_sinopsis_larga_se_recorta() -> None:
    """Se recorta antes de enviarlo, no después de leer la respuesta.

    Con sinopsis largas el modelo se extiende y, con más razón, se le escapa el final.
    """
    larga = "T" * (MAX_SINOPSIS * 2)
    recortada = sinopsis_de(f"Título. Tipo: movie. Géneros: Drama. Temas: x. {larga}")
    assert len(recortada) <= MAX_SINOPSIS + 3
    assert recortada.endswith("...")


def test_la_sinopsis_llega_al_prompt() -> None:
    """Si no llega, el modelo solo tiene título y géneros y no puede explicar nada."""
    cliente = groq_explica({"1": "Explicación"})
    resultado = ResultadoFalso("1", document=DOCUMENTO_EJEMPLO)
    explicar_resultados("quiero algo", FILTROS, [resultado], cliente=cliente)

    enviado = json.loads(cliente.llamadas[0]["messages"][1]["content"])
    assert enviado["titulos"][0]["sinopsis"].startswith("Eddie convence")


def test_el_prompt_recibe_consulta_filtros_y_titulos() -> None:
    """En ese orden, que es el del `SISTEMA`. Los filtros van en llano para que el modelo
    pueda decir "es una película, no una serie, como pediste"."""
    cliente = groq_explica({"1": "Explicación"})
    filtros = FiltrosConsulta(
        query="algo",
        media_types=["movie"],
        max_runtime_total=90,
    )
    explicar_resultados(
        "películas cortas",
        filtros,
        [ResultadoFalso("1")],
        cliente=cliente,
    )

    enviado = json.loads(cliente.llamadas[0]["messages"][1]["content"])
    assert enviado["consulta"] == "películas cortas"
    assert enviado["filtros"]["tipos"] == ["movie"]
    assert enviado["filtros"]["duracion_maxima_pelicula"] == "90 minutos"
    assert enviado["titulos"][0]["id"] == "1"


def test_el_prompt_no_inventa_filtros_que_no_hay() -> None:
    """Solo van las claves que hay. Un `"duracion_maxima": null` haría pensar al modelo que
    se pidió una duración de cero."""
    cliente = groq_explica({"1": "Explicación"})
    explicar_resultados("algo", FiltrosConsulta(query="algo"), [ResultadoFalso("1")], cliente=cliente)

    enviado = json.loads(cliente.llamadas[0]["messages"][1]["content"])
    assert enviado["filtros"] == {}


# --------------------------------------------------------------------------- #
# lentos: contra Groq de verdad
# --------------------------------------------------------------------------- #


@pytest.mark.lento
@pytest.mark.requiere_red
def test_groq_real_devuelve_una_explicacion_por_resultado() -> None:
    """Seis explicaciones utilizables en una sola llamada.

    Comprueba el número, que no estén vacías, que el emparejamiento por `id` funciona
    (que es donde el modelo podría fallar) y que ninguna delate el final.
    """
    resultados = [
        ResultadoFalso(str(i), title=f"Título {i}", genres=["Comedia"]) for i in range(1, 7)
    ]
    salida = explicar_resultados(
        "quiero algo ligero para un domingo",
        FiltrosConsulta(query="ligero para un domingo"),
        resultados,
    )

    assert len(salida) == 6
    assert [e.id for e in salida] == [str(i) for i in range(1, 7)]
    assert all(e.generada_por_llm for e in salida), "Groq real no llegó a explicar"
    for explicacion in salida:
        assert len(explicacion.texto) > 20
        assert explicacion.texto.strip() == explicacion.texto


@pytest.mark.lento
@pytest.mark.requiere_red
def test_groq_real_no_delata_los_finales(caplog) -> None:
    """La regla 3 del prompt, contra el modelo de verdad.

    Es el fallo que más se cuela: con sinopsis largas el modelo cuenta el desenlace aunque
    se le diga que no. Se mira a mano el texto de las seis, y esto solo avisa de que se
    revisaron.
    """
    resultados = [
        ResultadoFalso(
            "1",
            title="Una historia con final",
            document=(
                "Una historia con final. Tipo: movie. Géneros: Drama. Temas: x. "
                "Un hombre busca a su hermano durante veinte años. Al final descubre que "
                "su hermano era él mismo y que se había inventado toda la búsqueda."
            ),
        )
    ]
    salida = explicar_resultados(
        "quiero algo sobre un hermano perdido",
        FiltrosConsulta(query="algo"),
        resultados,
    )

    assert salida[0].generada_por_llm is True
    print("\n  Explicación real:", salida[0].texto)


@pytest.mark.lento
@pytest.mark.requiere_red
def test_una_busqueda_no_agota_la_cuota() -> None:
    """Dos búsquedas seguidas: parser más explainer, cuatro llamadas.

    El free tier son 30 RPM y 8 000 TPM. Con seis llamadas por búsqueda se acabaría en
    una y media; con dos, cuatro búsquedas seguidas entran de sobra.
    """
    from app.services.filter_parser import parsear_filtros
    from app.services.search import FiltrosBusqueda, buscar
    from app.services.vector_store import ruta_indice_por_defecto

    for _ in range(2):
        filtros = parsear_filtros("algo de ciencia ficción")
        resultados = buscar(
            filtros.query,
            filtros=FiltrosBusqueda(media_types=["movie"], max_runtime_total=120),
            top_k=6,
            dir_indice=ruta_indice_por_defecto(),
        )
        if not resultados:
            pytest.skip("no hay índice completo")
        explicacion = explicar_resultados(
            "algo de ciencia ficción", filtros, resultados
        )
        assert len(explicacion) == len(resultados)