"""Tests de `app/services/modelo_onnx.py`.

La referencia de equivalencia del tokenizador son los IDs congelados en
`tests/fixtures/tokenizer_ids.json`, y **no** `tokenizers.Tokenizer` en crudo,
porque AutoTokenizer ES el tokenizador con el que sentence-transformers construyó
el índice. Los dos difieren en un punto concreto: el de crudo emite un `▁` (id 6)
cuando el texto termina en espacio en blanco, AutoTokenizer no, y el índice no lo
tiene. Contra el objetivo equivocado (el de crudo) la equivalencia salía 204/205;
contra el índice, 205/205.

Congelar los IDs sustituye a llamar a `transformers.AutoTokenizer` en cada run:
transformers y sentence-transformers se quitaron del proyecto en la Fase A porque
solo existían para esta comprobación y cargaban ~1 GB a la imagen. El índice no
vuelve a cambiar, así que la referencia congelada no pierde vigencia; lo que se
pierde es avisar si un transformers nuevo segmentara distinto, que no es un
riesgo real con el índice ya construido.

Los dos fixtures se generaron el 3 de octubre de 2026, con sentence-transformers
6.1.0 y transformers 5.18.0 todavía instalados y antes del `uv sync` que los
quitó. Desde `backend/`, con `PYTHONPATH=.` para que `tests` sea importable:

    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    MODELO = "intfloat/multilingual-e5-small"
    st = SentenceTransformer(MODELO)
    vectores = st.encode(TEXTOS_DORADOS, normalize_embeddings=True)
    np.savez("tests/fixtures/vectores_dorados.npz", modelo=np.array(MODELO),
             textos=np.array(TEXTOS_DORADOS), vectores=vectores)

    tokenizador = AutoTokenizer.from_pretrained(MODELO)
    ids = tokenizador(texto, truncation=False)["input_ids"]  # o con max_length=512

Como en `test_embedder.py`, se cargan artefactos reales de la caché de Hugging
Face: sin red si ya están bajados, con red la primera vez. Lo que no se prueba
aquí es la velocidad ni la RAM, que eso es de medición y no de tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import hf_hub_download

from app.services.modelo_onnx import (
    ARCHIVO_SPM,
    MODELO_CON_PAREJA,
    ModeloOnnx,
    TokenizadorE5,
)

RUTA_FIXTURES = Path(__file__).parent / "fixtures"

# Los cuatro textos del guardia de vectores, en el mismo orden que en el npz.
TEXTOS_DORADOS = [
    "query: me siento solo y cansado, quiero algo que me levante el animo",
    "passage: El sastre de la mafia. Tipo: movie. Géneros: Crimen, Drama.",
    "passage: " + "palabra " * 600,
    "query: quiero algo corto y raro que me haga reír de verdad",
]

# Textos con lo que el tokenizador se ha equivocado alguna vez o ha estado a
# punto: acentos, CJK, emoji, controles, espacios repetidos, vacío, y el final
# en U+200E, que es como termina un documento real del índice (el doc 90 de la
# Fase 0: con una regla de `▁` final se rompía, y sin ella coincide).
TEXTOS = [
    (
        "passage: El sastre de la mafia. Tipo: movie. Géneros: Crimen, Drama. "
        "Temas: ambush, shotgun, machismo."
    ),
    "query: me siento solo y cansado, quiero algo que me levante el animo",
    "consulta con\xa0espacio\xa0no\xa0separable",
    "全角文字とtest",
    "emoji 🎬 y 😀",
    "   espacios   múltiples   ",
    "Tildes: áéíóú ñ ü ¿qué?",
    "\x01control\x7f",
    "",
    "final con espacio ",
    "final con tab\t",
    "final con newline\n",
    "termina en U+200E\u200e",
    "solo espacios   ",
    "palabra\u200bsin\u200bespacio",
    "me aburri....",
    "no se....",
]


class IndiceCongelado:
    """Los IDs que daba `transformers.AutoTokenizer`, resuelta desde el JSON.

    Mantiene las dos formas de llamar que los tests usan, sin truncar y con
    `truncation=True, max_length=512`, para que el resto del fichero no cambie
    respecto a cuando la referencia era la llamada real. Un texto que no esté
    congelado falla con su texto en el mensaje: regenera el fixture.
    """

    def __init__(self, ruta: Path) -> None:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        assert datos["modelo"] == MODELO_CON_PAREJA, (
            f"el fixture es de {datos['modelo']!r} y el modelo es {MODELO_CON_PAREJA!r}"
        )
        self._entradas: list[dict] = datos["entradas"]

    def __call__(
        self, texto: str, *, truncation: bool = False, max_length: int | None = None
    ) -> dict[str, list[int]]:
        if truncation and max_length != 512:
            raise AssertionError("los IDs congelados solo cubren max_length=512")
        maximo = 512 if truncation else None
        for entrada in self._entradas:
            if entrada["texto"] == texto and entrada.get("max_length") == maximo:
                return {"input_ids": entrada["ids"]}
        raise AssertionError(f"texto sin congelar en el fixture: {texto!r}")


@pytest.fixture(scope="module")
def tokenizador() -> TokenizadorE5:
    ruta = hf_hub_download(MODELO_CON_PAREJA, ARCHIVO_SPM)
    return TokenizadorE5(ruta)


@pytest.fixture(scope="module")
def indice() -> IndiceCongelado:
    return IndiceCongelado(RUTA_FIXTURES / "tokenizer_ids.json")


def test_los_ids_coinciden_con_el_tokenizador_del_indice(
    tokenizador: TokenizadorE5, indice: IndiceCongelado
) -> None:
    """El contrato entero: mismos IDs que los que tiene el índice.

    Si esto falla, las consultas nuevas se calculan en un espacio de vectores
    distinto al de los 9397 documentos ya indexados, y no hay error que lo
    delate: salen resultados razonables pero equivocados.
    """
    for texto in TEXTOS:
        nuestro = tokenizador(textos=[texto])["input_ids"][0]
        suyo = indice(texto, truncation=False)["input_ids"]
        assert nuestro == suyo, f"ids distintos para {texto!r}"


def test_divergencias_conocidas_y_medidas(
    tokenizador: TokenizadorE5, indice: IndiceCongelado
) -> None:
    """Fija por test las dos divergencias que sí existen, y que son aceptadas.

    Medido sobre el catálogo completo: sentencepiece difiere del tokenizer del
    índice en 2 de 9397 docs (desempates de igual probabilidad del Unigram que
    Rust y C++ resuelven distinto, y `de....` es un ejemplo reproducible en
    corto) y en 0 de 5 consultas. NEL (U+0085) es el único carácter de
    normalización que difiere, y no aparece en el catálogo. El impacto es un
    token segmentado al revés, del orden del ruido de cuantización INT8 que ya
    se aceptó en la Fase 0.

    Si este test llega a fallar, alguien igualó estas rutas: revalida el corpus
    entero antes de alegrarte, porque el resto de asunciones dependen de saber
    exactamente dónde no llegamos.
    """
    for texto in ["pelicula de.... terror", "final con NEL\x85"]:
        nuestro = tokenizador(textos=[texto])["input_ids"][0]
        suyo = indice(texto, truncation=False)["input_ids"]
        assert nuestro != suyo, f"divergencia conocida desapareció en {texto!r}"


def test_truncamiento_es_igual_al_del_indice(
    tokenizador: TokenizadorE5, indice: IndiceCongelado
) -> None:
    """A 512 incluidos los especiales, con `</s>` conservado, como transformers.

    Solo el 0,06 % de los docs pasa de 512, pero esos 6 docs están en el índice
    truncados de esta forma y la comprobación en el texto más largo del
    catálogo salió idéntica.
    """
    texto = "passage: " + "palabra " * 600
    nuestro = tokenizador([texto], add_special_tokens=True, truncation=True)["input_ids"][0]
    suyo = indice(texto, truncation=True, max_length=512)["input_ids"]

    assert nuestro == suyo
    assert len(nuestro) == 512
    assert nuestro[0] == 0 and nuestro[-1] == 2


def test_medir_truncamiento_usa_al_tokenizador(
    tokenizador: TokenizadorE5, indice: IndiceCongelado
) -> None:
    """`build_index.medir_truncamiento` llama al tokenizador como transformers.

    Es el consumidor real de `TokenizadorE5.__call__`: si la firma se le mueve,
    script y tests se enteran aquí y no en una reconstrucción del índice.
    """
    from scripts.build_index import medir_truncamiento

    largos = [
        "passage: corto",
        "passage: " + "palabra " * 600,
    ]
    informe = medir_truncamiento(tokenizador, ["a", "b"], largos, 512)

    assert informe is not None
    assert informe["total"] == 2
    assert informe["exceden"] == 1
    # Los largos incluyen <s></s>, igual que los contaría transformers.
    corto = tokenizador([largos[0]])["input_ids"][0]
    assert informe["maximo_observado"] == len(tokenizador([largos[1]])["input_ids"][0])
    assert corto == indice(largos[0], truncation=False)["input_ids"]


def test_rechaza_una_cadena_suelta(tokenizador: TokenizadorE5) -> None:
    """Un `str` iterado recorre caracteres: fallar es mejor que devolver basura."""
    with pytest.raises(TypeError):
        tokenizador("hola")  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def modelo() -> ModeloOnnx:
    return ModeloOnnx(MODELO_CON_PAREJA)


def test_los_vectores_estan_normalizados(modelo: ModeloOnnx) -> None:
    """Sin L2, la distancia coseno de ChromaDB no sería la del índice."""
    vectores = modelo.encode(["query: una consulta", "passage: un documento"])
    assert vectores.shape == (2, 384)
    normas = np.linalg.norm(vectores, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-5)
    assert np.isfinite(vectores).all()

    vacio = modelo.encode([])
    assert vacio.shape == (0, 384)


def test_el_relleno_del_lote_no_rompe_los_vectores(modelo: ModeloOnnx) -> None:
    """Lo que el contrato garantiza y lo que no, medido, sin adornos.

    - Mismo ancho: `[corto, corto]` contra `[corto]` da exactamente lo mismo,
      así que las filas del lote son independientes y la máscara se consume
      (verificado aparte: en un ancho fijo, quitar la máscara cambia los
      estados con coseno 0,90).
    - Ancho distinto: el relleno comparte escala con las filas reales. El
      grafo tiene 48 `DynamicQuantizeLinear`, es decir, ORT calcula la escala
      de las activaciones a runtime sobre TODO el tensor del lote: añadir
      filas, o cambiar el contenido del relleno, desplaza la cuantización de
      todas. Medido: coseno ≥ 0,997 entre un mismo texto codificado solo y
      junto a otro de otro largo. PyTorch no se ve afectado (coseno 1,0); es
      ruido propio del cuantizado, de la misma familia que el 0,9956 medio de
      la Fase 0 contra torch, ya aceptado.
    - En producción la consulta viaja sola (`encode([texto])`), así que su
      vector es determinista: mismo texto, mismo vector. Solo un reconstruido
      de índice con lotes mezclados hereda esta variación, y eso sería
      decisión explícita, no accidente.

    Si este test falla, cambió el comportamiento del cuantizador: revisa la
    banda medida antes de tocar el umbral.
    """
    corto = "query: consulta corta"
    otro = "query: otra consulta un poco mas larga pero corta"
    largo = "passage: " + "palabra " * 600

    solo = modelo.encode([corto])[0]
    duplicado = modelo.encode([corto, corto])[0]
    assert np.allclose(solo, duplicado, atol=1e-6)

    for otro_texto in (otro, largo):
        en_lote = modelo.encode([corto, otro_texto])[0]
        coseno = float(np.dot(solo, en_lote))
        assert coseno >= 0.99, f"coseno {coseno:.4f} contra {otro_texto!r}"


def test_embedder_lo_lee(modelo: ModeloOnnx) -> None:
    """La interfaz que consume `Embedder` (dimensión, longitud, tokenizador)."""
    from app.services.embedder import Embedder

    envoltorio = Embedder(modelo, MODELO_CON_PAREJA)
    assert envoltorio.dimension == 384
    assert envoltorio.max_seq_length == 512
    assert envoltorio.tokenizador is not None
    assert envoltorio.tokenizador is modelo.tokenizer


def test_los_vectores_parecen_los_dorados(modelo: ModeloOnnx) -> None:
    """El guardia de fondo: ONNX INT8 + sentencepiece contra los pesos del índice.

    Compara contra `tests/fixtures/vectores_dorados.npz`, con los vectores que
    daba sentence-transformers 6.1.0, en vez de volver a cargar el modelo: la
    Fase A quitó sentence-transformers y torch del proyecto, así que la
    comprobación ya no puede hacerse contra la librería original. Un fixture
    firmado por la librería original y guardado en git mantiene el mismo
    objetivo: si ONNX o el tokenizador dejan de reproducir los pesos con los
    que se construyó el índice, el coseno se cae aquí y no en una búsqueda.

    Son los mismos números de referencia que en la Fase 0, donde se midió contra
    PyTorch: coseno mínimo 0,9912 en docs y 0,9963 en consultas (media 0,9956).
    El umbral va en 0,98 para que pille errores gordos (pooling sin máscara,
    normalización olvidada, tokenizador equivocado) sin que un desempate de
    Unigram o la versión de onnxruntime lo ponga en riesgo. Los cuatro textos
    cubren una consulta corta, un passage, otro texto que pasa de 512 tokens
    (o sea, la ruta de truncamiento) y una segunda consulta.

    Se embebe **un texto por llamada**, que es como viaja la consulta en
    producción y como se midió en la Fase 0. En un solo lote el relleno
    comparte escala con las filas (ver
    `test_el_relleno_del_lote_no_rompe_los_vectores`) y el texto largo medido
    se va a 0,9789 contra 0,9932 estando solo: el umbral estaría midiendo el
    relleno, no la equivalencia con el índice.
    """
    datos = np.load(RUTA_FIXTURES / "vectores_dorados.npz", allow_pickle=False)
    assert str(datos["modelo"]) == MODELO_CON_PAREJA
    textos = [str(texto) for texto in datos["textos"]]
    assert textos == TEXTOS_DORADOS, "el npz y la lista de la suite han divergido"
    esperados = datos["vectores"]
    assert esperados.shape == (len(TEXTOS_DORADOS), 384)
    for esperado, texto in zip(esperados, textos):
        nuestro = modelo.encode([texto])[0]
        coseno = float(np.dot(esperado, nuestro))
        assert coseno >= 0.98, f"coseno {coseno:.4f} para {texto!r}"
