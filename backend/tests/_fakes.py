"""Embedders falsos para probar sin descargar el modelo.

Viven aquí y no en el fichero del índice porque los usan dos testfiles: el del
constructor del índice y el del buscador. Repetir el stub en los dos significaría que una
corrección en el stub se aplica a un lado y se olvida en el otro, que es como se cuelan
pruebas que miden algo distinto de lo que dicen medir.

`EmbedderFalso` **no aplica prefijos**, a propósito. El prefijo se prueba en
`test_build_index.py` contra `Embedder` de verdad con el modelo stub encima. Si este stub
se pusiera el prefijo él mismo, la prueba del prefijo probaría el prefijo del stub y no el
del `Embedder`, que es lo que de verdad importa.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

DIMENSION = 4


def vector_para(texto: str, dimension: int = DIMENSION) -> list[float]:
    """Vector determinista y normalizado, a partir del texto.

    Determinista y no `hash()`, porque `hash()` de cadenas lleva sal por proceso y el
    vector cambiaría entre ejecuciones.
    """
    digest = hashlib.sha256(texto.encode("utf-8")).digest()
    valores = [digest[i % len(digest)] / 255.0 for i in range(dimension)]
    norma = math.sqrt(sum(v * v for v in valores)) or 1.0
    return [v / norma for v in valores]


class ModeloFalso:
    """Stub del modelo real (`ModeloOnnx`) con lo justo que le pregunta `Embedder`."""

    def __init__(self) -> None:
        self.dimension = DIMENSION
        self.max_seq_length = 512
        self.tokenizador = None
        self.recibidos: list[list[str]] = []

    def __str__(self) -> str:
        return "falso/stub"

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, textos, batch_size=32, **kwargs) -> np.ndarray:
        self.recibidos.append(list(textos))
        return np.array([vector_para(t) for t in textos], dtype="float32")


class EmbedderFalso:
    """Sustituye a `Embedder` en las pruebas. No aplica prefijos.

    La dimensión es 4 por defecto porque a los unitarios les da igual y es más rápido de
    calcular. Contra un ChromaDB de verdad hay que pedir la que tenga la colección: si
    no, `query` falla con `Collection expecting embedding with dimension of 384, got 4`.
    Para eso está el argumento.
    """

    max_seq_length = 512
    tokenizador = None

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension
        self.nombre = f"falso/dimension-{dimension}"

    def incrustar_documentos(self, textos, batch_size: int = 32):
        return [vector_para(t, self.dimension) for t in textos]

    def incrustar_consultas(self, consultas):
        return [vector_para(t, self.dimension) for t in consultas]


class MensajeFalso:
    def __init__(self, content: str | None) -> None:
        self.content = content


class EleccionFalsa:
    def __init__(self, content: str | None) -> None:
        self.message = MensajeFalso(content)


class RespuestaFalsa:
    """La forma anidada que devuelve el SDK de Groq: `choices[0].message.content`."""

    def __init__(self, content: str | None) -> None:
        self.choices = [EleccionFalsa(content)]


class ClienteGroqFalso:
    """Doble del cliente de Groq. Sin red, sin key, sin esperar.

    Se le pasa lo que hay que devolver, o la excepción que hay que tirar. Un test por
    caso y sin ramales: si el doble decidiera cómo degradar, los tests del parser
    estarían probando el doble y no el parser.

    Guarda las llamadas en `llamadas` para poder comprobar que el `response_format` lleva
    `strict: True` y el schema que toca.

El contenido es una lista en vez de una, y por defecto devuelve la primera. Los tests
    que necesitan dos respuestas distintas (el pipeline completo hace una llamada al
    parser y otra al explainer) pasan dos y las va sacando en orden, que es más explícito
    que un doble que decide por su cuenta a quién responde.

    Cada elemento de `respuestas` puede ser un `str` para devolver, o una `Exception` para
    tirar. Así se puede tener el parser funcionando y el explainer caído en el mismo test,
    que es justo el caso que importa: el pipeline tiene que sobrevivir a que se le rompa la
    mitad cara.
    """

    def __init__(
        self,
        contenido: str | None = None,
        *,
        error: Exception | None = None,
        respuestas: list[str | Exception | None] | None = None,
    ):
        if respuestas is not None and contenido is not None:
            raise ValueError("pasa `contenido` o `respuestas`, no los dos")
        self.contenido = contenido
        self.error = error
        self.respuestas = list(respuestas) if respuestas is not None else [contenido]
        self.llamadas: list[dict] = []
        # Espejo de `groq_client.chat.completions.create`, que es a dos niveles.
        self.chat = _ChatFalso(self)

    def _create(self, **kwargs):
        self.llamadas.append(kwargs)
        if self.error is not None:
            raise self.error
        # Se gasta una respuesta por llamada. Si se acaban, se repite la última, que es
        # lo que hace un cliente real cuando dos módulos piden y solo había una cosa que
        # decir. Un test que se quede sin respuestas debería notarlo, y por eso `llamadas`
        # se puede mirar en vez de adivinar.
        siguiente = self.respuestas.pop(0) if len(self.respuestas) > 1 else self.respuestas[0]
        if isinstance(siguiente, Exception):
            raise siguiente
        return RespuestaFalsa(siguiente)


class _ChatFalso:
    def __init__(self, cliente: ClienteGroqFalso) -> None:
        self._cliente = cliente
        self.completions = _CompletionsFalso(cliente)


class _CompletionsFalso:
    def __init__(self, cliente: ClienteGroqFalso) -> None:
        self._cliente = cliente

    def create(self, **kwargs):
        return self._cliente._create(**kwargs)


class ColeccionFalsa:
    """Doble de la colección de ChromaDB: devuelve lo que se le pone.

    Se usa para comprobar cómo se construye la llamada a `query` (el `where`, el
    `n_results`, el embedding que llega) sin levantar ChromaDB. Lo que se le pasa en
    `respuesta` es lo que devuelve, sin más.
    """

    def __init__(self, respuesta: dict | None = None) -> None:
        self.respuesta = respuesta if respuesta is not None else {}
        self.llamadas: list[dict] = []

    def query(self, **kwargs):
        self.llamadas.append(kwargs)
        return self.respuesta

    @property
    def ultima(self) -> dict:
        assert self.llamadas, "no se llamó a query"
        return self.llamadas[-1]

# --------------------------------------------------------------------------- #
# Dobles para el explainer
# --------------------------------------------------------------------------- #

DOCUMENTO_EJEMPLO = (
    "Lock & Stock. Tipo: movie. Géneros: Comedia, Crimen. Temas: ambush, shotgun, "
    "machismo. Eddie convence a tres amigos para jugarse sus ahorros en una partida "
    "de cartas contra Harry el Hacha, un mafioso del barrio."
)


class ResultadoFalso:
    """Doble de `ResultadoBusqueda` para el explainer, sin ChromaDB detrás.

    Se construye con los mismos nombres de campo que el dataclass real, así que si el
    explainer deja de leerlos por atributo, este test falla también.
    """

    def __init__(
        self,
        id: str,
        title: str = "Un título",
        media_type: str = "movie",
        year: int | None = 2000,
        genres: list[str] | None = None,
        document: str | None = None,
        score: float = 1.0,
    ):
        self.id = id
        self.title = title
        self.media_type = media_type
        self.year = year
        self.genres = genres if genres is not None else ["Comedia"]
        self.document = document
        self.score = score

    def __repr__(self) -> str:
        return f"ResultadoFalso({self.id!r}, {self.title!r})"


def groq_explica(
    por_id: dict[str, str],
    *,
    en_otro_orden: bool = False,
    con_ids_inventados: list[str] | None = None,
) -> ClienteGroqFalso:
    """Un cliente falso que devuelve el array de explicaciones del explainer.

    `por_id` va del id al texto. `en_otro_orden=True` los invierte, que es el caso que
    separa el emparejamiento por `id` del emparejamiento por posición: si el código
    emparejara con `zip`, con el array invertido cada explicación caería en la tarjeta
    equivocada y el test lo vería.
    """
    entradas = [{"id": i, "explicacion": t} for i, t in por_id.items()]
    if en_otro_orden:
        entradas.reverse()
    for inventado in con_ids_inventados or []:
        entradas.append({"id": inventado, "explicacion": f"Algo inventado {inventado}"})
    return ClienteGroqFalso(json.dumps({"explicaciones": entradas}))


def groq_explica_mal() -> ClienteGroqFalso:
    """Groq responde algo que no es el array de explicaciones."""
    return ClienteGroqFalso(json.dumps({"explicaciones": "no es una lista"}))


def groq_explica_rota() -> ClienteGroqFalso:
    """Groq responde texto que ni siquiera es JSON."""
    return ClienteGroqFalso("esto no es un JSON, es una frase")
