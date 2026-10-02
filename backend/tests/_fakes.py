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
    """Stub de `SentenceTransformer` con lo justo que le pregunta `Embedder`."""

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
    """

    def __init__(self, contenido: str | None = None, *, error: Exception | None = None):
        self.contenido = contenido
        self.error = error
        self.llamadas: list[dict] = []
        # Espejo de `groq_client.chat.completions.create`, que es a dos niveles.
        self.chat = _ChatFalso(self)

    def _create(self, **kwargs):
        self.llamadas.append(kwargs)
        if self.error is not None:
            raise self.error
        return RespuestaFalsa(self.contenido)


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