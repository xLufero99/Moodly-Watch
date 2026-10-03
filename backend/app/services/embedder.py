"""Embeddings de sinopsis con `intfloat/multilingual-e5-small`, y sus prefijos obligatorios.

Este módulo existe por una razón concreta: el prefijo que se aplica al indexar y el
que se aplica al buscar tienen que ser el mismo esquema, y si cada script decide el
suyo por su cuenta hay una forma fácil de que el índice quede construido con `passage:`
y se consulte con otra cosa, o al revés. Eso no da error, da resultados malos y es
difícil de detectar. Aquí se centralizan los dos, y `scripts/build_index.py` y
`scripts/search_demo.py` los importan en vez de repetirlos.

El prefijo no es una convención propia: la ficha del modelo dice que se entrenó así y
que sin él el rendimiento baja. Para E5 es `query: ` en el lado de la pregunta y
`passage: ` en el lado del texto que se recupera. O sea que la búsqueda aquí es
asimétrica: la consulta y el documento no llevan el mismo prefijo, y ese desajuste
es intencionado.

Lo que sí es un detalle propio de este módulo: aplicar el prefijo es idempotente. Si
el texto ya viene con el prefijo puesto no se duplica, porque entre el script que
construye y el que busca es fácil acabar pasando un texto ya prefixado por aquí dos
veces y acabar con `passage: passage: ...`, que el modelo nunca vio así y que degrada
el embedding en silencio.

El modelo se carga con retardo, al primer uso, no al importar el módulo. Importar
esto cuesta milisegundos; descargar los pesos son varios cientos de MB, y los tests
importan este módulo sin necesitar el modelo.

Los vectores salen normalizados (`normalize_embeddings=True`). Es lo que hace útil la
distancia coseno de ChromaDB, y con los vectores ya normalizados el producto escalar
es la similitud, sin que cada consulta tenga que renormalizar por su cuenta.

Se comprobó en la ficha del modelo: dimensión 384, `max_seq_length` 512
(`do_lower_case` false), licencia MIT, español entre los idiomas que declara. No hace
falta fijar la dimensión a mano porque se le pregunta al modelo cargado; el 384 queda
aquí solo como referencia.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from app.config import settings

LOGGER = logging.getLogger(__name__)

# Lado del documento: lo que se guarda en el índice y contra lo que se compara.
PREFIJO_DOCUMENTO = "passage: "

# Lado de la consulta: lo que escribe quien busca.
PREFIJO_CONSULTA = "query: "

DIMENSION_ESPERADA = 384
MAX_SEQ_LENGTH_ESPERADO = 512


def _con_prefijo(texto: str, prefijo: str) -> str:
    """Pone `prefijo` delante de `texto` salvo que ya esté puesto."""
    if texto.startswith(prefijo):
        return texto
    return prefijo + texto


def documento_para_indexar(texto: str) -> str:
    """Devuelve `texto` listo para incrustar como documento (`passage: `)."""
    return _con_prefijo(texto, PREFIJO_DOCUMENTO)


def consulta_para_buscar(texto: str) -> str:
    """Devuelve `texto` listo para incrustar como consulta (`query: `)."""
    return _con_prefijo(texto, PREFIJO_CONSULTA)


class Embedder:
    """Envuelve un `SentenceTransformer` y aplica los prefijos por el lado correcto.

    Se pasan dos métodos en vez de uno con un parámetro de "modo" porque el prefijo es
    justamente la asimetría que no conviene dejar como opción: quien llama tiene que
    decir si está indexando o buscando, y no poder equivocarse de otra forma.
    """

    def __init__(self, modelo: Any, nombre: str) -> None:
        self._modelo = modelo
        self.nombre = nombre

        # sentence-transformers 6 renombró get_sentence_embedding_dimension a
        # get_embedding_dimension y el viejo avisa con FutureWarning. Se pregunta por el
        # nuevo y se cae al viejo, que es lo que hay en versiones anteriores.
        dimension = getattr(modelo, "get_embedding_dimension", None)
        if dimension is None:
            dimension = modelo.get_sentence_embedding_dimension
        self.dimension = int(dimension())
        self.max_seq_length = int(modelo.max_seq_length)
        self.tokenizador = getattr(modelo, "tokenizer", None)

        LOGGER.info(
            "Modelo %s listo: dimensión=%d max_seq_length=%d",
            self.nombre,
            self.dimension,
            self.max_seq_length,
        )
        if self.dimension != DIMENSION_ESPERADA:
            LOGGER.warning(
                "La dimensión es %d y se esperaba %d; revisa el modelo de settings",
                self.dimension,
                DIMENSION_ESPERADA,
            )

    @property
    def modelo(self) -> Any:
        """El `SentenceTransformer` subyacente, para el tokenizer y la longitud."""
        return self._modelo

    def incrustar_documentos(
        self, textos: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Incrusta documentos ya con `passage: ` y devuelve vectores normalizados."""
        prepared = [documento_para_indexar(t) for t in textos]
        return self._encode(prepared, batch_size)

    def incrustar_consultas(self, consultas: list[str]) -> list[list[float]]:
        """Incrusta consultas ya con `query: ` y devuelve vectores normalizados."""
        prepared = [consulta_para_buscar(c) for c in consultas]
        return self._encode(prepared, 32)

    def _encode(self, textos: list[str], batch_size: int) -> list[list[float]]:
        if not textos:
            return []
        vectores = self._modelo.encode(
            textos,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectores.tolist()


@functools.lru_cache(maxsize=4)
def cargar_embedder(nombre: str | None = None) -> Embedder:
    """Carga el modelo de embeddings. Retardado a propósito: pesa.

    Importar el módulo no descarga nada; el peso se baja la primera vez que se
    llama. El modelo sale de `settings.embedding_model` si no se pasa otro.

    **Va en caché, y sin esto el endpoint es inusable.** Medido: `SentenceTransformer(...)`
    tarda unos 12 s porque relee los pesos de disco y reconstruye el tokenizer en cada
    llamada, o sea 12 s por petición de `/recommend`. Antes de cablear el pipeline real
    esto no se notaba, porque `/recommend` contestaba con el mock y no llegaba a buscar.

    El coste de la caché es el mismo que ya asumió `vector_store.obtener_coleccion`:
    **el estado es por proceso**. Con `--workers 4` habría cuatro copias de ~700 MB de
    pesos en memoria. Hoy se corre un solo worker, y para escalar hay que sustituir esto
    por algo compartido antes, no después.

    `maxsize=4` y no `1` porque el nombre es parte de la clave: dos índices construidos
    con modelos distintos necesitan los dos cargados, y es un caso real de los tests.
    """
    from sentence_transformers import SentenceTransformer

    objetivo = nombre or settings.embedding_model
    LOGGER.info("Cargando modelo de embeddings %s", objetivo)
    # El nombre se pasa aparte porque `str(modelo)` devuelve el repr entero de la
    # cadena de módulos, y eso no es un nombre: es un volcado de doscientas líneas que
    # no sirve ni para comparar con otro índice ni para volver a cargarlo.
    return Embedder(SentenceTransformer(objetivo), objetivo)


def limpiar_cache_embedder() -> None:
    """Olvida los modelos cargados. Para tests que necesitan recargar con otro nombre."""
    cargar_embedder.cache_clear()