"""Embeddings con ONNX INT8 y tokenizador directo de sentencepiece.

Este módulo sustituye a `SentenceTransformer` en el camino caliente por una razón
de presupuesto, medida y no supuesta. La RAM de proceso se descompone así
(VmRSS, una consulta en servicio, proceso fresco):

    python + numpy + onnxruntime + huggingface-hub     ~72 MB
    sesión ONNX INT8 (`model_quantized.onnx`)         +138 MB
    tokenizer `tokenizers` de Hugging Face            +306 MB
    tokenizer sentencepiece (este módulo)              +57 MB
    --------------------------------------------------------
    servicio con sentencepiece                        ~250 MB
    (252 medidos de proceso fresco con el modelo listo)

El tokenizer era el problema, no los pesos: `Tokenizer.from_file` cuesta unos
1,1 KB por entrada de vocabulario (250 002 entradas = 306 MB), esa memoria no
se libera ni con `del` + `gc`, y es instancia por instancia. Con `tokenizers` el
servicio se quedaba en ~498 MB, pegado al techo de Railway Free (500 MB) antes
de contar nada más; con sentencepiece el suelo ronda los ~250 MB. PyTorch, que
es lo que construyó el índice, llegaba a ~856 MB tras codificar. La arena de
CPU de onnxruntime no es lo que decide: la matriz de configuraciones (arena ON/OFF,
memory pattern, opciones de grafo) midió los mismos 496/498 MB en todas.

El modelo ONNX es la exportación INT8 de `Xenova/multilingual-e5-small` de los
mismos pesos de `intfloat/multilingual-e5-small`, y el emparejamiento es a
palabra: si `settings.embedding_model` cambia, hay que cambiar `REPO_ONNX` y
revalidar, y por eso `ModeloOnnx` se niega a cargar cualquier otro nombre en vez
de acoplar mal y dar vectores sin avisar.

### Tres hechos que costaron descubrir, y que este módulo fija

1. **El vocabulario no comparte IDs.** El modelo sentencepiece tiene
   `<unk>=0, <s>=1, </s>=2` y las piezas comunes desde la 3; el tokenizer HF
   tiene `<s>=0, <pad>=1, </s>=2, <unk>=3` y las mismas piezas comunes
   desplazadas **+1**. El mapeo es `{0: 3, 1: 0, 2: 2}` y `id + 1` para el resto,
   verificado sobre las 249 997 piezas comunes: 0 fallos. Sin mapear, cada
   vector sale desplazado y la búsqueda da resultados basura sin ningún error.

2. **No hay que replicar la normalización, y no hay que añadir el `▁` final.**
   El normalizador Precompiled del tokenizer HF fue exportado desde este mismo
   modelo sentencepiece, así que la normalización intermedia ya la hace el
   propio modelo. El `▁` final es la trampa: `tokenizers.Tokenizer` en crudo
   emite un `▁` (id 6) cuando el texto termina en espacio en blanco, pero
   sentence-transformers —el que construyó el índice— **no lo emite**, y
   sentencepiece tampoco. Sobre los 205 textos de la Fase 0, sentencepiece
   directo iguala al tokenizer del índice **205/205**; con una regla que
   añadiera el `▁` final, 204/205 (rompe el texto del índice que termina en
   U+200E). El objetivo es el índice, no el tokenizador en crudo.

3. **Divergencias conocidas y cuantificadas.** Sobre el catálogo completo,
   sentencepiece y el tokenizer del índice difieren en **2 de 9397 docs** y en
   **0 de 5 consultas**: son desempates de igual probabilidad del modelo
   Unigram que el Viterbi en Rust de `tokenizers` resuelve distinto que el C++
   de sentencepiece (el reparto de `e....` y el de `Idomu`). En adversarial, el
   único carácter de normalización que difiere es NEL (U+0085), ausente del
   catálogo. El impacto es un token segmentado al revés, del mismo orden que el
   ruido de cuantización INT8 ya aceptado en la Fase 0 (coseno medio 0,9956
   contra PyTorch; 6/30 tarjetas del top-6 cambiaban, y el índice se dejó como
   estaba: sirve).

### Lo que el grafo ONNX no trae

La ruta de sentence-transformers es `Transformer -> Pooling(MEAN) -> Normalize`.
La exportación contiene solo la primera parte, así que aquí van las otras dos:
mean pooling respetando `attention_mask` y L2 al final. Sin la normalización,
las distancias de ChromaDB dejarían de ser comparables con las del índice, que
está hecho con vectores normalizados.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import sentencepiece as spm
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger(__name__)

# El único emparejamiento validado: pesos intfloat con su exportación INT8 de
# Xenova. No se deriva del nombre porque Xenova no exporta todos los modelos.
MODELO_CON_PAREJA = "intfloat/multilingual-e5-small"
REPO_ONNX = "Xenova/multilingual-e5-small"
ARCHIVO_ONNX = "onnx/model_quantized.onnx"
ARCHIVO_SPM = "sentencepiece.bpe.model"

MAX_SEQ_LENGTH = 512

# <pad> en el espacio de IDs de Hugging Face. El modelo sentencepiece no tiene
# pad: solo se usa en posiciones con attention_mask=0, que el pooling ignora.
ID_PAD_HF = 1

# Del espacio de IDs de sentencepiece al de Hugging Face (hecho 1 de la
# cabecera). Regular: id + 1.
ESPECIALES_SP = {0: 3, 1: 0, 2: 2}


def _mapear_ids(ids: Sequence[int]) -> list[int]:
    """Del espacio de IDs de sentencepiece al de Hugging Face (hecho 1)."""
    return [ESPECIALES_SP.get(i, i + 1) for i in ids]


class TokenizadorE5:
    """sentencepiece con los IDs de Hugging Face y la forma de llamada de `transformers`.

    `scripts/build_index.py` mide el truncamiento llamando al tokenizador como
    lo haría `transformers`
    (`tokenizador(textos, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]`),
    y con esta firma sigue funcionando sin tocar el script. Si se le pasa un
    `str` suelto en vez de una lista, falla enseguida: `for x in "hola"` recorre
    caracteres y devolvería algo plausible y completamente falso.
    """

    def __init__(self, ruta_modelo: str) -> None:
        self._sp = spm.SentencePieceProcessor(model_file=ruta_modelo)

    def ids_de(self, texto: str, truncar_en: int | None = None) -> list[int]:
        """IDs de `texto` con `<s>` y `</s>`, en el espacio de Hugging Face.

        `truncar_en` recorta el contenido dejando sitio para los dos especiales
        (510 de 512), que es lo que hace `transformers` con
        `truncation=True, max_length=512`: comprobado contra el tokenizer del
        índice, idénticos hasta en el texto más largo del catálogo, con `</s>`
        conservado.
        """
        contenido = _mapear_ids(self._sp.EncodeAsIds(texto))
        if truncar_en is not None:
            contenido = contenido[: max(truncar_en - 2, 0)]
        return [0] + contenido + [2]

    def __call__(
        self,
        textos: Sequence[str],
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        verbose: bool = False,  # existe porque así lo llama build_index.medir_truncamiento
    ) -> dict[str, list[list[int]]]:
        del verbose  # transformers avisa por pantalla al truncar; aquí no hay nada que avisar
        if isinstance(textos, str):
            raise TypeError("pasa una lista de textos, no una cadena")
        limite = max_length or MAX_SEQ_LENGTH
        salida: list[list[int]] = []
        for texto in textos:
            if add_special_tokens:
                salida.append(self.ids_de(texto, truncar_en=limite if truncation else None))
            else:
                contenido = _mapear_ids(self._sp.EncodeAsIds(texto))
                salida.append(contenido[:limite] if truncation else contenido)
        return {"input_ids": salida}


class ModeloOnnx:
    """El modelo de embeddings: sentencepiece + sesión ONNX INT8, sin torch.

    Implementa la interfaz que lee `Embedder`: `encode`, dimensión,
    `max_seq_length` y `tokenizer` (de ahí sale `Embedder.tokenizador`, con el
    que `build_index` mide el truncamiento).
    """

    def __init__(self, nombre: str) -> None:
        if nombre != MODELO_CON_PAREJA:
            raise ValueError(
                f"el emparejamiento ONNX solo está validado para {MODELO_CON_PAREJA!r}, "
                f"no para {nombre!r}: exporta el nuevo modelo a ONNX, actualiza "
                f"REPO_ONNX y revalida vectores contra el índice"
            )

        import onnxruntime as ort  # pesa importarlo; solo hace falta al cargar el modelo

        ruta_spm = hf_hub_download(nombre, ARCHIVO_SPM)
        ruta_onnx = hf_hub_download(REPO_ONNX, ARCHIVO_ONNX)

        self.max_seq_length = MAX_SEQ_LENGTH
        self.tokenizer = TokenizadorE5(ruta_spm)

        opciones = ort.SessionOptions()
        # La arena de CPU no cambia el piso de servicio (la matriz de
        # configuraciones midió los mismos 496/498 MB con ella ON y OFF), pero
        # sí agranda el pico de codificar lotes largos: +359 MB tras 200 docs
        # con ella ya apagada. En servicio solo se codifican consultas, así que
        # aquí no decide nada; se queda apagada para que construir el índice no
        # tenga un pico peor que el medido.
        opciones.enable_cpu_mem_arena = False
        self._sesion = ort.InferenceSession(
            ruta_onnx, opciones, providers=["CPUExecutionProvider"]
        )

        entradas = sorted(i.name for i in self._sesion.get_inputs())
        esperadas = sorted(["input_ids", "attention_mask", "token_type_ids"])
        if entradas != esperadas:
            raise ValueError(f"el ONNX trae entradas {entradas}, se esperaban {esperadas}")

        salidas = self._sesion.get_outputs()
        if len(salidas) != 1:
            raise ValueError(f"el ONNX trae {len(salidas)} salidas, se esperaba 1")
        self.dimension = int(salidas[0].shape[-1])

        LOGGER.info(
            "Modelo ONNX listo: dimensión=%d max_seq_length=%d",
            self.dimension,
            self.max_seq_length,
        )
        if self.dimension != 384:
            LOGGER.warning("La dimensión es %d y se esperaba 384", self.dimension)

    def get_embedding_dimension(self) -> int:
        """El nombre que `Embedder` busca (sentence-transformers 6 lo llamó así)."""
        return self.dimension

    def encode(
        self, textos: Sequence[str], batch_size: int = 32, **kwargs: Any
    ) -> np.ndarray:
        """Vectores L2-normalizados, con la misma interfaz que `SentenceTransformer.encode`.

        `kwargs` recibe y desestima `normalize_embeddings`, `convert_to_numpy` y
        `show_progress_bar`: son de sentence-transformers, y aquí normalizar y
        devolver numpy no son opción (sin normalizar, las distancias de
        ChromaDB no serían comparables con las del índice) ni hay progreso
        interno que enseñar (el progreso del índice va por trozos, en
        `build_index`).
        """
        del kwargs
        if not textos:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectores = []
        for inicio in range(0, len(textos), batch_size):
            lote = [
                self.tokenizer.ids_de(t, truncar_en=self.max_seq_length)
                for t in textos[inicio : inicio + batch_size]
            ]
            vectores.append(self._inferir(lote))
        return np.concatenate(vectores, axis=0)

    def _inferir(self, lote_ids: list[list[int]]) -> np.ndarray:
        """Un pase de la sesión: relleno al más largo del lote, pooling y L2.

        El eje de secuencia del ONNX es dinámico, así que rellenar al más largo
        del lote (y no siempre a 512) sale más barato. El relleno va con
        `ID_PAD_HF` y queda fuera de `attention_mask`, así que el mean pooling
        no lo cuenta.
        """
        n = len(lote_ids)
        largo = max(len(ids) for ids in lote_ids)
        entrada = np.full((n, largo), ID_PAD_HF, dtype=np.int64)
        mascara = np.zeros((n, largo), dtype=np.int64)
        for fila, ids in enumerate(lote_ids):
            entrada[fila, : len(ids)] = ids
            mascara[fila, : len(ids)] = 1

        feed: dict[str, np.ndarray] = {
            "input_ids": entrada,
            "attention_mask": mascara,
            # XLM-R no tiene token_type_ids; el grafo lo exige y espera ceros.
            "token_type_ids": np.zeros_like(entrada),
        }
        crudo = self._sesion.run(None, feed)[0]  # (n, largo, dim)

        m = mascara[..., None].astype(np.float32)
        agrupado = (crudo * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        return agrupado / np.linalg.norm(agrupado, axis=1, keepdims=True)
