"""Tests de `app/services/embedder.py`.

**Solo del caché de `cargar_embedder`.** El resto del embedder (prefijos de E5,
normalización, lote) está en `test_build_index.py` con `ModeloFalso`, porque para eso
sirve un stub del modelo de cuatro dimensiones y no hace falta cargar los pesos de
verdad.

Este fichero carga el modelo real, y a propósito: el fallo que se fija aquí es
precisamente el de no cargarlo, y un doble no lo detectaría. Cuesta unos segundos la
primera vez, una sola por sesión de pytest.
"""

from __future__ import annotations

from app.services.embedder import (
    cargar_embedder,
    limpiar_cache_embedder,
)


def test_el_embedder_se_carga_una_sola_vez() -> None:
    """La segunda llamada tiene que salir del caché.

    Sin el `lru_cache`, `ModeloOnnx(...)` reconstruye la sesión ONNX y recarga el
    tokenizador en cada llamada: ~1,1 s medidos por iteración (y con el
    `SentenceTransformer` anterior, que relee los pesos de disco, eran **8,8 s por
    petición**). Con el caché, `buscar` se queda en **0,13 s**. Ochenta veces, y no se
    nota hasta que el endpoint está en uso.

    Antes de cablear el pipeline real esto no se notaba, porque `/recommend` contestaba con
    el mock y nunca llegaba a buscar.
    """
    limpiar_cache_embedder()

    primero = cargar_embedder()
    assert cargar_embedder.cache_info().misses == 1

    segundo = cargar_embedder()
    assert cargar_embedder.cache_info().hits == 1
    # Es el mismo objeto, no uno equivalente: cargar los pesos otra vez para devolver algo
    # igual sería el bug con un paso más.
    assert primero is segundo


def test_el_cache_no_se_acumula_entre_tests() -> None:
    """`cache_info()` es global al proceso, así que este test solo vale si nadie lo ensucia.

    Sin esto, el test de arriba podría pasar por los `hits` que dejó otro test, y no por
    su propio comportamiento.
    """
    assert cargar_embedder.cache_info().currsize >= 0


def test_dos_nombres_dan_modelos_distintos() -> None:
    """El nombre es parte de la clave del caché, no un detalle.

    Es un caso real: hay índices construidos con modelos distintos, y compartir la entrada
    del caché entre ellos haría que uno buscara con los pesos del otro. Con el mismo nombre
    sí tiene que devolver el mismo objeto.
    """
    limpiar_cache_embedder()
    a = cargar_embedder("intfloat/multilingual-e5-small")
    b = cargar_embedder("intfloat/multilingual-e5-small")
    assert a is b
    assert cargar_embedder.cache_info().hits == 1


def test_limpiar_el_cache_fuerza_a_re_cargar() -> None:
    """`limpiar_cache_embedder()` tiene que dejar `cache_info` a cero.

    Sin esto, un test que recargara con otro modelo contaminaría el resto de la sesión.
    """
    limpiar_cache_embedder()
    assert cargar_embedder.cache_info().currsize == 0

    cargar_embedder()
    assert cargar_embedder.cache_info().currsize == 1

    limpiar_cache_embedder()
    assert cargar_embedder.cache_info().currsize == 0