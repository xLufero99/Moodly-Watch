# Estado del proyecto

Última actualización: **2 de octubre de 2026**.

Este fichero es la bitácora: qué hay hecho, qué está verificado y qué falta. El *por qué*
de cada decisión de implementación está en los docstrings del propio código, que es donde
pertenece; aquí solo está lo que no se puede deducir leyendo el código.

---

## Qué es el proyecto

Recomenda películas, series y anime a partir de una frase libre del usuario ("quiero algo
corto y raro"), usando búsqueda semántica sobre sinopsis y un LLM que explica el porqué
de cada recomendación.

Dos paquetes independientes, sin orquestación en la raíz: `backend/` (Python/FastAPI, uv)
y `frontend/` (React 19 + Vite 8, npm).

---

## Fases

| Fase | Qué es | Estado | Commit |
|---|---|---|---|
| Andamiaje | API y frontend mínimos | Hecha | `87c9ecb`, `63a46f0`, `dc0bb6a` |
| Endpoint falso | `POST /recommend` con schemas y mock | Hecha | `b2175bf` |
| Datos TMDB | Descarga con reintentos y reanudación | Hecha | `3658a03` |
| Datos MAL | API oficial, con Jikan de respaldo | Hecha | `e8146aa`, `2050e4d` |
| Catálogo | Unifica TMDB + MAL en un parquet | Hecha | `4443db2` |
| Análisis | Notebook de EDA | Hecha | `d1fd9b6` |
| Refactor runtime | Parte `runtime` en total y por episodio | Hecha | `ec423ec` |
| **Índice semántico** | **Embeddings + ChromaDB** | **Hecha, sin commitear** | — |
| **Índice completo** | **Construir los 9397 vectores** | **Hecha, sin commitear** | — |
| **Recommendidor real** | **`/recommend` con búsqueda + LLM** | **Pendiente** | — |

---

## El catálogo

`backend/data/processed/catalog.parquet`, una fila por título:

- **9397 filas × 23 columnas**.
- `media_type`: 4925 películas, 3467 anime, 1005 series.
- `source`: 5930 de TMDB, 3467 de MAL.
- `vote_count` va de 500 a 3 115 304. Es una escala mezclada: la mediana de MAL (37 765)
  es casi 19 veces la de TMDB (2018).
- 9 filas sin duración en ninguna de sus dos columnas.
- Sin sinopsis nulas: los scripts de descarga rellenaron los huecos con el original en
  inglés.
- No se versiona en git (`data/` está en `.gitignore`); se regenera con
  `scripts/build_catalog.py`.

---

## Índice semántico (lo reciente)

### Ficheros

| Fichero | Qué hace |
|---|---|
| `app/services/embedder.py` | Prefijos `passage:` / `query:` y carga del modelo. Compartido por el indexador y el buscador para que no diverjan |
| `scripts/build_index.py` | Construye el índice: percentiles, metadatos, truncamiento, escritura |
| `scripts/search_demo.py` | Prueba de humo por consola |
| `tests/test_build_index.py` | 33 tests con un embedder falso, sin descargar el modelo |

### Números medidos, no estimados

Cifras del catálogo completo, que son las que valen:

| | |
|---|---|
| Modelo | `intfloat/multilingual-e5-small`, dimensión **384** |
| `max_seq_length` | **512** |
| Vectores | **9397**, uno por fila del catálogo |
| Ritmo en CPU | **~3,2 docs/s** (3,1-3,3 según el tramo) |
| Tiempo del índice completo | **52 minutos** (3099 s de reloj) |
| Texto que pasa de 512 tokens | **0,06 %** (6 de 9397), máximo observado **839 tokens** |
| Tamaño del índice | **99 MB** en disco (80 MB de `chroma.sqlite3`) |
| `torch` en disco | 737 MB (solo CPU) |

Torch ya usa los 4 núcleos físicos de la máquina (8 con hyperthreading), que es lo
correcto: es un Ryzen 5 3500U.

**La cifra de 4,3 docs/s que se midió antes venía del muestreo de 200, y sobreestimaba
el ritmo.** El índice completo va a ~3,2, unos 25 % más lento, porque las sinopsis largas
del catálogo entero gastan más tokens que las de la muestra. Extrapolar desde una muestra
pequeña con este modelo no vale: el tiempo depende de la longitud del texto, no solo del
número de documentos. Consecuencia práctica: un `--limit` sirve para comprobar que todo
funciona, pero su ritmo no sirve para prever el del catálogo completo.

### Decisiones y su motivo

El razonamiento largo de cada una está en el docstring del fichero correspondiente.

- **Prefijos obligatorios.** La ficha del modelo dice que se entrenó así y que sin ellos
  el rendimiento baja. La búsqueda es asimétrica: la consulta lleva `query: ` y el
  documento `passage: `.
- **El prefijo va al modelo, no al índice.** Lo que se guarda en el campo `document` de
  ChromaDB es el `embed_text` limpio. Si el `passage: ` acabara dentro, una búsqueda por
  texto plano dejaría de encontrar nada.
- **Percentiles dentro de cada `source`.** Un percentil global mediría sobre todo "de
  qué fuente es el título" en vez de "qué popular es", porque las dos escalas de votos no
  son comparables.
- **Sin logaritmo en `popularity_pct`.** El percentil va por rangos y el logaritmo es
  monótono, así que `log10(vote_count)` da exactamente el mismo número. Se comprobó sobre
  el catálogo entero y en las dos fuentes. El log solo importaría si esto deixara de ser
  un percentil y fuera un umbral absoluto.
- **Empates con `method="average"`**, que además es el valor por defecto de pandas. Es el
  único que mantiene la propiedad de que dos títulos con el mismo `vote_count` reciben el
  mismo `popularity_pct`; con `first` se reparten en 0.5 y 0.75.
- **Los nulos se omiten, no se sustituyen por centinelas.** Un `-1` en `year` mentiría;
  la ausencia de la clave significa "no lo sé", que es la verdad.
- **`genres` va como texto unido por `|`,** no como lista. Ver más abajo por qué.
- **Coseno con vectores normalizados.** Así la distancia es `1 - similitud` y las
  consultas no necesitan renormalizar.
- **Reconstrucción por defecto.** Se borra la colección y se rehace. Añadir a una
  existente dejaría vectores viejos conviviendo con los nuevos.
- **`--limit` reparte proporcionalmente,** no a mitades. Las series son el 10,7 % del
  catálogo; un reparto equitativo inflaría el tiempo estimado y rompería la extrapolación.

### Hechos que costaron tiempo descubrir

Los cuatro son de ejecutar, no de leer documentación:

1. **ChromaDB rechaza `None` en los metadatos, aunque su validador de Python lo admita.**
   El `add` falla con `TypeError: Cannot convert Python object to MetadataValue` porque la
   comprobación que manda está en la capa de Rust. Confundir los dos validadores lleva a
   un `add` que revienta a mitad de la carga.

2. **`np.int64` no es subclase de `int` en Python.** ChromaDB valida los tipos con
   `isinstance`, así que el `year` y el `vote_count` de una fila se rechazan tal cual si
   no se bajan a tipos nativos antes.

3. **Las listas en metadatos también fallan,** y de dos maneras: la lista vacía se rechaza
   siempre, y una lista con `None` dentro también. Hay 58 géneros de MAL sin equivalente
   canónico, así que hay filas con la lista vacía de verdad.

4. **FastAPI 0.142 anida el router como `_IncludedRouter`** en vez de aplanar las rutas en
   `app.routes`. Recorrer `app.routes` para ver qué endpoints hay ya no las enseña.

---

## Estado del frontend ↔ backend

**Ya están cableados.** Esto se corrigió respecto a lo que decía antes este mismo
documento:

- `frontend/vite.config.js` tiene `server.proxy` para `/recommend` y `/health` hacia
  `http://localhost:8000`.
- `frontend/src/api/client.js:3` tiene `USE_MOCK = false`.

La app funciona hoy de punta a punta, pero **devuelve datos mock**: lo que atiende
`POST /recommend` sigue siendo `app/services/mock_recommender.py`.

El contrato ya está declarado por `frontend/src/api/mockData.js`: petición
`{ text, media_types, liked_ids }`, respuesta `{ results: [...] }`, y cada item con
`{ id, title, media_type, year, genres[], poster_url, score, explanation }`.

---

## Servicio de búsqueda

`app/services/search.py` traduce una frase a recomendaciones con la forma del contrato.
No es el endpoint: no sabe de HTTP y no genera la `explanation`.

| Fichero | Qué hace |
|---|---|
| `app/services/vector_store.py` | Abre el índice y avisa si el modelo no es el del indexado. Estado global cacheado con `lru_cache` |
| `app/services/search.py` | Embebe la consulta, filtra y devuelve `ResultadoBusqueda` |
| `scripts/search_demo.py` | CLI fina sobre el servicio. Ya no abre el índice por su cuenta |

El singleton de `vector_store` es **por proceso**: con `--workers > 1` cada worker
cargaría su copia del modelo. Hoy se corre un solo worker y no es un problema, pero es
decisión consciente, no descuido.

### El `score` es relativo, no una confianza

Las distancias de este modelo están comprimidas. Medido en la misma consulta: el top-1
daba `d=0.1368` y el top-5 `d=0.1435`, o sea similitudes **0.8632** y **0.8565**. Como
`ResultCard.jsx:46` hace `Math.round(score * 100)`, con el score crudo los cinco
resultados salían a **"86%"**: cero información.

Por eso el score va reescalado con min-max sobre lo devuelto, al rango **[0,5 – 1,0]**.
El primer resultado marca 100% en todas las consultas y eso es el techo del reescalado,
no una coincidencia perfecta. La `distance` cruda se conserva en `ResultadoBusqueda` y se
registra en el log, pero no viaja en la respuesta de la API.

### Limitación conocida: las negaciones

**La búsqueda vectorial no entiende negaciones.** Medido con `"no quiero anime, algo real
y corto"`: salen **3 anime de 5**, y el primero es *Hokuto no Ken Movie*. También sale
*"No soy un robot"* en el puesto 3, señal de que el modelo acerca el token "no" a
títulos que empiezan por "No".

**La limpieza de negaciones con regex se descartó a propósito.** Aguanta `"no quiero
anime"` y se rompe con `"fuera de anime"`, `"algo que no sea anime"` o `"no me apetece el
anime"`. Un arreglo que funciona en el caso que pruebas y falla en el siguiente es peor
que ninguno, porque da confianza falsa.

Lo que sí funciona hoy es `media_types` como filtro duro: con `--sin-anime` la misma
consulta deja de devolver anime. Y no es casualidad, es que el filtro va a ChromaDB en el
`where`, así que actúa antes de elegir el top y no después. Con `"no quiero anime"` sin
el filtro salen 3 de 5; con `--sin-anime`, ninguno.

### Lo que falta

En orden, con lo que depende de lo anterior:

1. **Groq, y con él el parser de intención.** `groq_api_key` está en `settings` y no lo
   usa nada en todo el repo. El parser es lo que convierte el texto libre en filtros
   exactos: `{"query": str, "media_types": [...], "max_runtime": int | None}`. Es la
   pieza que resuelve las negaciones de verdad, porque un `where` sobre metadatos es
   exacto y un vector cercano no lo es nunca. Por eso va antes que cablear el router.
2. **La `explanation` que genera Groq** para cada resultado.
3. **Que `/recommend` use `buscar_como_contrato`** en lugar de
   `get_mock_recommendations`.

`explanation` ya es opcional en el schema (`str | None = None`) para que la búsqueda viva
sin Groq. `ResultCard.jsx:76` renderiza `{explanation}` sin condición, así que un `null`
pinta un párrafo vacío en vez de romper. Cuando Groq esté, se vuelve a cerrar.

`/recommend` sigue con el mock a propósito: cablearlo ahora obligaría a inventar la
`explanation`, y un texto con formato de recomendación que el sistema no generó es
indistinguible de uno real.

---

## Comandos

```bash
# Todo el backend necesita cwd=backend/
cd backend

uv run pytest                    # 291 tests (51 son del buscador)
uv run ruff check .              # el único check configurado

# Reconstruir el catálogo desde los JSON crudos
uv run python -m scripts.build_catalog

# Índice de prueba: 200 docs repartidos, ~1 minuto
uv run python -m scripts.build_index --limit 200 --dir-indice data/index/prueba

# Índice completo: ~52 minutos en CPU
uv run python -m scripts.build_index

# Buscar en el índice
uv run python -m scripts.search_demo "un hombre que viaja a otros planetas y se vuelve loco"
uv run python -m scripts.search_demo "no quiero anime" --sin-anime
```

Servidor: `uv run uvicorn app.main:app --reload` desde `backend/`.

### Cómo lanzar el índice completo sin que muera

Son 52 minutos, y hay dos formas de perderlos por el camino, las dos aprendidas a la
mala:

```bash
cd backend
setsid nohup uv run python -m scripts.build_index > data/build_index.log 2>&1 < /dev/null & disown
```

- **`setsid`**: sin él, si la sesión que lanzó el comando se corta, el proceso se va con
  ella. Ya pasó una vez y se perdieron 45 minutos.
- **Log en `data/`, no en `/tmp`**: `/tmp` es tmpfs, así que un reinicio de la máquina lo
  borra junto con el proceso. `data/` está en disco y en `.gitignore`.

Perder el run **no deja el índice a medias**: `build_index.py` calcula todos los embeddings
antes de abrir ChromaDB, así que si se corta, simplemente no hay índice y hay que relanzar
limpio. Eso es intencional y evita tener que reparar un estado a medias.