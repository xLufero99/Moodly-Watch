# Estado del proyecto

Última actualización: **3 de octubre de 2026**.

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
| Índice semántico | Embeddings + ChromaDB | Hecha | `1f0a7b3` |
| Índice completo | Construir los 9397 vectores | Hecha, sin commitear | — |
| Filtros con Groq | Parser de intención a filtros exactos | Hecha | `1f0a7b3` |
| **Recommendidor real** | **`/recommend` con parser + búsqueda + explainer** | **Hecha, sin commitear** | — |
| **Fase A: limpieza ONNX** | **Quita torch y sentence-transformers del proyecto** | **Hecha, sin commitear** | — |

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
| `torch` en disco | 737 MB (solo CPU) — **ya no está en el proyecto**, ver la Fase A |

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

## Fase A: quitar torch y sentence-transformers

El swap del embedder a ONNX INT8 + sentencepiece (commit `7203656`) dejó a torch y
sentence-transformers sin ningún importador: el camino caliente, `build_index` y
`search_demo` pasan todos por `app/services/modelo_onnx.py`. Ahora están fuera de
`pyproject.toml`, junto con `transformers`, que solo llegaba por sentence-transformers y
solo lo usaban los tests.

**El ahorro son ~1 GB, no los 737 MB de torch:** `backend/.venv` baja de **1,75 GB a
725 MB**. Los 15 paquetes que se fueron: torch (739 MB), scipy (94), transformers (57),
sympy (41), scikit-learn, sentence-transformers, networkx, regex, safetensors, joblib,
narwhals, cloudpickle, setuptools, threadpoolctl y mpmath. `chromadb` **no** depende de
torch (su dependencia pesada es `onnxruntime`, que se queda), así que el ahorro era real y
no un cambio de declaración: `uv tree --package torch --invert` sale vacío.

**La referencia de los tests son dos fixtures**, no las librerías:

| Fichero | Qué guarda |
|---|---|
| `tests/fixtures/vectores_dorados.npz` | Los 4 vectores que daba sentence-transformers 6.1.0, con sus textos y el nombre del modelo dentro |
| `tests/fixtures/tokenizer_ids.json` | Los 21 juegos de IDs que daba `transformers.AutoTokenizer`, incluidos los truncados a 512 |

Se generaron el 3 de octubre de 2026 con las librerías todavía instaladas y **antes** del
`uv sync` que las quitó; el cómo está en el docstring de `tests/test_modelo_onnx.py`. El
guardia de vectores va contra el npz con **umbral 0,98** y embebiendo un texto por llamada:
en un solo lote el texto de más de 512 tokens cae a 0,9789 por compartir escala con el
relleno, contra 0,9932 estando solo.

**La RAM no se mueve**: con el embedder cargado y el índice de ChromaDB abierto, 296 MB
medidos en un proceso propio; el servicio entero sigue en torno a los 351 MB de antes, o sea
cómodo en los 500 MB del free tier de Railway.

### Lo que hay que tocar en Railway

No hay Dockerfile ni nixpacks.toml en el repo, así que la imagen la decide Nixpacks y
**`uv sync` se lleva también el grupo `dev`** (pytest, ruff, matplotlib, ipykernel,
nbconvert). El ahorro de la Fase A llega igual —torch va en `dependencies`—, pero para no
pagar el dev group hay que poner en Railway el install command:

```bash
uv sync --no-dev --frozen
```

O escribir el Dockerfile de la Fase B, que es cuando toque decidir la imagen de verdad.

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

## Filtros y explicaciones con Groq

`app/services/filter_parser.py` traduce el texto libre a filtros exactos, y
`app/services/groq_client.py` es el cliente. No es un modelo de intención: es un
extractor de filtros, y por eso el dataclass es `FiltrosConsulta` y el fichero
`filter_parser.py`.

Salida: `{query, media_types, max_runtime_total, max_runtime_episode}`, con
`json_schema` y `strict: true` sobre `openai/gpt-oss-20b`. Los tres campos van en
`required` con `additionalProperties: false`, y las duradas son unión con `null` porque
Groq no admite opcionales de otra forma. Un `null` significa "no pidió duración", no
"faltó el campo".

**El modelo no es negociable.** Structured outputs con `strict: true` solo funcionan en
`openai/gpt-oss-20b`, `openai/gpt-oss-120b` y `qwen/qwen3.8-27b`. El default que había en
`config.py` era `llama-3.3-70b-versatile`, que **no** los soporta: devolvía 400 y el
parser se degradaba siempre. Ahora, si el modelo configurado no sirve, hay aviso
obligatorio en el log antes de degradar, porque resultados sin filtros sin explicación son
peores que un error.

### Dos reglas semánticas que van en el prompt

Las dos existen porque los datos no son homogéneos. Medido sobre `catalog.parquet`:

| media_type | `runtime_total` | `runtime_episode` |
|---|---|---|
| movie | 100 % | 0 % |
| tv | **0 %** | 100 % |
| anime | 22 % | 78 % |

**1. `max_runtime_total` fuerza `media_types: ["movie"]`.** `runtime_total` no existe para
ninguna serie, así que prometer ese filtro con series devolvería vacío y parecería un
fallo. Tampoco se acepta `["movie", "anime"]`: en anime el dato está solo en el 22 %, o
sea que descartaría títulos que sí duran lo pedido.

**2. "series de menos de 2 horas" es duración de episodio, no total.** Una serie de 8
capítulos de 45 minutos no dura 2 horas, así que el filtro total sería imposible de
satisfacer. Va a `max_runtime_episode`, que es la lectura útil. Comprobado contra el
modelo real: `"series de menos de 2 horas"` da `{"query": "", "media_types": ["tv"],
"max_runtime_episode": 120}`.

**El buscador no lleva lógica por tipo.** Solo aplica el `where` que le llega. La decisión
de qué es coherente está en el prompt, una vez.

### Degradación

Si Groq falla (sin key, 429, timeout, JSON inválido), se busca el texto crudo **sin
filtros** y se marca `degradado=True` con el motivo. Nunca se inventan filtros.

`degradado` significa **fallo de infraestructura**, no "el modelo no pidió nada". Que
Groq conteste `"media_types": []` y las duradas a `null` es una respuesta legítima con
`degradado=False`.

Comprobado contra el modelo real, los cinco casos:

| Texto | query | tipos | total | episodio |
|---|---|---|---|---|
| no quiero anime, algo real y corto | `algo real` | `[movie]` | 120 | — |
| películas de menos de 90 minutos sobre Submission | `Submission` | `[movie]` | 90 | — |
| series de menos de 2 horas | `""` | `[tv]` | — | 120 |
| series con capítulos cortos de terror | `terror` | `[tv]` | — | 25 |
| algo triste y lento para un domingo | `algo triste y lento` | `[]` | — | — |

El caso 3 devuelve `query` vacía a propósito: la persona solo dijo filtros, no tema. Eso
**no** es degradación. Se busca solo por metadatos. Degradar ahí tiraría los filtros que el
modelo sí entendió, que es justo lo contrario de lo que se quiere.

### El filtro de duración en ChromaDB

ChromaDB admite **un solo operador por `where`**, así que con tipo más duración hace
falta `$and`:

```
ValueError: Expected where to have exactly one operator,
got {'media_type': {'$in': [...]}, 'runtime_total': {'$lte': 120}}
```

Comprobado contra el índice completo que `$and` sí funciona:

```
movie + runtime_total <= 120  →  3 de 9397
tv    + runtime_episode <= 25 →  3 de 9397
anime + runtime_episode <= 25 →  3 de 9397
```

### Lo que falta de esta fase

Nada. El parser, el explainer y el cableado de `/recommend` están hechos y verificados
contra el modelo de verdad; lo que queda abierto es de otros módulos:

- **El frontend** contra el endpoint real, en vez de contra el mock.

---

## El pipeline de `/recommend`

Tres servicios y un router que no sabe nada de ellos:

    texto -> filter_parser -> search -> explainer -> contrato

`app/services/recommendation.py` es el que los manda, para que `routes.py` sean tres
líneas y el orden se lea en un sitio. El orden importa: **buscar antes de explicar**,
porque el LLM justifica lo que el índice ya decidió y no elige títulos. Al revés se
gastaría la parte que el índice hace bien y barato.

`mock_recommender.py` **sigue en el repo, fuera del camino**. No se borra: sirve para
comparar lo que devuelve cada uno sin tener que apagar Groq.

### La regla de mezcla de `media_types`

Hay dos fuentes de tipos: el selector de la UI (`request.media_types`) y lo que el modelo
entendió del texto (`filtros.media_types`). Se combinan así:

| request | parser | resultado |
|---|---|---|
| vacío | vacío | sin filtro |
| con tipos | vacío | manda el request |
| vacío | con tipos | manda el parser |
| con tipos | con tipos | **intersección** |

La intersección es lo correcto porque las dos son restricciones. Si alguien marca "anime"
y escribe "no quiero anime", gana la frase: el selector no puede tapar una negación.

### `None` no es `[]`, y es la parte que no hay que tocar

El último caso acaba en **cero resultados**, no en "sin filtro": no hay ningún título que
valga cuando anime y "no anime" se contradicen. Devolver el catálogo entero ahí sería el
fallo más grave posible de este módulo, porque es justo el que el parser vino a arreglar.

La tentación es representar las dos cosas con `[]`. Con un solo `[]` no hay forma de
distinguir "nadie pidió nada" de "se contradijeron", así que el código acaba con
heurísticas y el bug vuelve. Por eso `mezclar_media_types` devuelve:

    None -> sin filtro: `where=None`, se busca en todo el catálogo
    []   -> cero resultados: no se llama a ChromaDB

El tipo lo dice solo y no hay bandera que mantener sincronizada. Está en
`tests/test_recommendation.py`, y hay un test que comprueba que el caso contradictorio
**no llega a llamar a ChromaDB**.

### Una sola llamada de Groq para los seis resultados

No una por resultado. Con seis, una búsqueda y media agotan los 30 RPM del free tier, y
además cada título se explicaría por su cuenta y dos parecidos acabarían con argumentos
distintos.

El array va en el mismo orden que los resultados, pero el emparejamiento es **por `id`**.
Con `minItems`/`maxItems` clavados al número de resultados el modelo no puede cambiar la
cantidad, pero **sí puede reordenar**, y con emparejamiento por posición la explicación de
una película caería en la tarjeta de otra. El texto resultante es plausible, que es lo que
lo hace peligroso. Hay un test con el array invertido.

### La sinopsis no está en los metadatos

Vive al final del `document` del índice, después de `Temas:`, y los metadatos no la
llevan. Sus 14 claves son genres, language, media_type, popularity, popularity_pct,
poster_url, rating, rating_pct, runtime_total, source, status, title, vote_count y year.

Por eso `search.buscar` pide `documents` en el `include`. Sin eso el explainer solo tendría
título y géneros, y con eso no se puede decir por qué algo encaja con un estado de ánimo.
Son seis documentos de ~590 caracteres, unos 880 tokens.

Hacer falta **dos** cortes para sacarla: `Temas: ` es una lista de palabras separadas por
comas que solo termina en `". "`. Con un solo `partition`, la sinopsis salía precedida de
"ambush, shotgun, machismo.".

### Degradación por ítem, no todo o nada

El parser degrada entero porque devuelve **un** objeto: o hay filtros o no hay. El explainer
devuelve seis textos independientes, así que un fallo se queda en el ítem que falló. Tirar
cinco explicaciones buenas porque la sexta vino mal es justo lo que aquí se evita.

Las cuatro salidas: todo Groq, Groq parcial (las que vinieron por `id` y plantilla en las
que falten), Groq inservible (error, sin key, JSON inválido, contenido vacío) y plantilla
en las seis. La plantilla nombra los géneros, porque son ciertos por construcción.

`content` vacío es un caso propio: gpt-oss-20b es un modelo de razonamiento y puede
gastarse el presupuesto antes de escribir. No es JSON inválido, es nada, así que
`max_tokens` está en 1400 y el aviso dice cuál de los dos fue.

---

## Latencia real del pipeline

Medido con el índice completo (9397 vectores), `uv run python`, CPU:

| paso | tiempo |
|---|---|
| cargar el embedder (una vez por proceso) | ~22 s |
| parser | ~1,1 s |
| búsqueda | **~0,13 s** |
| explainer (6 explicaciones) | **1,5 s - 24 s** |

### La del explainer es muy variable, y eso hay que saberlo

Seis llamadas medidas al mismo prompt: 1,86 / 1,47 / 1,72 / 9,18 / **24,40** / 13,53 s. La
mediana ronda los 2 s y la cola se va a más de 20.

O sea que **el riesgo que se anotaba como "5-10 s" es real y peor**, pero no de forma
uniforme. Importa porque:

- Si el frontend tiene un timeout por debajo de 10 s, habrá consultas que cortan de forma
  intermitente y parece un fallo del modelo cuando en realidad es latencia.
- Si no lo tiene, la espera es aceptable en la mediana y mala en la cola.

**Decisión con el dato del test lento:** se queda con `openai/gpt-oss-20b` y **no** se
recortan las sinopsis a 400 caracteres. El test lento pasa en ~1,5 s, así que la latencia
típica está bien y recortar sinopsis quitaría información para un problema que es de
latencia, no de tamaño. `gpt-oss-120b` se deja anotado como salida si la varianza llega a
molestar: el mismo prompt con más parámetros suele ser más lento, no menos.

---

## Dos fallos de rendimiento que costaron encontrar

Ninguno de los dos da error: los dos devuelven la respuesta correcta. Por eso se
comprobaron con medición y no leyendo el código.

### `cargar_embedder` no estaba cacheado: 8,8 s por petición

`SentenceTransformer(...)` tarda ~12-22 s porque relee los pesos de disco y reconstruye
el tokenizer. Como `cargar_embedder` no llevaba `lru_cache`, **cada `buscar` lo cargaba**:
8,8 s por petición, medidos, con el modelo ya en memoria del sistema.

No se notaba porque `/recommend` contestaba con el mock y nunca llegaba a buscar. El día
que se cableó el pipeline real pasó al camino caliente. Con el caché son **0,13 s**, una
cosa 68 veces más rápida.

`maxsize=4` y no `1` porque el nombre del modelo es parte de la clave: hay índices
construidos con modelos distintos y ambos tienen que estar disponibles. El coste es el
mismo que ya asume `vector_store.obtener_coleccion`: **el estado es por proceso**, y con
`--workers 4` habría cuatro copias de los pesos en memoria.

Fijado en `tests/test_embedder.py`, que **comprueba que los tests fallan si alguien quita
el `lru_cache`**, porque un test que pasa con el bug puesto no arregla nada.

### El cliente de Groq reintentaba solo: hasta 30 s

`groq_client.py` tenía escrito que no había reintentos y por qué ("un reintento en medio de
un 429 empeora justo lo que está fallando"). **Era falso**: el SDK viene con
`max_retries=2` y nadie lo puso a 0.

Con eso, `timeout=10` **no acota nada**: son tres intentos de hasta 10 s. Medido, una
llamada tardó 24,4 s. En el free tier además es peor que lento, porque reintentar ante un
429 consume cuota y hace el 429 más probable.

Ahora es `max_retries=0` explícito, con el `timeout` sí acotando. Fijado en
`tests/test_groq_client.py`.

---

## La cuota del free tier

Las cuentas que dan, con dos llamadas por búsqueda (parser + explainer):

| límite | valor | margen |
|---|---|---|
| peticiones por minuto | 30 | 15 búsquedas/min |
| tokens por minuto | 8.000 | ~4 búsquedas/min |

El de los tokens manda. Medido: el prompt del parser son ~600 tokens y el del explainer
~1.400 con las seis sinopsis, o sea ~2.000 por búsqueda y **unas 4 por minuto**. Para uso
personal es de sobra; para varios usuarios a la vez, no, y el sitio degrada a plantillas
sin romperse, que es justo el comportamiento buscado.

**La consecuencia incómoda: `pytest -m lento` entero no es fiable.** Los 8 tests lentos son
unos 16.000 tokens y la ventana es de 8.000 por minuto, así que en una sola pasada 3 fallan
con `RateLimitError` **sin que nada esté roto**: el parser degrada a texto crudo y el
explainer a plantillas, que es lo que deben hacer, pero los tests afirman
`degradado is False` y por eso fallan. Hay que correrlos por lotes con un minuto entre
uno, como está en los comandos de abajo. Verificado: por lotes pasan los 8.

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

1. **La `explanation` que genera Groq**, en un prompt aparte. El parser de filtros ya
   está; falta la otra mitad de la llamada al LLM.
2. **Que `/recommend` use el parser y `buscar_como_contrato`** en lugar de
   `get_mock_recommendations`. Con eso la negación está resuelta de verdad: sin filtros,
   `"no quiero anime"` devuelve anime; con los filtros del parser, no.

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

uv run pytest                    # 418 tests, 8 de ellos lentos fuera de la suite
uv run ruff check .              # el único check configurado

# Reconstruir el catálogo desde los JSON crudos
uv run python -m scripts.build_catalog

# Índice de prueba: 200 docs repartidos, ~1 minuto
uv run python -m scripts.build_index --limit 200 --dir-indice data/index/prueba

# Índice completo: ~52 minutos en CPU
uv run python -m scripts.build_index

# Tests que llaman a Groq de verdad (fuera de la suite normal).
# OJO: en una sola pasada fallan 3 de 8 por RateLimitError, no por un fallo del código.
# Los 8 son ~2.000 tokens cada uno y el free tier da 8.000 por minuto. Por lotes:
uv run pytest tests/test_filter_parser.py -m lento    # 3
sleep 70
uv run pytest tests/test_explainer.py -m lento        # 3
sleep 70
uv run pytest tests/test_recommendation.py -m lento   # 2

# Reparto actual: 54 search, 50 filter_parser, 37 explainer, 23 recommendation,
# 9 groq_client, 7 recommend (contrato), 4 embedder

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