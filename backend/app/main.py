import logging
import time
from contextlib import asynccontextmanager

from chromadb.errors import ChromaError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.services.embedder import cargar_embedder
from app.services.vector_store import IndiceNoDisponibleError, obtener_coleccion

# uvicorn solo pone handler a sus propios loggers, y un `LOGGER.info` de la raíz se
# pierde sin más. Sin esto, en los logs de Railway no se ve cuánto tardó el arranque.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Calienta el modelo y el índice en el proceso que va a servir.

    El calentamiento tiene que pasar **aquí** y no en un comando previo a uvicorn:
    un proceso aparte se termina, su memoria se libera y el servidor sigue frío, de
    forma que la primera petición real pagaría los ~1,5 s de sesión ONNX. Tampoco va
    en `/health`, porque ese latiría solo cuando alguien lo llame. Aquí se ejecuta
    una vez, antes de que FastAPI acepte conexiones.

    Tiene una segunda función: Railway solo marca el despliegue como healthy cuando
    `/health` contesta 200, y `/health` no se sirve hasta que este bloque termina.
    El deploy, pues, no se da por bueno con el proceso vivo sino con el modelo
    cargado, que es lo que de verdad importa.

    Un índice que falta o que no se puede abrir no lanza: un contenedor en bucle de
    arranque deja peor diagnóstico que uno vivo que contesta 503 con el motivo. El
    modelo, en cambio, sí lanza: sin él no hay servicio que valga y el error de
    arranque es lo más visible que hay.
    """
    inicio = time.perf_counter()
    embedder = cargar_embedder()
    # Una consulta real, no solo cargar: así la arena de onnxruntime queda asignada
    # y la primera petición no paga tampoco esa asignación.
    embedder.incrustar_consultas(["warm-up"])
    try:
        obtener_coleccion()
    # Doble red: `abrir_coleccion` ya traduce `ChromaError` a `IndiceNoDisponibleError`,
    # pero si chroma lanza desde otro punto (una query, un cambio suyo en los tipos de
    # excepción) aquí no se lleva por delante el arranque entero.
    except (IndiceNoDisponibleError, ChromaError) as error:
        LOGGER.error("El índice no está disponible: %s", error)
    LOGGER.info("Calentado en %.1f s", time.perf_counter() - inicio)
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(router)

# Un solo origen, el de Cloudflare Pages en producción y el de Vite en local. Con
# `allow_origins=["*"]` Starlette refleja el `Origin` de la petición, o sea que
# terminaba permitiendo a cualquiera con credenciales: aquí no hay cookies que
# proteger, pero tampoco hace falta abrirlo a todo el mundo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "app": settings.app_name}


@app.get("/health")
def health():
    """El healthcheck de Railway, y también el aviso de despliegue roto.

    200 solo con el índice delante. Si no lo está —falta, está vacío o está
    corrupto—, 503 con el motivo: es el mismo criterio que `POST /recommend`
    (un problema de índice es un fallo del despliegue, no del código), y permite
    que el healthcheck detecte un contenedor que arrancó pero que no puede buscar.
    El modelo se da por cargado porque `lifespan` no deja servir antes de tenerlo.
    """
    try:
        obtener_coleccion()
    except (IndiceNoDisponibleError, ChromaError) as error:
        raise HTTPException(
            status_code=503, detail=f"índice no disponible: {error}"
        ) from error
    return {"status": "healthy"}
