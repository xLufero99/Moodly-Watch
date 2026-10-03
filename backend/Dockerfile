# Python 3.11 (el de `.python-version`) con uv ya dentro, sobre Debian slim: los
# wheels de onnxruntime, chromadb y numpy son manylinux, o sea glibc, y en alpine
# no hay con los que trabajar.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# libstdc++ la enlazan onnxruntime y pyarrow (verificado con readelf) y el slim no
# siempre la trae. No hace falta libgomp1: los .so de onnxruntime no la piden.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    UV_CACHE_DIR=/tmp/uv-cache

# Dependencias primero: esta capa solo se reconstruye si cambia el lock, no si
# cambia una línea de app/. Sin --no-install-project el build pediría README y
# src/ antes de que estén copiados.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
# El `rm` es obligatorio: la caché de uv (ruedas descomprimidas en archive-v0)
# pesó 339 MB, casi el doble que el propio venv, y fuera de este RUN no le sirve
# a nadie. Se borra en la misma capa, que es lo único que de verdad la quita.
RUN uv sync --frozen --no-dev --no-install-project \
    && rm -rf /tmp/uv-cache

# Artefactos de Hugging Face, antes del código para que un cambio en app/ no
# obligue a bajar otra vez 220 MB. `--local-dir` es lo que espera
# vector_store.DEFAULT_DIR_INDICE = data/index. El fichero del modelo se nombra a
# pelo: la carpeta onnx/ del repo Xenova pesa 2 GB y solo sirve el cuantizado.
RUN hf download DevLufero/moodly-watch-index --repo-type dataset --local-dir data/index \
    && rm -rf data/index/.cache
RUN hf download intfloat/multilingual-e5-small sentencepiece.bpe.model \
    && hf download Xenova/multilingual-e5-small onnx/model_quantized.onnx

# Ya está todo bajado: a partir de aquí nada tiene que salir a la red en el arranque.
# Va después de los downloads porque la variable también vale para la fase de build.
ENV HF_HUB_OFFLINE=1

# `app/` no está en el venv (uv_build solo empaqueta src/), así que se sirve desde
# el directorio de trabajo, igual que en local.
COPY app/ ./app/
COPY scripts/ ./scripts/
RUN uv sync --frozen --no-dev \
    && rm -rf /tmp/uv-cache

EXPOSE 8000
# $PORT lo inyecta Railway; `exec` para que uvicorn reciba el SIGTERM y pueda
# drenar. Un solo worker: cada uno llevaría su copia del modelo en RAM.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
