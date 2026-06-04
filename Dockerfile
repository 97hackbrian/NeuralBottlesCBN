# =========================================================================
# ETAPA 1: ENTORNO DE ENTRENAMIENTO (Python / YOLO26)
# =========================================================================
# SOLUCIÓN: Uso de Debian 12 para forzar la inyección de Python 3.11 (Compatible con OpenVINO y Numpy precompilado)
FROM debian:12-slim AS python_env
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Creación de entorno virtual
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# SOLUCIÓN: Actualización explícita del motor de empaquetado previo a la descarga pesada
RUN pip install --upgrade pip setuptools wheel

# Copiar el fichero de dependencias y construir un wheel cache.
# Esto permite que Docker reutilice la capa si `requirements.txt` no cambia,
# evitando volver a descargar paquetes ya wheelizados en cada build.
COPY ws_py/requirements.txt /tmp/requirements.txt

# Construir ruedas en /wheels (caché por capa). Si cambias versiones en
# `requirements.txt` sólo se reconstruirán las ruedas nuevas.
RUN python3 -m pip wheel --wheel-dir=/wheels -r /tmp/requirements.txt

# Instalar desde las ruedas locales para evitar descargas innecesarias.
RUN python3 -m pip install --no-index --find-links=/wheels -r /tmp/requirements.txt

WORKDIR /workspace/ws_py



# =========================================================================
# ETAPA 2: BASE C++ (Cadena de herramientas)
# Objetivo: Entorno con compiladores y librerías OpenVINO oficiales
# =========================================================================
FROM openvino/ubuntu22_dev:2023.3.0 AS cpp_base
ENV DEBIAN_FRONTEND=noninteractive
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config libopencv-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/ws_cpp

# =========================================================================
# ETAPA 3: COMPILACIÓN CRUZADA SEGURA (Para Producción)
# Objetivo: Construir el binario inyectando los flags SSE4.2 obligatorios
# =========================================================================
FROM cpp_base AS cpp_builder_edge
COPY ws_cpp /workspace/ws_cpp
# Compilación forzada para arquitectura Silvermont (Celeron J1900)
RUN mkdir build && cd build && \
    cmake .. -G Ninja -DCMAKE_CXX_FLAGS="-msse4.2 -march=silvermont -O3" && \
    ninja

# =========================================================================
# ETAPA 4: EDGE RUNTIME (Micro-imagen para la computadora industrial)
# Objetivo: Despliegue daemonless en 4GB RAM sin compiladores
# =========================================================================
FROM openvino/ubuntu22_runtime:2023.3.0 AS edge_runtime
ENV DEBIAN_FRONTEND=noninteractive
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5 libopencv-videoio4.5 libopencv-imgproc4.5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash cbn_user && usermod -aG video cbn_user
WORKDIR /app
# Extracción estrictamente del binario compilado en la Etapa 3
COPY --from=cpp_builder_edge /workspace/ws_cpp/build/cbn_inference_node /app/
# Asignación de propiedad
RUN chown -R cbn_user:cbn_user /app
USER cbn_user
ENTRYPOINT ["./cbn_inference_node"]
