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

# Instalación de dependencias de ML con versiones ancladas para garantizar inmutabilidad
RUN pip install --no-cache-dir ultralytics openvino==2023.3.0 openvino-dev==2023.3.0 opencv-python
WORKDIR /workspace/ws_py



# =========================================================================
# ETAPA 2: BASE C++ (Cadena de herramientas)
# Objetivo: Entorno con compiladores y librerías de Intel
# =========================================================================
FROM debian:13-slim AS cpp_base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config libopencv-dev wget gnupg2 ca-certificates \
    && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    && echo "deb https://apt.repos.intel.com/openvino/2023 debian main" > /etc/apt/sources.list.d/intel-openvino.list \
    && apt-get update && apt-get install -y --no-install-recommends openvino-2023.3.0 \
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
FROM debian:13-slim AS edge_runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.6 libopencv-videoio4.6 libopencv-imgproc4.6 \
    wget gnupg2 ca-certificates \
    && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    && echo "deb https://apt.repos.intel.com/openvino/2023 debian main" > /etc/apt/sources.list.d/intel-openvino.list \
    && apt-get update && apt-get install -y --no-install-recommends openvino-2023.3.0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash cbn_user && usermod -aG video cbn_user
WORKDIR /app
# Extracción estrictamente del binario compilado en la Etapa 3
COPY --from=cpp_builder_edge /workspace/ws_cpp/build/cbn_inference_node /app/
# Asignación de propiedad
RUN chown -R cbn_user:cbn_user /app
USER cbn_user
ENTRYPOINT ["./cbn_inference_node"]
