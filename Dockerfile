# =========================================================================
# ETAPA 1: ENTORNO DE ENTRENAMIENTO (Python / YOLO26)
# =========================================================================
# SOLUCIÓN: Uso de Debian 12 para forzar la inyección de Python 3.11 (Compatible con OpenVINO y Numpy precompilado)
FROM debian:12-slim AS python_env

# Backend de GPU: cpu | rocm | cuda
ARG GPU_BACKEND=cpu
ENV DEBIAN_FRONTEND=noninteractive
ENV GPU_BACKEND=${GPU_BACKEND}

# Para ROCm con GPUs legacy (RX 580 / gfx803)
ENV HSA_OVERRIDE_GFX_VERSION=8.0.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Creación de entorno virtual
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# SOLUCIÓN: Actualización explícita del motor de empaquetado previo a la descarga pesada
RUN pip install --upgrade pip setuptools wheel

# --- CAPA GPU: PyTorch con backend seleccionado ---
# Se instala ANTES de ultralytics para que no descargue torch CPU.
# Cada variante tiene un index URL diferente (~2.5GB).
RUN if [ "$GPU_BACKEND" = "rocm" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1; \
    elif [ "$GPU_BACKEND" = "cuda" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124; \
    else \
      echo "Backend CPU seleccionado, torch se instalará vía ultralytics"; \
    fi

# --- CAPA PESADA (ultralytics, openvino, opencv) ---
# Esta capa solo se reconstruye si cambias requirements_base.txt.
# Agregar módulos ligeros a requirements.txt NO invalida esta caché.
COPY ws_py/requirements_base.txt /tmp/requirements_base.txt
RUN pip install -r /tmp/requirements_base.txt

# --- CAPA LIGERA (albumentations, scikit-learn, etc.) ---
# Esta capa se reconstruye rápido (~50MB) cuando agregas nuevos módulos.
COPY ws_py/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace/ws_py



# =========================================================================
# ETAPA 2: BASE C++ (Cadena de herramientas)
# Objetivo: Entorno con compiladores y librerías OpenVINO oficiales
# =========================================================================
FROM openvino/ubuntu22_dev:2023.3.0 AS cpp_base
ENV DEBIAN_FRONTEND=noninteractive
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config libopencv-dev ca-certificates gpiod \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglfw3-dev libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/ws_cpp

# =========================================================================
# ETAPA 3: COMPILACIÓN CRUZADA SEGURA (Para Producción)
# Objetivo: Construir el binario inyectando los flags SSE4.2 obligatorios
# =========================================================================
FROM cpp_base AS cpp_builder_edge
COPY ws_cpp /workspace/ws_cpp
# Compilación forzada para arquitectura Silvermont (Celeron J1900)
RUN rm -rf build && mkdir build && cd build && \
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
    ca-certificates gpiod \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash cbn_user && usermod -aG video cbn_user
WORKDIR /app

# Copiar las librerías de OpenCV (v4.8) empaquetadas con OpenVINO dev
COPY --from=cpp_builder_edge /opt/intel/openvino_2023.3.0.13775/extras/opencv/lib/ /usr/local/lib/
RUN ldconfig

# Extracción estrictamente del binario compilado en la Etapa 3
COPY --from=cpp_builder_edge /workspace/ws_cpp/build/main /app/

ENTRYPOINT ["./main"]
