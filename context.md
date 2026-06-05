# Contexto de Proyecto: NeuralBottlesCBN
**Documento de Especificación Arquitectónica y Estado del Entorno**

## 1. Definición del Proyecto
Sistema de visión artificial para inferencia en tiempo real (borde industrial). La cámara inspecciona las botellas desde una vista superior (**topview**). Desarrollado en una estación de trabajo de alto rendimiento (**Ubuntu 22.04 LTS** con GPU NVIDIA) y desplegado en hardware de producción restringido (Intel Celeron J1900, 4GB RAM, Debian 13). Utiliza la arquitectura **YOLO26** (última generación de Ultralytics) cuantizada a enteros de 8 bits (INT8) mediante OpenVINO para ejecución máxima en CPU sin acelerador dedicado.

## 2. Topología de Infraestructura (OCI)
El sistema opera bajo el motor de contenedores **Podman** (Rootless, Daemonless).
* **Nodo de Desarrollo (Ubuntu 22.04):** Utiliza `podman-compose` (instalado vía `pip3`) para orquestación multi-etapa y mapeo de volúmenes locales. **⚠️ El acceso a GPU NVIDIA dentro de contenedores Podman aún no está operativo** (ver sección 5.5 y 6.5).
* **Nodo Industrial (Celeron J1900):** Operación minimalista sin orquestador. El contenedor se arranca vía comandos nativos de `podman` y se asegura su persistencia delegando el ciclo de vida al administrador de servicios del sistema operativo (`systemd` a nivel de usuario local).

## 3. Matriz de Dependencias y Entornos (Dockerfile Multi-stage)

### Etapa 1: Entrenamiento y Exportación (`cbn_train`)
* **SO Base:** `debian:12-slim` (Forzado para obtener Python 3.11, compatible con precompilados `.whl` de Numpy y librerías ML).
* **Herramientas Nativas:** `build-essential`, `python3-dev` (para compilación de dependencias si fallan los wheels).
* **Framework de ML:** `ultralytics` (**YOLO26** en versión flotante `>=8.3`, mandatorio para habilitar la API `safe_globals` y evitar el colapso de seguridad `weights_only=True` introducido en PyTorch 2.6).
* **Aceleración Multi-GPU:** Soporta AMD ROCm y NVIDIA CUDA mediante `ARG GPU_BACKEND` en el Dockerfile (`cpu` | `rocm` | `cuda`). PyTorch se instala con el index URL correcto antes de Ultralytics. El passthrough de dispositivos GPU se hace vía Podman 5.8.2 CDI (`/dev/kfd`, `/dev/dri` para ROCm; `nvidia.com/gpu=all` para NVIDIA). Para la RX 580 (gfx803, legacy) se aplica `HSA_OVERRIDE_GFX_VERSION=8.0.3`.
* **Transpilador:** `openvino==2023.3.0` y `openvino-dev==2023.3.0` (Anclados estrictamente por retrocompatibilidad con la Etapa C++).

### Etapa 2 y 3: Compilación C++ (`cpp_base` / `cbn_builder`)
* **SO Base:** `debian:13-slim`.
* **Cadena de Herramientas:** `cmake`, `ninja-build`, `gcc/g++`.
* **Librerías C++:** `libopencv-dev`, Repositorio APT de Intel OpenVINO 2023.3.0, `libglfw3-dev`, `libgl1-mesa-dev` (para Dear ImGui y OpenGL).
* **Cross-Compilation:** Parametrizado explícitamente para la microarquitectura del Celeron con los flags `-msse4.2 -march=silvermont`.

### Etapa 4: Inferencia en Producción (`cbn_edge`)
* **Micro-imagen inmutable:** Solo lectura de código, montura RO (Read-Only) para modelos **YOLO26** exportados.
* **Binarios:** Contiene el ejecutable C++ resultante y estrictamente las librerías compartidas (`.so`) necesarias de OpenCV 4.6 y OpenVINO Runtime. Sin compiladores ni código fuente.
* **Acceso a Hardware:** Permisos de grupo `video` en el host y contenedor para acceso DMA a `/dev/video0`.

## 4. Archivos Críticos Desarrollados

### `setup_podman.sh`
Script de aprovisionamiento *bare-metal* determinista. Identifica automáticamente el SO subyacente (APT o Pacman) y gestiona los registros en formato TOML V2 (`00-shortnames.conf` con `unqualified-search-registries`).
* **Uso en Desarrollo:** Instala Podman, redes (`slirp4netns`), mapeo de usuarios (`uidmap` o `shadow`). En Debian/Ubuntu instala `python3-pip` y luego `podman-compose` vía `pip3 install podman-compose`. En Arch/Manjaro instala `podman-compose` desde el gestor de paquetes nativo.
* **Uso en Producción (`--production`):** Instala Podman y las herramientas de diagnóstico del bus óptico del kernel (`v4l-utils`). Bloquea deliberadamente la instalación de `compose` y Python.

### `docker-compose.yaml`
Manifiesto de orquestación de desarrollo. Contiene la inyección de volúmenes (`ws_py`, `ws_cpp/models`). La sección de GPU está **comentada** con instrucciones para ejecución manual vía `podman run --device nvidia.com/gpu=all` (requiere `nvidia-container-toolkit` y CDI generado). `podman-compose` no soporta passthrough de GPU directamente.

### `ws_py/test_env.py`
Script de auditoría integral (Smoke Test). Instancia el entorno virtual, verifica el acceso efectivo a la memoria VRAM (CUDA), descarga la arquitectura **YOLO26** base, ejecuta una época de entrenamiento simulada (Autograd) y valida la serialización a representación intermedia (IR) de Intel.

## 5. Decisiones Técnicas y Gaps Resueltos
1.  **Crash de Numpy/Meson:** Resuelto mediante degradación a Debian 12 (Python 3.11) y actualización forzada de `pip/setuptools/wheel` (PEP 632).
2.  **Error 404 Hello World:** Resuelto aplicando sintaxis TOML V2 a los registros de ingesta OCI (`unqualified-search-registries = ["docker.io", "quay.io"]`).
3.  **Crash de Seguridad PyTorch 2.6:** Resuelto liberando el *version pinning* de `ultralytics` para que el sistema descargue **YOLO26**, el cual es compatible con la nueva API de deserialización segura de tensores.
4.  **Paridad C++/Python:** OpenVINO mantenido rígidamente en 2023.3.0 en ambos ecosistemas para evitar un "IR Version Mismatch" fatal al momento de ejecutar la inferencia en el borde.
5.  **GPU en contenedores Podman (✅ RESUELTO):** Podman 5.8.2 soporta CDI nativamente. Se implementó soporte multi-GPU (AMD ROCm + NVIDIA CUDA + CPU fallback) mediante `ARG GPU_BACKEND` en el Dockerfile. Para ROCm, se pasan `/dev/kfd` y `/dev/dri` como devices en `docker-compose.yaml`. Para NVIDIA, se usa CDI (`nvidia.com/gpu=all`) previa generación de `/etc/cdi/nvidia.yaml`. `train.py` detecta automáticamente el backend (ROCm vía `torch.version.hip`, CUDA vía `torch.version.cuda`) con `--device auto`.

## 6. Estado Actual del Repositorio y Modificaciones Recientes

### 6.1 Cambios ya aplicados
* **Optimización de Caché en Docker:** Se separaron las dependencias de Python en `ws_py/requirements_base.txt` (pesadas: ultralytics, openvino, opencv) y `ws_py/requirements.txt` (ligeras: albumentations, scikit-learn). El `Dockerfile` instala estas capas de forma secuencial, evitando volver a descargar librerías pesadas al agregar módulos de experimentación. (El mecanismo antiguo de `pip wheel` fue descartado).
* **Preparación de Dataset:** Se implementó `ws_py/prepare_dataset.py` para organizar las imágenes crudas, realizar un split de 80/20 (train/val) usando `scikit-learn`, y aplicar Data Augmentation offline físico (opcional, vía Albumentations con transformaciones industriales como Motion Blur, Flips H/V y Gauss Noise) o dejarlo para aumento online por defecto.
* `.gitignore` fue extendido para ignorar contenido generado por `ws_py/datasets/`, `ws_py/runs/` y todos los archivos `*.pt`.
* Se agregaron placeholders `.gitkeep` en carpetas vacías que deben preservarse en Git: `ws_py/dataset/`, `ws_py/datasets/` y `ws_py/runs/`.
* `setup_podman.sh` actualizado: en Debian/Ubuntu instala `python3-pip` y ejecuta `pip3 install podman-compose` (con fallback `--break-system-packages` para PEP 668). Los registros OCI ahora usan formato TOML V2 (`unqualified-search-registries`).
* `docker-compose.yaml`: Se añadió `network_mode: host` en `cbn_test_cpp` para solucionar fallos de resolución DNS (aardvark-dns) con Podman. El comando de compilación se actualizó para compilar y ejecutar `laboratorio_cbn`.
* **Fusión de Entorno C++**: Se migró el código C++ del antiguo workspace (`laboratorio_cbn.cpp`, `pipeline.cpp`, `pipeline.h`) a `ws_cpp/src/` y `ws_cpp/include/`. Se estructuraron las configuraciones en `ws_cpp/config/` (migrando `config*.yaml` e `imgui.ini` y consolidando `labels.yaml` dinámico).
* `Dockerfile` y `CMakeLists.txt` actualizados: Se agregaron las dependencias gráficas (`libglfw3-dev`, `libgl1-mesa-dev`) en una nueva capa de caché y se configuró `FetchContent` en CMake para descargar e integrar `Dear ImGui`. Se activó la compilación cruzada hacia Celeron J1900 inyectando banderas SSE4.2 y extrayendo dinámicamente las librerías precompiladas de OpenCV de Intel OpenVINO.
* **Inferencia Nativa C++ (Completada):** Se desarrollaron y probaron dos ejecutables base utilizando el API `ov::Core`: `cbn_test_inference` (con GUI y dibujo de cajas delimitadoras corregidas de `[xmin, ymin, xmax, ymax]`) y `cbn_inference_node` (nodo headless de producción, operando a alta velocidad en CPU). Ambos cargan las etiquetas de `ws_cpp/config/labels.yaml` vía OpenCV `FileStorage`.
* **Rutas Dinámicas (C++)**: Se implementó un esquema de rutas robusto mediante `getWorkspacePath()` (vía `/proc/self/exe`), integrando el guardado de capturas (Burst) y el procesamiento de lotes directamente en `ws_py/dataset/` (`Captures/`, `raw/`, `processed/`), con creación automática de carpetas faltantes para garantizar que funcione al primer intento ("plug and play") en cualquier entorno.
* **Mapeo de Dispositivos (Podman Rootless):** Se removió el usuario sin privilegios `cbn_user` del contenedor Edge para evitar errores de permisos V4L2 (`Camera index out of range`) sobre `/dev/video0` mapeado desde el host.

### 6.2 Estado funcional detectado
* La etapa C++ es funcional: `ws_cpp/CMakeLists.txt` compila exitosamente el binario principal `laboratorio_cbn`, los ejecutables de validación, y los nodos de inferencia OpenVINO. Se validó que el modelo YOLO26 exportado es consumido correctamente logrando hasta ~60 FPS (16-24ms) en inferencia nativa por CPU.
* **Pipeline MLOps Operativo:** La cadena completa en Python (Preparación, Entrenamiento y Exportación) está implementada y auditada sin bugs:
  - `prepare_dataset.py` maneja divisiones train/val y data augmentation (Albumentations) soportando imágenes de fondo sin etiquetas.
  - `train.py` entrena el modelo usando `argparse` para consumir el yaml generado dinámicamente.
  - `export_int8.py` se refactorizó con `argparse` (recibe `--data` y `--weights`) para resolver un bug crítico de calibración, exportando exitosamente los pesos finales a formato OpenVINO INT8 y auto-copiándolos a la carpeta C++ de producción.
* `ws_py/test/test_env.py` detecta y reporta tanto ROCm (AMD) como CUDA (NVIDIA) además de la variable `GPU_BACKEND`.
* `docker-compose.yaml` mantiene dos flujos: `cbn_train` para entrenamiento con soporte multi-GPU (ROCm/CUDA/CPU); `cbn_test_cpp` para pruebas C++; `cbn_edge` para producción.
* **El entrenamiento YOLO26 soporta GPU AMD (ROCm) y NVIDIA (CUDA)** con detección automática. CPU se usa como fallback.

### 6.3 Flujo recomendado para agentes futuros
* Antes de editar, leer este `context.md`, `Dockerfile`, `docker-compose.yaml`, `ws_py/requirements.txt` y el árbol de `ws_cpp/` para confirmar si hubo cambios nuevos.
* Preferir cambios pequeños y trazables: primero dependencias y build reproducible, luego skeletons de código C++, luego scripts Python, y al final ajuste de despliegue.
* Validar siempre con un build focalizado: `podman build --target python_env ...` para la etapa Python y un build CMake aislado para C++ cuando existan fuentes reales.
* Para probar funcionamiento de scripts Python dentro del contenedor de entrenamiento usar: `podman-compose run --rm cbn_train python3`.
* Por ningún motivo crear entornos virtuales Python en el sistema local del host; toda ejecución de entrenamiento/exportación/pruebas debe correr dentro del contenedor `cbn_train`.
* Mantener dos ficheros de dependencias Python: `ws_py/requirements_base.txt` (paquetes pesados que rara vez cambian: ultralytics, openvino, opencv) y `ws_py/requirements.txt` (paquetes ligeros/experimentales: albumentations, scikit-learn, etc.). El `Dockerfile` los instala en capas separadas para que agregar módulos ligeros no invalide la caché de los pesados.
* No comitear datasets, resultados de `runs/` ni pesos `.pt`; sólo mantener `.gitkeep` en directorios que deban sobrevivir vacíos.

### 6.4 Observaciones operativas
* El repo se encuentra en un estado funcional avanzado tanto para la experimentación C++ (interfaz de laboratorio integrada con Dear ImGui) como para el entrenamiento del modelo.
* **El pipeline completo de Python (`prepare`, `train`, `export`) está finalizado y libre de bugs.**
* **El puente a C++ (OpenVINO) también está finalizado**. El modelo `.xml`/`.bin` es procesado a alta velocidad por la lógica de ejecución en `ws_cpp/src/cbn_detector_inference.cpp` lista para desplegarse en la máquina industrial. El trabajo futuro deberá concentrarse en integraciones GPIO, PLCs o lógicas de negocio según lo demande la fábrica.

### 6.5 Problemas conocidos pendientes
* **GPU AMD RX 580 (gfx803) — hardware legacy:** La RX 580 fue eliminada del soporte oficial de ROCm a partir de la versión 6.0. Se fuerza compatibilidad con `HSA_OVERRIDE_GFX_VERSION=8.0.3`. Puede haber inestabilidad en operaciones avanzadas. Para GPUs Vega/RDNA/RDNA2+ no se necesita este workaround.
* **Auto-actualización de OpenVINO (RESUELTO):** Ultralytics intentaba auto-actualizar OpenVINO de `2023.3.0` a `2026.2.0` durante la exportación. Esto se bloqueó inyectando un *monkey-patch* (`ultralytics.utils.checks.check_requirements = lambda *args, **kwargs: None`) en `test_env.py` y `export_int8.py` justo antes de importar YOLO, preservando la paridad con la etapa C++.
* **Driver de almacenamiento Podman:** `ERRO graph driver "overlay" overwritten by "vfs"`. La base de datos local de Podman se creó con `vfs` pero la configuración pide `overlay`. Solución: `podman system reset` (elimina imágenes y contenedores existentes).
* **Red CNI:** `WARN plugin firewall does not support config version "1.0.0"`. No bloquea la ejecución, es un aviso de incompatibilidad menor en la configuración de red.