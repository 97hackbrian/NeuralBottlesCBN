# Contexto de Proyecto: NeuralBottlesCBN
**Documento de Especificación Arquitectónica y Estado del Entorno**

## 1. Definición del Proyecto
Sistema de visión artificial para inferencia en tiempo real (borde industrial). Desarrollado en una estación de trabajo de alto rendimiento (**Ubuntu 22.04 LTS** con GPU NVIDIA) y desplegado en hardware de producción restringido (Intel Celeron J1900, 4GB RAM, Debian 13). Utiliza la arquitectura **YOLO26** (última generación de Ultralytics) cuantizada a enteros de 8 bits (INT8) mediante OpenVINO para ejecución máxima en CPU sin acelerador dedicado.

## 2. Topología de Infraestructura (OCI)
El sistema opera bajo el motor de contenedores **Podman** (Rootless, Daemonless).
* **Nodo de Desarrollo (Ubuntu 22.04):** Utiliza `podman-compose` (instalado vía `pip3`) para orquestación multi-etapa y mapeo de volúmenes locales. **⚠️ El acceso a GPU NVIDIA dentro de contenedores Podman aún no está operativo** (ver sección 5.5 y 6.5).
* **Nodo Industrial (Celeron J1900):** Operación minimalista sin orquestador. El contenedor se arranca vía comandos nativos de `podman` y se asegura su persistencia delegando el ciclo de vida al administrador de servicios del sistema operativo (`systemd` a nivel de usuario local).

## 3. Matriz de Dependencias y Entornos (Dockerfile Multi-stage)

### Etapa 1: Entrenamiento y Exportación (`cbn_train`)
* **SO Base:** `debian:12-slim` (Forzado para obtener Python 3.11, compatible con precompilados `.whl` de Numpy y librerías ML).
* **Herramientas Nativas:** `build-essential`, `python3-dev` (para compilación de dependencias si fallan los wheels).
* **Framework de ML:** `ultralytics` (**YOLO26** en versión flotante `>=8.3`, mandatorio para habilitar la API `safe_globals` y evitar el colapso de seguridad `weights_only=True` introducido en PyTorch 2.6).
* **Aceleración:** CUDA **no operativo aún** dentro del contenedor. `podman-compose` no soporta la sintaxis CDI (`--device nvidia.com/gpu=all`) ni `deploy > resources > reservations > devices`. El entrenamiento actual opera en CPU.
* **Transpilador:** `openvino==2023.3.0` y `openvino-dev==2023.3.0` (Anclados estrictamente por retrocompatibilidad con la Etapa C++).

### Etapa 2 y 3: Compilación C++ (`cpp_base` / `cbn_builder`)
* **SO Base:** `debian:13-slim`.
* **Cadena de Herramientas:** `cmake`, `ninja-build`, `gcc/g++`.
* **Librerías C++:** `libopencv-dev`, Repositorio APT de Intel OpenVINO 2023.3.0.
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
5.  **GPU en contenedores Podman (⚠️ NO RESUELTO):** `podman-compose` no soporta passthrough de GPU ni por `deploy.resources.reservations.devices` ni por CDI (`nvidia.com/gpu=all` en `devices`). El método manual `podman run --device nvidia.com/gpu=all` también falla con `stat nvidia.com/gpu=all: no such file or directory`, lo que indica que Podman 3.x (versión de Ubuntu 22.04) no soporta CDI (requiere Podman ≥4.1). **Solución pendiente:** actualizar Podman a 4.x+ o configurar OCI hooks con `nvidia-ctk runtime configure --runtime=podman`.

## 6. Estado Actual del Repositorio y Modificaciones Recientes

### 6.1 Cambios ya aplicados
* `ws_py/requirements.txt` ahora concentra las dependencias Python que antes estaban embebidas en el `Dockerfile`.
* `Dockerfile` de `python_env` fue actualizado para construir un wheelhouse en `/wheels` y reinstalar desde ruedas locales con `pip --no-index`, reduciendo descargas repetidas cuando cambian poco las dependencias.
* `.gitignore` fue extendido para ignorar contenido generado por `ws_py/datasets/`, `ws_py/runs/` y todos los archivos `*.pt`.
* Se agregaron placeholders `.gitkeep` en carpetas vacías que deben preservarse en Git: `ws_py/dataset/`, `ws_py/datasets/` y `ws_py/runs/`.
* `setup_podman.sh` actualizado: en Debian/Ubuntu instala `python3-pip` y ejecuta `pip3 install podman-compose` (con fallback `--break-system-packages` para PEP 668). Los registros OCI ahora usan formato TOML V2 (`unqualified-search-registries`).
* `docker-compose.yaml`: la sección de GPU fue comentada con instrucciones para ejecución manual vía `podman run`.

### 6.2 Estado funcional detectado
* `ws_cpp/CMakeLists.txt`, `ws_cpp/src/main.cpp`, `ws_cpp/src/cbn_detector.cpp` y `ws_cpp/include/cbn_detector.hpp` están vacíos; el binario `cbn_inference_node` no puede compilarse todavía.
* `ws_py/train.py` y `ws_py/export_int8.py` están vacíos; el servicio `cbn_train` hoy no realiza entrenamiento ni exportación real.
* `ws_py/test/test_env.py` referencia `YOLO('yolo11n.pt')`, lo que debe verificarse porque puede no coincidir con el artefacto real esperado para esta línea de trabajo.
* `docker-compose.yaml` mantiene dos flujos: `cbn_train` para etapa Python y `cbn_test_cpp` para compilación local; `cbn_edge` depende de una imagen externa llamada `neuralbottles_edge:latest`.
* **El entrenamiento YOLO26 funciona correctamente pero solo en CPU.** La GPU NVIDIA no es accesible desde dentro del contenedor Podman.

### 6.3 Flujo recomendado para agentes futuros
* Antes de editar, leer este `context.md`, `Dockerfile`, `docker-compose.yaml`, `ws_py/requirements.txt` y el árbol de `ws_cpp/` para confirmar si hubo cambios nuevos.
* Preferir cambios pequeños y trazables: primero dependencias y build reproducible, luego skeletons de código C++, luego scripts Python, y al final ajuste de despliegue.
* Validar siempre con un build focalizado: `podman build --target python_env ...` para la etapa Python y un build CMake aislado para C++ cuando existan fuentes reales.
* Para probar funcionamiento de scripts Python dentro del contenedor de entrenamiento usar: `podman-compose run --rm cbn_train python3`.
* Por ningún motivo crear entornos virtuales Python en el sistema local del host; toda ejecución de entrenamiento/exportación/pruebas debe correr dentro del contenedor `cbn_train`.
* Mantener una sola fuente de verdad para dependencias Python en `ws_py/requirements.txt` y evitar duplicarlas dentro del `Dockerfile`.
* No comitear datasets, resultados de `runs/` ni pesos `.pt`; sólo mantener `.gitkeep` en directorios que deban sobrevivir vacíos.

### 6.4 Observaciones operativas
* El repo está en una fase intermedia: la infraestructura de contenedores ya fue racionalizada parcialmente, pero la capa de aplicación aún no está implementada en C++ ni en Python.
* El trabajo futuro debe centrarse en completar el binario C++, definir el pipeline de entrenamiento/exportación y cerrar la paridad OpenVINO entre Python y C++.

### 6.5 Problemas conocidos pendientes
* **GPU no disponible en contenedores:** Podman 3.x (default en Ubuntu 22.04) no soporta CDI (Container Device Interface). Se requiere Podman ≥4.1 o configurar OCI hooks manualmente con `nvidia-ctk runtime configure --runtime=podman`. Mientras tanto, el entrenamiento opera exclusivamente en CPU.
* **Auto-actualización de OpenVINO:** Ultralytics auto-actualizó OpenVINO de `2023.3.0` a `2026.2.0` durante el smoke test. Esto **rompe la paridad** con la etapa C++ que depende de OpenVINO 2023.3.0. Se debe anclar `openvino==2023.3.0` en `ws_py/requirements.txt` y bloquear la auto-actualización.
* **Driver de almacenamiento Podman:** `ERRO graph driver "overlay" overwritten by "vfs"`. La base de datos local de Podman se creó con `vfs` pero la configuración pide `overlay`. Solución: `podman system reset` (elimina imágenes y contenedores existentes).
* **Red CNI:** `WARN plugin firewall does not support config version "1.0.0"`. No bloquea la ejecución, es un aviso de incompatibilidad menor en la configuración de red.