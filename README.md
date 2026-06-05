# NeuralBottlesCBN: Sistema de Inspección Visual Industrial

## Descripción del Proyecto
**NeuralBottlesCBN** es un *pipeline* de visión artificial de alto rendimiento diseñado para el control de calidad en líneas de empaque de la Cervecería Boliviana Nacional (CBN). Su objetivo singular es la validación binaria ("pasa/no pasa") de casilleros de 12 botellas, determinando su completitud mediante arquitecturas de detección de objetos (YOLO).

Este repositorio está arquitectónicamente dividido para aislar el entorno de entrenamiento profundo (basado en Python/PyTorch) del motor de inferencia determinista (basado en C++/OpenVINO), garantizando una ejecución óptima y de baja latencia sobre hardware industrial heredado (*Legacy Edge Computing*).

---

## Restricciones de Hardware y Despliegue (Edge Node)
El nodo de inferencia final está sujeto a restricciones computacionales severas que dictan la arquitectura de este repositorio:
* **Procesador:** Intel Celeron J1900 (Microarquitectura Silvermont, 4 núcleos @ 2.41 GHz).
* **Limitaciones Críticas:** Ausencia total de instrucciones vectoriales AVX/AVX2/AVX-512. Soporte máximo para SSE4.2.
* **Memoria RAM:** 4 GB DDR3.
* **Motor de Inferencia:** Obligatoriamente **OpenVINO (Intel)** operando sobre tensores cuantizados a enteros de 8 bits (INT8).

**⚠️ ADVERTENCIA CRÍTICA DE COMPILACIÓN (SIGILL):**
Queda terminantemente prohibido compilar el código fuente de C++ (`ws_cpp`) directamente en la máquina de destino (J1900) debido al riesgo inminente de agotamiento de memoria (*OOM Killer*). El binario debe ser construido mediante **compilación cruzada (*cross-compilation*)** en una estación de trabajo x86_64 superior, restringiendo explícitamente los *flags* del compilador a la arquitectura objetivo.

---

## Topología del Monorepositorio

La arquitectura del repositorio segrega las responsabilidades lógicas y procedimentales:

```text
NeuralBottlesCBN/
├── ws_py/               # Entorno de Entrenamiento y Cuantización (Python)
│   ├── train.py         # Script de entrenamiento de arquitectura YOLO
│   └── export_int8.py   # Congelación de grafo y conversión a OpenVINO IR (.xml / .bin)
├── ws_cpp/              # Entorno de Inferencia y Despliegue (C++)
│   ├── CMakeLists.txt   # Manifiesto de compilación (OpenCV, OpenVINO, ImGui, GLFW)
│   ├── config/          # Configuraciones YAML y layouts de ImGui (imgui.ini)
│   ├── src/             # Fuentes C++ (laboratorio_cbn.cpp, pipeline.cpp)
│   ├── include/         # Cabeceras C++ (pipeline.h)
│   ├── test/            # Scripts de validación de hardware
│   └── models/          # Directorio receptor para la Representación Intermedia INT8
├── docker-compose.yaml  # Orquestador local para compilación y desarrollo
└── Dockerfile           # Receta multi-etapa para compilación C++ y despliegue
```

## Fase de Entrenamiento y MLOps (Python)

El flujo de Machine Learning se ejecuta estrictamente dentro del contenedor `cbn_train`, el cual empaqueta PyTorch, Ultralytics (YOLO) y las herramientas de OpenVINO en un entorno Debian aislado. ¡No instales dependencias localmente!

### Selección de GPU (Build Time)

El contenedor soporta tres backends de aceleración. Selecciona el que corresponda a tu hardware editando `GPU_BACKEND` en `docker-compose.yaml` o pasándolo como build arg:

```bash
# AMD ROCm (RX 580, Vega, RDNA) — default en docker-compose.yaml
podman-compose build --build-arg GPU_BACKEND=rocm cbn_train

# NVIDIA CUDA (GTX/RTX)
podman-compose build --build-arg GPU_BACKEND=cuda cbn_train

# CPU solamente (sin GPU)
podman-compose build --build-arg GPU_BACKEND=cpu cbn_train
```

> **⚠️ Nota AMD RX 580:** Esta GPU (gfx803) es hardware legacy para ROCm. Se usa `HSA_OVERRIDE_GFX_VERSION=8.0.3` para forzar compatibilidad. Para GPUs Vega/RDNA más nuevas, esta variable no es necesaria.

> **⚠️ Nota NVIDIA:** Requiere `nvidia-container-toolkit` instalado y CDI generado: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Además, descomentar la sección NVIDIA en `docker-compose.yaml`.

### Comandos del Pipeline:

1. **Construir el entorno de entrenamiento:**
   ```bash
   podman-compose build cbn_train
   ```

2. **Verificar detección de GPU:**
   ```bash
   podman-compose run --rm cbn_train python3 test/test_env.py
   ```

3. **Preparar y Aumentar el Dataset:**
   Toma las imágenes de la cámara (crudos + txt), hace un split de entrenamiento/validación y genera el YAML. El flag `--offline-aug` aplica transformaciones extremas (blur, ruido, giros) simulando el entorno de la fábrica.
   ```bash
   podman-compose run --rm cbn_train python3 prepare_dataset.py --input-dir dataset/lote_1 --offline-aug
   ```

4. **Ejecutar el Entrenamiento:**
   Inicia el ajuste fino de la arquitectura YOLO. La GPU se detecta automáticamente (`--device auto` es el default).
   ```bash
   podman-compose run --rm cbn_train python3 train.py --data dataset/lote_1_done/cbn_dataset.yaml --epochs 50 --batch 16
   ```

5. **Exportar y Cuantizar a Producción (OpenVINO INT8):**
   Toma los mejores pesos del entrenamiento (`best.pt`), los cuantiza a enteros de 8-bits para máxima aceleración en el procesador Celeron J1900, y auto-copia el modelo resultante a la carpeta C++ (`ws_cpp/models/`).
   ```bash
   podman-compose run --rm cbn_train python3 export_int8.py --data dataset/lote_1_done/cbn_dataset.yaml
   ```

---

## Ejecución del Laboratorio C++ (Desarrollo)

El entorno C++ local (`cbn_test_cpp`) integra una interfaz gráfica basada en **Dear ImGui** para configurar parámetros, inspeccionar visualmente los casilleros de botellas y operar las cámaras en tiempo real.

Debido a que el contenedor debe comunicarse con la pantalla principal, es **obligatorio** habilitar el acceso al servidor gráfico X11 antes de arrancarlo.

### Pasos:

1. **Permitir la conexión de la interfaz gráfica local:**
   ```bash
   xhost +local:
   ```
2. **Construir la imagen de contenedor C++:**
   ```bash
   podman-compose build cbn_test_cpp
   ```
3. **Lanzar la interfaz visual (Laboratorio CBN):**
   ```bash
   podman-compose run --rm cbn_test_cpp
   ```
4. **Probar el nodo de inferencia puro en desarrollo:**
   ```bash
   podman-compose run --rm cbn_test_cpp /bin/bash -lc "mkdir -p build && cd build && cmake .. -G Ninja && ninja cbn_inference_node && cd .. && ./build/cbn_inference_node"
   ```

> **Nota:** En caso de fallos de descarga al compilar (DNS en Podman), el contenedor usa por defecto `network_mode: host` para resolver conexiones.

---

## Fase de Inferencia en Producción (Edge Deployment)

El entorno final se compila utilizando un método de *cross-compiling* optimizado para el procesador Celeron J1900. No se necesitan compiladores en la máquina destino.

### 1. Generar la imagen para producción (En la máquina de desarrollo)

Este comando compila el binario con soporte SSE4.2 y lo aísla en una micro-imagen limpia:
```bash
podman build --target edge_runtime -t localhost/neuralbottles_edge:latest .
```

Luego, empaqueta la imagen para transferirla a un USB/red:
```bash
podman save -o neuralbottles_edge.tar localhost/neuralbottles_edge:latest
```

### 2. Despliegue en la máquina industrial (Celeron J1900)

Transfiere el `.tar`, la carpeta `ws_cpp/models` y `ws_cpp/config` al equipo final, y carga la imagen:
```bash
podman load -i neuralbottles_edge.tar
```

Finalmente, levanta el contenedor mapeando la cámara industrial y los modelos. Podman debe correr como tu usuario normal, pero se elimina el cambio de usuario interno para no perder el mapeo de permisos de `/dev/video0`:
```bash
podman run -d \
  --name neuralbottles_inference \
  --device /dev/video0:/dev/video0 \
  --group-add video \
  -v ./ws_cpp/models:/app/models:ro \
  -v ./ws_cpp/config:/app/config:ro \
  --restart unless-stopped \
  localhost/neuralbottles_edge:latest
```

---

Para consultar detalles a fondo sobre resolución de problemas (Gaps resueltos) y decisiones arquitectónicas, revisa el archivo [`context.md`](context.md).
