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

### Comandos del Pipeline:

1. **Construir el entorno de entrenamiento:**
   Compila las capas con dependencias pesadas optimizadas en caché.
   ```bash
   podman-compose build cbn_train
   ```

2. **Preparar y Aumentar el Dataset:**
   Toma las imágenes de la cámara (crudos + txt), hace un split de entrenamiento/validación y genera el YAML. El flag `--offline-aug` aplica transformaciones extremas (blur, ruido, giros) simulando el entorno de la fábrica.
   ```bash
   podman-compose run --rm cbn_train python3 prepare_dataset.py --input-dir dataset/lote_1 --offline-aug
   ```

3. **Ejecutar el Entrenamiento:**
   Inicia el ajuste fino de la arquitectura YOLO utilizando el dataset generado en el paso anterior.
   ```bash
   podman-compose run --rm cbn_train python3 train.py --data dataset/lote_1_done/cbn_dataset.yaml --epochs 50 --batch 16
   ```

4. **Exportar y Cuantizar a Producción (OpenVINO INT8):**
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

> **Nota:** En caso de fallos de descarga al compilar (DNS en Podman), el contenedor usa por defecto `network_mode: host` para resolver conexiones.

---

Para consultar detalles a fondo sobre resolución de problemas (Gaps resueltos) y decisiones arquitectónicas, revisa el archivo [`context.md`](context.md).
