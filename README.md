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
│   ├── CMakeLists.txt   # Manifiesto de compilación (Dependencias: OpenCV, OpenVINO)
│   ├── src/             # Implementación de decodificación de tensores y NMS
│   ├── include/         # Cabeceras y estructuras de datos abstractas
│   ├── scripts/         # Automatización de compilación Ninja
│   └── models/          # Directorio receptor para la Representación Intermedia INT8
└── Dockerfile           # Receta de empaquetado multi-etapa para aislamiento de despliegue# NeuralBottlesCBN
