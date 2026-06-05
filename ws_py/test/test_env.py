import sys
import torch
import ultralytics.utils.checks
# Bloqueo de auto-actualización de Ultralytics para preservar OpenVINO 2023.3.0
ultralytics.utils.checks.check_requirements = lambda *args, **kwargs: None
from ultralytics import YOLO

def auditar_entorno():
    print("--- Auditoría de Entorno de Entrenamiento (MLOps) ---")
    print(f"Versión de Python: {sys.version.split()[0]}")
    print(f"Versión de PyTorch: {torch.__version__}")

    # Detección de backend GPU
    cuda_disponible = torch.cuda.is_available()
    hip_version = getattr(torch.version, 'hip', None)

    if cuda_disponible and hip_version:
        print(f"Aceleración ROCm (AMD) disponible: True")
        print(f"Versión HIP: {hip_version}")
        print(f"Dispositivo GPU detectado: {torch.cuda.get_device_name(0)}")
    elif cuda_disponible:
        print(f"Aceleración CUDA (NVIDIA) disponible: True")
        print(f"Versión CUDA: {torch.version.cuda}")
        print(f"Dispositivo GPU detectado: {torch.cuda.get_device_name(0)}")
    else:
        print("Aceleración GPU disponible: False")
        print("ADVERTENCIA: Operando en CPU. Entrenamiento será subóptimo.")

    # Backend activo reportado por el contenedor
    import os
    gpu_backend_env = os.environ.get("GPU_BACKEND", "no definido")
    print(f"GPU_BACKEND (Dockerfile ARG): {gpu_backend_env}")

    try:
        print("\n[Fase 1] Instanciando topología YOLO y ejecutando propagación hacia adelante/atrás (1 epoch)...")
        model = YOLO('yolo26n.pt')

        dispositivo = '0' if cuda_disponible else 'cpu'
        model.train(data='coco8.yaml', epochs=1, imgsz=320, device=dispositivo, workers=1)

        print("\n[Fase 2] Iniciando transpilación del modelo a OpenVINO IR...")
        path_exportado = model.export(format='openvino', int8=False)
        print(f"\n[Fase 3] Exportación exitosa. Artefacto generado en: {path_exportado}")
        print("\n--- VALIDACIÓN DEL CONTENEDOR COMPLETADA SIN ERRORES ---")

    except Exception as e:
        print(f"\n--- FALLA CRÍTICA EN EL PIPELINE DE EJECUCIÓN ---")
        print(f"Traza del error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    auditar_entorno()
