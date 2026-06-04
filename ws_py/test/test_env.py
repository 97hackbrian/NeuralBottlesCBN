import sys
import torch
from ultralytics import YOLO

def auditar_entorno():
    print("--- Auditoría de Entorno de Entrenamiento (MLOps) ---")
    print(f"Versión de Python: {sys.version.split()[0]}")
    print(f"Versión de PyTorch: {torch.__version__}")
    
    cuda_disponible = torch.cuda.is_available()
    print(f"Aceleración CUDA (NVCC) disponible: {cuda_disponible}")
    
    if cuda_disponible:
        print(f"Dispositivo GPU detectado: {torch.cuda.get_device_name(0)}")
    else:
        print("ADVERTENCIA: Operando en CPU. Entrenamiento será subóptimo.")

    try:
        print("\n[Fase 1] Instanciando topología YOLO y ejecutando propagación hacia adelante/atrás (1 epoch)...")
        # Descarga automática del modelo más ligero para prueba
        # Cambie 'yolov8n.pt' por la arquitectura moderna
        model = YOLO('yolo26s.pt')
        
        # Entrenamiento simulado para compilar el grafo de autograd
        dispositivo = '0' if cuda_disponible else 'cpu'
        model.train(data='coco8.yaml', epochs=1, imgsz=320, device=dispositivo, workers=1)

        print("\n[Fase 2] Iniciando transpilación del modelo a OpenVINO IR...")
        # Exportación del modelo para validar las librerías de Intel
        path_exportado = model.export(format='openvino', int8=False)
        print(f"\n[Fase 3] Exportación exitosa. Artefacto generado en: {path_exportado}")
        print("\n--- VALIDACIÓN DEL CONTENEDOR COMPLETADA SIN ERRORES ---")

    except Exception as e:
        print(f"\n--- FALLA CRÍTICA EN EL PIPELINE DE EJECUCIÓN ---")
        print(f"Traza del error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    auditar_entorno()
