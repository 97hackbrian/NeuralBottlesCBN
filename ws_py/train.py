import argparse
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# PARAMETROS POR DEFECTO (Pueden sobreescribirse por CLI)
# -----------------------------
MODEL_WEIGHTS = "yolo26n.pt"  # YOLO26 no existe oficialmente, Ultralytics usará yolov8/11. Puedes cambiarlo a yolo11n.pt si usas v11
WORKERS = 2
PROJECT = "runs/detect"
RUN_NAME = "cbn_train"

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de YOLO para NeuralBottlesCBN")
    parser.add_argument("--data", type=str, required=True, help="Ruta al cbn_dataset.yaml generado (ej: dataset/lote_1_done/cbn_dataset.yaml)")
    parser.add_argument("--weights", type=str, default=MODEL_WEIGHTS, help="Pesos base del modelo")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño del batch")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de entrada de la red (ej: 320, 640)")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo: 'cpu', '0' (GPU) o 'auto' (detecta GPU automáticamente)")
    return parser.parse_args()

def resolve_device(device: str) -> str:
    """Detecta el acelerador disponible: ROCm (AMD), CUDA (NVIDIA) o CPU."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            backend = getattr(torch.version, 'hip', None)
            if backend:
                print(f"[GPU] AMD ROCm {backend} detectado: {gpu_name}")
            else:
                print(f"[GPU] NVIDIA CUDA {torch.version.cuda} detectado: {gpu_name}")
            return "0"
        else:
            print("[GPU] No se detectó GPU, usando CPU.")
            return "cpu"
    except Exception:
        return "cpu"

def run_training(args) -> Path:
    model = YOLO(args.weights)
    device = resolve_device(args.device)

    print(f"Iniciando entrenamiento con dataset: {args.data}")
    
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=WORKERS,
        device=device,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
    )

    save_dir = Path(result.save_dir)
    best_path = save_dir / "weights" / "best.pt"

    print("=== ENTRENAMIENTO FINALIZADO ===")
    print(f"Directorio de salida: {save_dir}")
    print(f"Pesos best: {best_path}")

    return best_path

if __name__ == "__main__":
    args = parse_args()
    run_training(args)

