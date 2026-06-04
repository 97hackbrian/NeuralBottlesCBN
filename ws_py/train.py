from pathlib import Path

from ultralytics import YOLO


# -----------------------------
# PARAMETROS RAPIDOS (EDITABLES)
# -----------------------------
MODEL_WEIGHTS = "yolo26n.pt"
DATASET_YAML = "cbn_dataset.yaml"
EPOCHS = 50
IMGSZ = 640
BATCH = 16
WORKERS = 2
DEVICE = "cpu"  # "cpu", "0" (GPU 0), o "auto"
PROJECT = "runs/detect"
RUN_NAME = "cbn_train"


def resolve_device(device: str) -> str:
	if device != "auto":
		return device

	try:
		import torch

		return "0" if torch.cuda.is_available() else "cpu"
	except Exception:
		return "cpu"


def run_training() -> Path:
	model = YOLO(MODEL_WEIGHTS)
	device = resolve_device(DEVICE)

	result = model.train(
		data=DATASET_YAML,
		epochs=EPOCHS,
		imgsz=IMGSZ,
		batch=BATCH,
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
	run_training()

