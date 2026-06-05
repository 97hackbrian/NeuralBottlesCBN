import shutil
from pathlib import Path

import ultralytics.utils.checks
# Bloqueo de auto-actualización de Ultralytics para preservar OpenVINO 2023.3.0
ultralytics.utils.checks.check_requirements = lambda *args, **kwargs: None
from ultralytics import YOLO


# -----------------------------
# PARAMETROS RAPIDOS (EDITABLES)
# -----------------------------
SOURCE_WEIGHTS = "runs/detect/cbn_train/weights/best.pt"
DATASET_YAML = "cbn_dataset.yaml"
IMGSZ = 640
EXPORT_INT8 = True
TARGET_CPP_MODELS_DIR = "../ws_cpp/models"
TARGET_XML_NAME = "cbn_model.xml"
TARGET_BIN_NAME = "cbn_model.bin"


def export_openvino() -> Path:
	weights_path = Path(SOURCE_WEIGHTS)
	if not weights_path.exists():
		raise FileNotFoundError(
			f"No se encontraron pesos en: {weights_path}. Ejecuta train.py primero o ajusta SOURCE_WEIGHTS."
		)

	model = YOLO(str(weights_path))
	export_dir = Path(
		model.export(
			format="openvino",
			int8=EXPORT_INT8,
			data=DATASET_YAML,
			imgsz=IMGSZ,
		)
	)

	xml_candidates = list(export_dir.glob("*.xml"))
	bin_candidates = list(export_dir.glob("*.bin"))

	if not xml_candidates or not bin_candidates:
		raise FileNotFoundError(
			f"Exportacion incompleta en {export_dir}. No se encontraron ambos archivos .xml y .bin"
		)

	source_xml = xml_candidates[0]
	source_bin = bin_candidates[0]

	target_dir = Path(TARGET_CPP_MODELS_DIR)
	target_dir.mkdir(parents=True, exist_ok=True)

	target_xml = target_dir / TARGET_XML_NAME
	target_bin = target_dir / TARGET_BIN_NAME

	shutil.copy2(source_xml, target_xml)
	shutil.copy2(source_bin, target_bin)

	print("=== EXPORTACION OPENVINO COMPLETADA ===")
	print(f"Modelo exportado en: {export_dir}")
	print(f"XML canónico C++: {target_xml.resolve()}")
	print(f"BIN canónico C++: {target_bin.resolve()}")

	return export_dir


if __name__ == "__main__":
	export_openvino()

