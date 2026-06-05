import shutil
import argparse
from pathlib import Path

import ultralytics.utils.checks
# Bloqueo de auto-actualización de Ultralytics para preservar OpenVINO 2023.3.0
ultralytics.utils.checks.check_requirements = lambda *args, **kwargs: None
from ultralytics import YOLO


# -----------------------------
# PARAMETROS DE EXPORTACION A C++
# -----------------------------
TARGET_CPP_MODELS_DIR = "../ws_cpp/models"
TARGET_XML_NAME = "cbn_model.xml"
TARGET_BIN_NAME = "cbn_model.bin"
IMGSZ = 640


def parse_args():
    parser = argparse.ArgumentParser(description="Exportar modelo YOLO a OpenVINO INT8")
    parser.add_argument("--weights", type=str, default="runs/detect/runs/detect/cbn_train/weights/best.pt", help="Ruta a los pesos entrenados")
    parser.add_argument("--data", type=str, required=True, help="Ruta al cbn_dataset.yaml usado en el entrenamiento (necesario para calibración INT8)")
    parser.add_argument("--no-int8", action="store_true", help="Desactiva la cuantización INT8 (exporta en FP16/FP32)")
    return parser.parse_args()


def export_openvino(args) -> Path:
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No se encontraron pesos en: {weights_path}. Ejecuta train.py primero o ajusta --weights."
        )

    model = YOLO(str(weights_path))
    export_dir = Path(
        model.export(
            format="openvino",
            int8=not args.no_int8,
            data=args.data,
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
    args = parse_args()
    export_openvino(args)

