import os
import glob
import shutil
import argparse
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A

def parse_args():
    parser = argparse.ArgumentParser(description="Prepara el dataset de NeuralBottlesCBN")
    parser.add_argument("--input-dir", type=str, required=True, help="Carpeta cruda a procesar (ej: dataset/lote_1)")
    parser.add_argument("--offline-aug", action="store_true", help="Aplica aumento físico con Albumentations")
    return parser.parse_args()

def get_augmentation_pipeline():
    # Pipeline optimizado para topview en entorno industrial
    return A.Compose([
        A.MotionBlur(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), # Seguro usar porque es topview
        A.SafeRotate(limit=10, p=0.3), # Rotación ligera simulando botellas torcidas
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_yolo_label(label_path):
    # Lee el txt y retorna una lista de cajas y otra de labels
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_labels.append(int(parts[0]))
                    # Albumentations YOLO format: [x_center, y_center, width, height] normalizados
                    bboxes.append([float(x) for x in parts[1:5]])
    return bboxes, class_labels

def write_yolo_label(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            f.write(f"{cls} {' '.join([f'{x:.6f}' for x in bbox])}\n")

def process_and_copy(data_split, target_dir, apply_aug=False):
    images_dir = os.path.join(target_dir, "images")
    labels_dir = os.path.join(target_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    transform = get_augmentation_pipeline() if apply_aug else None
    
    for img_path, txt_path in data_split:
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        
        target_img_path = os.path.join(images_dir, base_name)
        target_txt_path = os.path.join(labels_dir, name + ".txt")
        
        # Copiar versión original
        shutil.copy2(img_path, target_img_path)
        if txt_path and os.path.exists(txt_path):
            shutil.copy2(txt_path, target_txt_path)
        else:
            open(target_txt_path, 'a').close()
            
        if apply_aug and txt_path and os.path.exists(txt_path):
            # Cargar imagen original para aumentos
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, class_labels = read_yolo_label(txt_path)
            
            # Solo aumentamos si hay bboxes válidos para que Albumentations no falle
            if len(bboxes) > 0:
                try:
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    
                    aug_base_name = f"{name}_aug{ext}"
                    aug_img_path = os.path.join(images_dir, aug_base_name)
                    aug_txt_path = os.path.join(labels_dir, f"{name}_aug.txt")
                    
                    cv2.imwrite(aug_img_path, aug_img)
                    write_yolo_label(aug_txt_path, aug_bboxes, aug_labels)
                except Exception as e:
                    print(f"Advertencia: No se pudo aumentar {img_path}: {e}")

def main():
    args = parse_args()
    input_dir = os.path.normpath(args.input_dir)
    if not os.path.exists(input_dir):
        print(f"Error: La carpeta '{input_dir}' no existe.")
        return
        
    print("Buscando imágenes...")
    valid_data = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                name, _ = os.path.splitext(file)
                txt_path = os.path.join(root, name + ".txt")
                
                # Aceptamos la imagen incluso si el txt no existe aún (YOLO asume background),
                # pero idealmente queremos las que tienen txt.
                valid_data.append((img_path, txt_path))
                
    if not valid_data:
        print(f"No se encontraron imágenes en la carpeta '{input_dir}'.")
        return
        
    print(f"Se encontraron {len(valid_data)} imágenes.")
    
    # Split 80/20
    train_data, val_data = train_test_split(valid_data, test_size=0.2, random_state=42)
    
    base_done_dir = input_dir + "_done"
    train_dir = os.path.join(base_done_dir, "train")
    val_dir = os.path.join(base_done_dir, "val")
    
    print("Limpiando carpetas de salida anteriores si existen...")
    if os.path.exists(base_done_dir):
        shutil.rmtree(base_done_dir)
            
    print("Generando conjunto de Entrenamiento...")
    process_and_copy(train_data, train_dir, apply_aug=args.offline_aug)
    
    print("Generando conjunto de Validación (sin aumentos)...")
    # Validación nunca se aumenta, ni siquiera en offline, para mantener una métrica pura
    process_and_copy(val_data, val_dir, apply_aug=False)
    
    # Copiar o generar cbn_dataset.yaml adaptado
    yaml_content = f"""path: {os.path.abspath(base_done_dir)}
train: train/images
val: val/images
test: test/images

names:
  0: cap
  1: empty_slot
"""
    yaml_path = os.path.join(base_done_dir, "cbn_dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"¡Dataset preparado exitosamente en '{base_done_dir}'!")
    print(f"Archivo de configuración generado: {yaml_path}")
    
    if args.offline_aug:
        print("Aumento offline aplicado con Albumentations.")
    else:
        print("Aumento online esperado (YOLO se encargará durante el entrenamiento).")

if __name__ == "__main__":
    main()
