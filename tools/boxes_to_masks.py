# import cv2
# import numpy as np
# import os
# from dataclasses import dataclass

# @dataclass
# class BoxesAndClass:
#     cls: str
#     boxes: list[float]

# def boxes_to_masks(image_shape, boxes):
#     """
#     Converte bounding boxes em máscara retangular.

#     Args:
#         image_shape (tuple): (height, width) da imagem
#         boxes (list ou np.array): cada box como [x_min, y_min, x_max, y_max]

#     Returns:
#         mask (np.array): máscara binária, 1 dentro da box, 0 fora
#     """
#     mask = np.zeros(image_shape[:2], dtype=np.uint8)  # máscara 0
#     for box in boxes:
#         x_min, y_min, x_max, y_max = map(int, box)
#         cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color=1, thickness=-1)  # -1 preenche
#     return mask


# path = "/home/ale/Downloads/PI2/dataset/train_selected"
# imgs_path = path + "/images"
# labels_path = path + "/labels"

# label_list = [
#         f for f in os.listdir(labels_path)
#         if os.path.splitext(f)[1].lower() in {".txt"}
#     ]

# for label in label_list:
#     filename = label
#     boxes_and_classes: list[BoxesAndClass] = []
#     with open(labels_path + "/" + filename) as f:
#         if filename != "classes.txt": 
#             data = f.readlines()
#             yolo_boxes = []
#             for d in data:
#                 (c, x_center, y_center, w, h) = d.strip().split(" ")
                
#                 x_center = float(x_center)
#                 y_center = float(y_center)
#                 w = float(w)
#                 h = float(h)

#                 x1 = x_center - w/2
#                 x2 = x_center + w/2
#                 y1 = x_center - h/2
#                 y2 = x_center + h/2

#                 box_and_class = BoxesAndClass(c, [x1, y1, x2, y2])

#                 boxes_and_classes.append(box_and_class)
    
#     img_path = os.path.join(imgs_path, filename.split(".")[0] + ".jpg")
#     if not os.path.exists(img_path):
#         img_path = os.path.join(imgs_path, filename.split(".")[0] + ".png")

#     img = cv2.imread(img_path)

#     boxes = [bc.boxes for bc in boxes_and_classes]
#     mask = boxes_to_masks(img.shape, boxes)

#     save_dir = os.path.join(labels_path, "masks", "labels")
#     os.makedirs(save_dir, exist_ok=True) 
#     file_path = os.path.join(save_dir, label)

#     with open(file_path, "w") as f:
#         for m, bc in zip(mask, boxes_and_classes):
#             f.write(f"{bc.cls} {m}")
#             f.write("\n")
#     cv2.imwrite(labels_path + f"/masks/images_with_masks/" + label, mask*255)

    

import cv2
import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BoxesAndClass:
    cls: int  # classe como inteiro
    boxes: list[float]  # [x_min, y_min, x_max, y_max] pixels

def yolo_to_box(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w / 2) * img_w
    x2 = (x_center + w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    y2 = (y_center + h / 2) * img_h
    return [x1, y1, x2, y2]

def boxes_to_mask(image_shape, boxes_and_classes):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # 0=fundo
    for b in boxes_and_classes:
        x_min, y_min, x_max, y_max = map(int, b.boxes)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color=b.cls + 1, thickness=-1)
    return mask

# Caminhos
path = "/home/ale/Downloads/PI2/dataset/train_selected"
imgs_path = os.path.join(path, "images")
labels_path = os.path.join(path, "labels")

# Pastas de saída
masks_dir = os.path.join(labels_path, "masks")
images_masked_dir = os.path.join(masks_dir, "images_with_masks")
os.makedirs(masks_dir, exist_ok=True)
os.makedirs(images_masked_dir, exist_ok=True)

# Processa cada label
label_list = [f for f in os.listdir(labels_path) if f.endswith(".txt") and f != "classes.txt"]
for label_file in label_list:
    base_name = os.path.splitext(label_file)[0]
    img_path_jpg = os.path.join(imgs_path, base_name + ".jpg")
    img_path_png = os.path.join(imgs_path, base_name + ".png")

    img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
    if not os.path.exists(img_path):
        print(f"Imagem não encontrada para {label_file}, pulando...")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Falha ao abrir imagem {img_path}, pulando...")
        continue

    h, w = img.shape[:2]
    boxes_and_classes = []

    with open(os.path.join(labels_path, label_file)) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_c, y_c, bw, bh = map(float, parts)
            box_px = yolo_to_box(x_c, y_c, bw, bh, w, h)
            boxes_and_classes.append(BoxesAndClass(int(cls), box_px))

    mask = boxes_to_mask(img.shape, boxes_and_classes)

    # Salva máscara para treinamento UNet/DeepLab
    mask_path = os.path.join(masks_dir, base_name + ".png")
    cv2.imwrite(mask_path, mask)

    # Salva imagem com máscara sobreposta (visualização)
    color_map = np.array([
        [0, 0, 0],        # fundo
        [0, 0, 255],      # classe 1
        [0, 255, 0],      # classe 2
        [255, 0, 0],      # classe 3
        [0, 255, 255],    # classe 4
        [255, 0, 255],    # classe 5
        [255, 255, 0]     # classe 6
    ], dtype=np.uint8)
    overlay_color = color_map[mask]
    img_overlay = cv2.addWeighted(img, 0.7, overlay_color, 0.3, 0)
    cv2.imwrite(os.path.join(images_masked_dir, base_name + ".png"), img_overlay)

    print(f"Processado {label_file} -> máscara {mask_path}")
