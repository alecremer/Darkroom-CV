import os
import torch
import urllib.request
import tarfile
import torchvision
import zipfile
import cv2
import numpy as np





def mask_to_yolo_seg(mask_path, output_txt_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return

    height, width = mask.shape[:2]
    obj_ids = np.unique(mask)[1:] 

    polygons = []
    for obj_id in obj_ids:
        binary_mask = np.where(mask == obj_id, 255, 0).astype(np.uint8)
        
        res = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[-2]

        for cnt in contours:
            if len(cnt) < 3: continue 
            poly = cnt.reshape(-1, 2).astype(float)
            poly[:, 0] /= width
            poly[:, 1] /= height
            polygons.append(poly.reshape(-1).tolist())

    with open(output_txt_path, 'w') as f:
        for poly in polygons:
            poly_str = " ".join([f"{coord:.6f}" for coord in poly])
            f.write(f"{class_id} {poly_str}\n")



def download_demo_data(path="./demo"):
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = os.path.join(path, "PennFudanPed.zip")
    extract_path = os.path.join(path, "PennFudanPed")

    if not os.path.exists(extract_path):
        os.makedirs(path, exist_ok=True)
        
        print(f"Download dataset demo (Penn-Fudan)... (~50MB).")
        try:
            urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path)
            
            os.remove(zip_path)
            print(f"Data path: {extract_path}")

            mask_dir = os.path.join(extract_path, "PedMasks")
            label_dir = os.path.join(extract_path, "seg_labels") 
            os.makedirs(label_dir, exist_ok=True)

            print("Converting masks to YOLO segmentation...")
            
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
            
            for mask_file in mask_files:
                mask_full_path = os.path.join(mask_dir, mask_file)
                txt_name = mask_file.replace("_mask.png", ".txt")
                output_txt_path = os.path.join(label_dir, txt_name)
                
                mask_to_yolo_seg(mask_full_path, output_txt_path)

            print(f"Done! Images path: {os.path.join(extract_path, 'PNGImages')}")
            print(f"YOLO labels path: {label_dir}")
            
        except Exception as e:
            print(f"Download error: {e}")
    else:
        print("Demo dataset setup done.")

# def download_weights(model_url, save_path):
#     if not os.path.exists(save_path):
#         print(f"Pesos não encontrados em {save_path}. Baixando modelo treinado...")
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         urllib.request.urlretrieve(model_url, save_path)
#         print("Download concluído.")
#     else:
#         print("Pesos do modelo carregados localmente.")

# # No código da demo:
# MODEL_URL = "https://seu-link-direto.com/meu_modelo.pt"
# WEIGHTS_PATH = "weights/best_model.pt"

# if args.demo:
#     download_weights(MODEL_URL, WEIGHTS_PATH)
#     model = torch.load(WEIGHTS_PATH)
#     # ... resto da demo