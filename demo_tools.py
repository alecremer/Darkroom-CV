import os
import torch
import urllib.request
import tarfile
import torchvision
import zipfile

def download_demo_data(path="./demo"):
    # Link direto e estável do Penn-Fudan
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
            
            # Remove o zip para limpar a casa
            os.remove(zip_path)
            print(f"Data path: {extract_path}")
            
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