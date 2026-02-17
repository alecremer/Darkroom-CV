import torch
from transformers import ViTModel, ViTConfig
import os

from mae.setr_pup import SetrPup

# Defina as dimensões para garantir a compatibilidade
# IMAGE_SIZE = 224 # Tamanho que você usou no pré-treinamento do MAE
# NUM_CLASSES = 2 # Ex: 0 (Fundo) e 1 (Stringing)

class SetrPupLoader:
    

    def load_mae_segmenter(self, encoder_path: str, num_classes: int, image_size: int):
        # 1. Carregar a configuração base do MAE
        config = ViTConfig.from_pretrained("facebook/vit-mae-base")

        # 2. Carregar o modelo ViT (APENAS o Encoder)
        vit_encoder = ViTModel.from_pretrained("facebook/vit-mae-base", config=config)
        
        # 3. Carregar os pesos pré-treinados do seu arquivo
        # encoder_path = "mae_checkpoints/mae_encoder_for_segmentation.pth"
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Erro: Arquivo {encoder_path} não encontrado. Execute o Passo 1 primeiro.")
        
        vit_encoder.load_state_dict(torch.load(encoder_path), strict=False)
        
        # 4. Instanciar o modelo de segmentação
        model = SetrPup(vit_encoder, num_classes, image_size)
        
        return model

# Exemplo de uso
# segmenter_model = load_mae_segmenter()
# print("Modelo MAE Segmenter carregado com sucesso!")