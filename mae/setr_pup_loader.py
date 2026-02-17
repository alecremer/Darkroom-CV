import torch
from torch import nn
from transformers import ViTModel, ViTConfig
import os

# Defina as dimensões para garantir a compatibilidade
IMAGE_SIZE = 224 # Tamanho que você usou no pré-treinamento do MAE
NUM_CLASSES = 2 # Ex: 0 (Fundo) e 1 (Stringing)

class SetrPupLoader(nn.Module):
    """
    Arquitetura de Segmentação usando o MAE Encoder como backbone.
    Faz a conversão dos tokens 1D do ViT para um feature map 2D e o decodifica.
    
    """
    
    
    def __init__(self, encoder, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.patch_size = encoder.config.patch_size # Ex: 16
        self.embed_dim = encoder.config.hidden_size # Ex: 768 (para ViT-Base)
        

        # O tamanho da grade de patches de saída (ex: 224/16 = 14)
        self.grid_size = IMAGE_SIZE // self.patch_size
        
        # HEAD DE SEGMENTAÇÃO (Decoder para upsampling)
        
        # 1. Projeção: Reduz a dimensão do canal após a conversão 1D -> 2D
        self.proj_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # 2. Sequência de Transposed Convolutions para reconstruir a resolução
        # 14x14 -> 28x28 (stride=2)
        self.up_sample1 = nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=4, stride=2, padding=1)
        # 28x28 -> 56x56
        self.up_sample2 = nn.ConvTranspose2d(self.embed_dim // 4, self.embed_dim // 8, kernel_size=4, stride=2, padding=1)
        # 56x56 -> 112x112
        self.up_sample3 = nn.ConvTranspose2d(self.embed_dim // 8, self.embed_dim // 16, kernel_size=4, stride=2, padding=1)
        
        # 3. Output Final (112x112 -> 224x224): Gera os logits da máscara para as classes
        self.segmentation_head = nn.ConvTranspose2d(self.embed_dim // 16, num_classes, kernel_size=4, stride=2, padding=1)
    
    
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Passar pelo Encoder ViT (com pesos do MAE)
        # Saída: [B, num_tokens (197), embed_dim (768)]
        encoder_output = self.encoder(x).last_hidden_state
        
        # 2. Descartar o CLS token (o primeiro) e manter apenas os tokens de patch
        patch_tokens = encoder_output[:, 1:, :] # [B, num_patches (196), embed_dim]
        
        # 3. Converter a sequência 1D de volta para Feature Map 2D
        # [B, 196, 768] -> [B, 768, 196] -> reshape -> [B, 768, 14, 14]
        x = patch_tokens.transpose(1, 2)
        x = x.reshape(B, self.embed_dim, self.grid_size, self.grid_size)

        # 4. Passar pela Head de Segmentação (Decoder)
        x = self.proj_head(x) 

        x = nn.GELU()(self.up_sample1(x))
        x = nn.GELU()(self.up_sample2(x))
        x = nn.GELU()(self.up_sample3(x))
        
        mask_logits = self.segmentation_head(x)
        
        # mask_logits tem o tamanho [B, 2, 224, 224]
        return mask_logits

    def load_mae_segmenter(self, encoder_path: str = "mae_checkpoints/mae_encoder_for_segmentation.pth"):
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
        model = SetrPupLoader(encoder=vit_encoder)
        
        return model

# Exemplo de uso
# segmenter_model = load_mae_segmenter()
# print("Modelo MAE Segmenter carregado com sucesso!")