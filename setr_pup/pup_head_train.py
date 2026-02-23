import torch
from torch import nn, optim, tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import cv2
from torchvision.transforms.functional import to_pil_image

# Importar o modelo definido no Passo 2
from setr_pup.dataset_path_mapper import DatasetPathMapper
from setr_pup.setr_pup_loader import SetrPupLoader 

from PIL import Image
import matplotlib.pyplot as plt


    
class SegmentationDataset(Dataset):
    """
    Carrega imagens e reconstrói máscaras a partir de arquivos YOLO (.txt) com polígonos.
    """
    def __init__(self, dataset_path, image_size, split="train"):
        
        image_path, label_path = DatasetPathMapper.dataset_to_images_and_labels(dataset_path, split)

        # self.images_dir = os.path.join(dataset_path, split, "images")
        # self.labels_dir = os.path.join(dataset_path, split, "labels")

        self.images_dir = image_path
        self.labels_dir = label_path
        
        # Filtra apenas arquivos de imagem
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                   if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.image_size = image_size
        
        # Transformação para a Imagem (ViT/MAE espera Normalização ImageNet)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # O nome do label .txt é o mesmo que o nome do arquivo de imagem (sem a extensão)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        # 1. Carregar e Obter Dimensões Originais da Imagem
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV lê BGR, converte para RGB
        H_orig, W_orig = img.shape[:2]
        
        # 2. Aplicar Transformação na Imagem (do CV2 para PIL/Tensor)
        img_pil = Image.fromarray(img)
        img_tensor = self.transform(img_pil)

        # 3. Reconstruir a Máscara (Ground Truth)
        # Inicializa a máscara binária (0 = fundo) no tamanho original
        mask = np.zeros((H_orig, W_orig), dtype=np.uint8)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    
                    # O primeiro valor é o índice da classe (0 para stringing, no seu caso)
                    cls_idx = int(vals[0]) 
                    # Coordenadas do polígono normalizadas (resto da linha)
                    poly_coords_norm = vals[1:] 

                    # A classe de stringing (assumindo cls_idx=0 no YOLO) deve ser 1 no modelo.
                    if cls_idx == 0: 
                        mask_value = 1 # Classe de Interesse (Stringing)
                    # A classe de underextrusion (assumindo cls_idx=1 ou outro) e outras
                    # que não interessam devem ser mapeadas para 0 (Fundo).
                    else:
                        # Todas as outras classes (incluindo underextrusion) são ignoradas.
                        continue

                    # Converte coordenadas normalizadas para coordenadas de pixel originais
                    poly = np.array(poly_coords_norm, dtype=np.float32).reshape(-1, 2)
                    poly[:, 0] *= W_orig
                    poly[:, 1] *= H_orig
                    
                    # Desenha o polígono na máscara com o valor da classe (cls_idx + 1)
                    # Se você tem apenas 1 classe de defeito (stringing), o valor da máscara será 1.
                    # O fundo (background) será 0.
                    cv2.fillPoly(mask, [poly.astype(np.int32)], color=mask_value) 
                    
        
        # 4. Redimensiona a máscara para o tamanho do modelo (224x224)
        mask_resized = cv2.resize(mask, (self.image_size, self.image_size), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # 5. Converte para tensor Long: CrossEntropyLoss espera [H, W] do tipo Long
        mask_label = torch.tensor(mask_resized, dtype=torch.long)
        
        return img_tensor, mask_label



class PupHeadTrain:
    def save_model_checkpoint(self, model, path, epoch, loss):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, path)

    

    def unnormalize_tensor(self, tensor):
        """Reverte a normalização do ImageNet."""
        
        # Média e Desvio Padrão do ImageNet (para desnormalização)
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Garante que o tensor e o mean/std estejam no mesmo dispositivo para operação
        mean = IMAGENET_MEAN.to(tensor.device)
        std = IMAGENET_STD.to(tensor.device)
        return tensor * std + mean
    
    def visualize_segmentation_val(self, model, val_loader, save_path, epoch, device):
        """
        Avalia e salva a visualização de TODAS as imagens do val_loader, 
        sobrepondo GT e Predição na Imagem Original.
        """
        model.eval()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        cmap_reds = plt.cm.get_cmap('Blues', 2) # Fundo é 0, Stringing é 1
        
        all_images_processed = 0

        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # Outputs são [B, C, H, W]. Pegamos a classe com maior probabilidade (argmax)
                mask_pred = outputs.argmax(dim=1) # [B, H, W]

                for idx in range(images.size(0)):
                    
                    # Desnormalizar a imagem original
                    img_tensor = self.unnormalize_tensor(images[idx]).cpu().clamp(0.0, 1.0)
                    img_pil = to_pil_image(img_tensor)
                    
                    # Máscaras
                    mask_gt = masks[idx].cpu().numpy()
                    mask_pred_np = mask_pred[idx].cpu().numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(f"Epoch {epoch} | Imagem {all_images_processed+1}", fontsize=16)

                    # 1. GROUND TRUTH SOBRE IMAGEM ORIGINAL (Translúcido)
                    axes[0].imshow(img_pil)
                    # Cria um overlay colorido apenas para a classe 1 (Stringing)
                    overlay_gt = np.ma.masked_where(mask_gt == 0, mask_gt)
                    axes[0].imshow(overlay_gt, cmap=cmap_reds, vmin=0, vmax=1, alpha=0.5) # alpha=0.5 para transparência
                    axes[0].set_title("1. Ground Truth (Sobreposto)")
                    axes[0].axis('off')

                    # 2. PREDIÇÃO SOBRE IMAGEM ORIGINAL (Translúcido)
                    axes[1].imshow(img_pil)
                    overlay_pred = np.ma.masked_where(mask_pred_np == 0, mask_pred_np)
                    axes[1].imshow(overlay_pred, cmap=cmap_reds, vmin=0, vmax=1, alpha=0.5)
                    axes[1].set_title("2. Predição (Sobreposto)")
                    axes[1].axis('off')
                    
                    # 3. IMAGEM ORIGINAL (Sem sobreposição, como referência)
                    axes[2].imshow(img_pil)
                    axes[2].set_title("3. Imagem Original (Referência)")
                    axes[2].axis('off')


                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(os.path.join(save_path, f"epoch_{epoch:03d}_val_{all_images_processed:03d}.png"))
                    plt.close(fig)
                    
                    all_images_processed += 1
                    
        model.train() 

    def train(self, dataset_path: str, train_path: str, val_path: str, config: dict):

        # head -> training
        # backbone -> fine tuning

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            encoder_path = config["encoder_path"]
        except KeyError as e:
            raise KeyError(f"Error: {e}")
        
        FINE_TUNE_EPOCHS = config.get("head_epochs", 100)
        visualize_steps: bool = config.get("head_visualize_steps", False)
        visualize_epochs = config.get("head_visualize_frequency", 10)
        image_size = config.get("image_size", 224)
        FT_LR = 1e-4 
        FT_LR_HEAD = config.get("head_lr", 1e-4)
        FT_LR_ENCODER = config.get("backbone_finetuning_lr", 1e-5)
        try:
            num_classes = config["num_classes"]
        except KeyError as e:
            raise KeyError(f"Error: {e}")
        
        

        loader = SetrPupLoader()
        model = loader.load_mae_segmenter(encoder_path, num_classes, image_size).to(device)

        train_dataset = SegmentationDataset(dataset_path, image_size, split=train_path)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        val_dataset = SegmentationDataset(dataset_path, image_size, split=val_path)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        criterion = nn.CrossEntropyLoss().to(device)
        
        

        param_groups = [
            {'params': model.encoder.parameters(), 'lr': FT_LR_ENCODER},
            {'params': [p for name, p in model.named_parameters() if 'encoder' not in name], 'lr': FT_LR_HEAD}
        ]
        
        # optimizer = optim.AdamW(model.parameters(), lr=FT_LR)
        optimizer = optim.AdamW(param_groups)

        # Opcional: Scheduler para decaimento do LR
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FT_LR, 
        #                                           steps_per_epoch=len(train_loader), 
        #                                           epochs=30)
        

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=[FT_LR_ENCODER, FT_LR_HEAD], 
                                                steps_per_epoch=len(train_loader), 
                                                epochs=FINE_TUNE_EPOCHS)
        

        # Definições para Checkpoint e Visualização
        BEST_LOSS = float('inf')
        CHECKPOINT_DIR = os.path.join(dataset_path, "setrpup_checkpoints")
        VISUALS_DIR = os.path.join(dataset_path, "setrpup_visuals")

        # 4. Loop de Treinamento
        print(f"Start Fine-Tuning")
        print(f"backbone lr: {FT_LR_ENCODER}")
        print(f"Head lr: {FT_LR_HEAD}")
        
        for epoch in range(FINE_TUNE_EPOCHS):
            model.train()
            running_loss = 0.0
            
            last_images, last_outputs, last_masks = None, None, None

            for i, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                
                # Forward: Saída é [B, NUM_CLASSES, H, W]
                outputs = model(images)
                
                # Loss
                loss = criterion(outputs, masks)
                
                # Backward
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item() * images.size(0)
                last_images, last_outputs, last_masks = images.detach().cpu(), outputs.detach().cpu(), masks.detach().cpu()

            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{FINE_TUNE_EPOCHS}], Loss: {epoch_loss:.4f}")

            if epoch_loss < BEST_LOSS:
                BEST_LOSS = epoch_loss
                # Salva o modelo com a melhor perda até agora
                self.save_model_checkpoint(model, 
                                    os.path.join(CHECKPOINT_DIR, "best_model.pth"), 
                                    epoch + 1, 
                                    BEST_LOSS)
                
            # if (epoch + 1) % 5 == 0 or epoch == FINE_TUNE_EPOCHS - 1:
            #      model.eval() # Não é estritamente necessário, mas boa prática
            #      with torch.no_grad():
            #          visualize_segmentation_step(last_images, last_outputs, last_masks, VISUALS_DIR, epoch + 1)
            #      print(f" -> Imagem de visualização salva para a Epoch {epoch+1}")
            #      model.train()
            if visualize_steps and (epoch + 1) % visualize_epochs == 0 or epoch == FINE_TUNE_EPOCHS - 1:
                self.visualize_segmentation_val(model, val_loader, VISUALS_DIR, epoch + 1, device) # <-- Nova função

        print("\nTrain done!")
        print(f"Best Loss: {BEST_LOSS:.4f}")
        