import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt
import os

# A grande estrela do nosso novo pipeline
import segmentation_models_pytorch as smp

from setr_pup.pup_head_train import SegmentationDataset

# Presumindo que SegmentationDataset e DatasetPathMapper já estão importados aqui no seu arquivo

class SwinUnetTrain:
    def save_model_checkpoint(self, model, path, epoch, loss):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, path)

    def unnormalize_tensor(self, tensor):
        """Reverte a normalização do ImageNet."""
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        mean = IMAGENET_MEAN.to(tensor.device)
        std = IMAGENET_STD.to(tensor.device)
        return tensor * std + mean
    
    def visualize_segmentation_val(self, model, val_loader, save_path, epoch, device):
        """
        Avalia e salva a visualização de TODAS as imagens do val_loader, 
        sobrepondo GT e Predição na Imagem Original.
        """
        model.eval()
        os.makedirs(save_path, exist_ok=True)
        
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

                    # 1. GROUND TRUTH
                    axes[0].imshow(img_pil)
                    overlay_gt = np.ma.masked_where(mask_gt == 0, mask_gt)
                    axes[0].imshow(overlay_gt, cmap=cmap_reds, vmin=0, vmax=1, alpha=0.5)
                    axes[0].set_title("1. Ground Truth (Sobreposto)")
                    axes[0].axis('off')

                    # 2. PREDIÇÃO
                    axes[1].imshow(img_pil)
                    overlay_pred = np.ma.masked_where(mask_pred_np == 0, mask_pred_np)
                    axes[1].imshow(overlay_pred, cmap=cmap_reds, vmin=0, vmax=1, alpha=0.5)
                    axes[1].set_title("2. Predição (Sobreposto)")
                    axes[1].axis('off')
                    
                    # 3. IMAGEM ORIGINAL
                    axes[2].imshow(img_pil)
                    axes[2].set_title("3. Imagem Original (Referência)")
                    axes[2].axis('off')

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(os.path.join(save_path, f"epoch_{epoch:03d}_val_{all_images_processed:03d}.png"))
                    plt.close(fig)
                    
                    all_images_processed += 1
                    
        model.train() 

    def train(self, dataset_path: str, train_path: str, val_path: str, config: dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        FINE_TUNE_EPOCHS = config.get("head_epochs", 100)
        visualize_steps: bool = config.get("head_visualize_steps", False)
        visualize_epochs = config.get("head_visualize_frequency", 10)
        image_size = config.get("image_size", 224)
        
        FT_LR_HEAD = config.get("head_lr", 1e-4)
        FT_LR_ENCODER = config.get("backbone_finetuning_lr", 1e-5) # Menor para não destruir os pesos do ImageNet
        
        try:
            num_classes = config["num_classes"]
        except KeyError as e:
            raise KeyError(f"Error: {e}")

        print("Inicializando Swin-UNet via Segmentation Models PyTorch...")
        # ==========================================
        # CRIAÇÃO DO MODELO COM SMP
        # ==========================================
        model = smp.Unet(
            encoder_name="tu-swin_base_patch4_window7_224", # Backbone do timm
            encoder_weights="imagenet",                     # Pesos pré-treinados vitais!
            in_channels=3,
            classes=num_classes,                            # Ex: 2 (Fundo e Stringing)
        ).to(device)

        train_dataset = SegmentationDataset(dataset_path, image_size, split=train_path)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        val_dataset = SegmentationDataset(dataset_path, image_size, split=val_path)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        criterion = nn.CrossEntropyLoss().to(device)
        
        # ==========================================
        # SEPARAÇÃO DE LEARNING RATES (Encoder vs Decoder)
        # ==========================================
        # Agrupamos tudo que não é encoder (decoder + segmentation head) para aprender mais rápido
        head_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
        
        param_groups = [
            {'params': model.encoder.parameters(), 'lr': FT_LR_ENCODER},
            {'params': head_params, 'lr': FT_LR_HEAD}
        ]
        
        optimizer = optim.AdamW(param_groups)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[FT_LR_ENCODER, FT_LR_HEAD], 
            steps_per_epoch=len(train_loader), 
            epochs=FINE_TUNE_EPOCHS
        )

        BEST_LOSS = float('inf')
        CHECKPOINT_DIR = os.path.join(dataset_path, "swin_unet_checkpoints")
        VISUALS_DIR = os.path.join(dataset_path, "swin_unet_visuals")

        print(f"Start Fine-Tuning")
        print(f"Backbone LR: {FT_LR_ENCODER}")
        print(f"Head LR: {FT_LR_HEAD}")
        
        for epoch in range(FINE_TUNE_EPOCHS):
            model.train()
            running_loss = 0.0

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

            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{FINE_TUNE_EPOCHS}], Loss: {epoch_loss:.4f}")

            if epoch_loss < BEST_LOSS:
                BEST_LOSS = epoch_loss
                self.save_model_checkpoint(model, 
                                    os.path.join(CHECKPOINT_DIR, "best_model.pth"), 
                                    epoch + 1, 
                                    BEST_LOSS)
                
            if visualize_steps and ((epoch + 1) % visualize_epochs == 0 or epoch == FINE_TUNE_EPOCHS - 1):
                self.visualize_segmentation_val(model, val_loader, VISUALS_DIR, epoch + 1, device)

        print("\nTrain done!")
        print(f"Best Loss: {BEST_LOSS:.4f}")