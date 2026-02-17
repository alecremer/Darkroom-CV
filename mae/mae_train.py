from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import nn, optim
import torch
from torchvision import datasets, transforms
from mae.yolo_dataset import YOLODetectionDataset, YOLOSegmentationDataset
from transformers import ViTMAEForPreTraining, ViTConfig
import sys
import os

# dataset_arg = sys.argv[1]
# dataset_val_arg = sys.argv[2]
# Constantes de normalização do modelo base ViT (ImageNet)
IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class MAE_Train:

    def unnormalize(self, tensor):
        """Desnormaliza o tensor de volta para a faixa [0, 1]."""
        return tensor * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN

    def prepare_image_for_save(self, tensor):
        """Converte o tensor (C, H, W) em RGB float [0, 1] e garante que não haja clamping."""
        # A desnormalização do ViT do transformers é um pouco complexa. 
        # Para o seu dataset, se você usou apenas transforms.ToTensor(), a imagem original já está entre [0, 1].
        # return tensor
        
        # Se você *não* usou normalização no seu transforms.Compose:
        # return tensor.clamp(0.0, 1.0) 

        # Se você *usou* normalização (que é o padrão para ViT):
        return self.unnormalize(tensor).clamp(0.0, 1.0)

    def save_single_mae_visualization(self, orig_img, masked_img, recon_img, epoch, idx, output_dir="reconstructions"):
        """Função auxiliar para salvar uma única imagem de comparação."""
        comparison_image = torch.cat([orig_img, masked_img, recon_img], dim=2)
        
        os.makedirs(output_dir, exist_ok=True)
        save_image(comparison_image, f'{output_dir}/epoch_{epoch}_val_{idx:03d}.png')

    def visualize_mae_step(self, model, val_loader, epoch, device, output_dir="reconstructions"):
        """
        Visualiza e salva a imagem original, a imagem mascarada e a reconstrução.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model.eval()
        all_images_processed = 0
        with torch.no_grad():
            for images_batch, _ in val_loader:
                # images_batch = torch.stack(images_batch)
                # Passa o batch pelo modelo (apenas os primeiros 4 para agilizar)
                outputs = model(pixel_values=images_batch.to(device), return_dict=True)

                # 1. Reconstrução (logits) - [batch_size, sequence_length, patch_size**2 * num_channels]
                logits = outputs.logits
                
                # 2. Máscara - [batch_size, sequence_length] (1 para masked, 0 para visible)
                mask = outputs.mask 

                # A ViTMAEForPreTraining tem um método 'unpatchify' para converter patches/tokens de volta para imagem.
                # Converte os logits de volta para o formato de imagem (C, H, W)
                y_recon = model.unpatchify(logits) # [B, C, H, W]
                
                # Converte a imagem original de volta (apenas para consistência)
                x_original = images_batch 

                # Cria a imagem Mascarada (Original + Buracos Pretos)
                # ----------------------------------------------------
                # a) Transforma a máscara (1/0) para o formato de imagem (C, H, W)
                mask_expanded = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)
                mask_img = model.unpatchify(mask_expanded)
                
                # b) Inverte a máscara (0 para manter visível, 1 para mascarar)
                # O MAE usa 1 para patches MASCARADOS, mas para visualização, queremos 0 no buraco.
                
                # Prepara a máscara para visualização: 1 onde a imagem deve ser mantida, 0 onde foi mascarada.
                # A mask do outputs.mask é 1 para MASCARADO. Queremos o inverso para multiplicar.
                mask_final = (1 - mask_img) # 1 (visível) e 0 (buraco)

                # Imagem Mascarada = Original * (1 - Máscara MAE)
                im_masked = x_original * mask_final.cpu() # Onde a máscara é 1, a imagem fica. Onde é 0 (mascarada), fica preto.
                
                # Cria a imagem Completa (Original visível + Reconstrução dos buracos)
                # ----------------------------------------------------
                # Reconstrução total = (Original * Visível) + (Reconstrução * Máscara)
                im_paste = x_original * mask_final.cpu() + y_recon.cpu() * (1 - mask_final.cpu())
                

                for idx in range(images_batch.size(0)):

                # Salvar o primeiro exemplo do batch (índice 0)
                    
                    # Prepara os tensores para salvar (desnormaliza)
                    orig_img = self.prepare_image_for_save(x_original[idx])
                    masked_img = self.prepare_image_for_save(im_masked[idx])
                    recon_img = self.prepare_image_for_save(im_paste[idx])

                    self.save_single_mae_visualization(orig_img, masked_img, recon_img, epoch, all_images_processed, output_dir)
                    all_images_processed += 1

                # Empilha as três imagens (Original, Mascarada, Reconstrução)
                # comparison_image = torch.cat([orig_img, masked_img, recon_img], dim=2)
                
                # save_image(comparison_image, f'{output_dir}/epoch_{epoch}_comparison.png')

        model.train() # Volta para o modo de treinamento


    def train(self, dataset: str, train_path: str, val_path: str, model_config: dict):

        model_dir = os.path.join(dataset, "mae_checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        encoder_save_path = os.path.join(model_dir, "mae_encoder_for_segmentation.pth")


        epochs = model_config.get('epochs', 300)
        lr = model_config.get('lr', 1e-4)
        batch_size = model_config.get('batch_size', 4)
        visualize_steps: bool = model_config.get("visualize_steps", False)
        VISUALIZATION_FREQ: int = model_config.get("visualize_frequency", 10)

        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device founded: {device}")

        image_size = 224


        # load dataset
        shuffle = True
        collate_fn = lambda x: tuple(zip(*x)) # o que???
        best_loss = float('inf')

        # transform = transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor()
        # ])

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])

        print("loading dataset...")
        # train_dataset = YOLOSegmentationDataset(dataset, image_size, transform, split="train")
        train_dataset = YOLOSegmentationDataset(dataset, image_size, transform, split=train_path)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

        # val_dataset = YOLOSegmentationDataset(dataset, image_size, transform, split="val") # Adiciona split
        val_dataset = YOLOSegmentationDataset(dataset, image_size, transform, split=val_path) # Adiciona split
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # sample_batch = next(iter(loader))
        # sample_imgs, _ = sample_batch
        # sample_imgs_batch = torch.stack(sample_imgs) # Não mover para o device ainda

    

        print("Creating model...")
        # model = timm.create_model('vit_base_patch16_224.mae', pretrained=False)
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        model = model.to(device)
        # criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        print("Starting train...")
        for epoch in range(epochs):
            model.train() # o que essa funcao faz se estou percorrendo as imagens depois?
            total_loss = 0
            num_batches = 0


            for imgs, _ in loader:
                optimizer.zero_grad() 
                imgs_batch = torch.stack(imgs).to(device)
                outputs = model(pixel_values=imgs_batch, return_dict=True)
                loss = outputs.loss
                # loss = model(imgs_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, avg_loss: {avg_loss}, loss: {loss.item():.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                encoder_state_dict = model.vit.state_dict()
                torch.save(encoder_state_dict, encoder_save_path)

            if visualize_steps and (epoch + 1) % VISUALIZATION_FREQ == 0:
                self.visualize_mae_step(
                    model=model, 
                    val_loader=val_loader, 
                    epoch=epoch + 1, 
                    device=device
                )

    # train_mae(dataset_arg, dataset_val_arg)