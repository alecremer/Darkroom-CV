import os

import torch.optim as optim
import torch
import torch.nn as nn
from torchvision.utils import save_image

from swin_mae.swin_mae import SwinMae
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from setr_pup.yolo_dataset import YOLOSegmentationDataset

IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class SwinMaeTrain:

    def unnormalize(self, tensor):
        return tensor * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN

    def prepare_image_for_save(self, tensor):
        return self.unnormalize(tensor).clamp(0.0, 1.0)

    def save_single_mae_visualization(self, orig_img, masked_img, recon_img, epoch, idx, output_dir="reconstructions"):
        comparison_image = torch.cat([orig_img, masked_img, recon_img], dim=2)
        os.makedirs(output_dir, exist_ok=True)
        save_image(comparison_image, f'{output_dir}/epoch_{epoch}_val_{idx:03d}.png')

    def visualize_mae_step(self, model, val_loader, epoch, device, output_dir="reconstructions"):
        os.makedirs(output_dir, exist_ok=True)
        
        model.eval()
        all_images_processed = 0

        with torch.no_grad():
            for images_batch, _ in val_loader:

                if isinstance(images_batch, tuple) or isinstance(images_batch, list):
                    images_batch = torch.stack(images_batch)
                images_batch = images_batch.to(device)

                reconstructed, mask = model(images_batch)

                # Move everything to CPU for visualization math to save VRAM
                x_original = images_batch.cpu()
                y_recon = reconstructed.cpu()
                mask_cpu = mask.cpu() # Shape: (B, 1, H, W). 1 is masked, 0 is visible.

                # 3. Create Masked Image (Original + Black Holes)
                visible_mask = 1 - mask_cpu # 1 is visible, 0 is black hole
                im_masked = x_original * visible_mask 
                
                # 4. Create Paste Image (Original visible + Reconstructed holes)
                im_paste = (x_original * visible_mask) + (y_recon * mask_cpu)

                # 5. Save visualizations
                for idx in range(images_batch.size(0)):
                    orig_img = self.prepare_image_for_save(x_original[idx])
                    masked_img = self.prepare_image_for_save(im_masked[idx])
                    recon_img = self.prepare_image_for_save(im_paste[idx])

                    self.save_single_mae_visualization(orig_img, masked_img, recon_img, epoch, all_images_processed, output_dir)
                    all_images_processed += 1
                    
                    # Optional: Break early if you only want to visualize a few images per epoch
                    if all_images_processed >= 8: 
                        break
                if all_images_processed >= 8:
                    break

        model.train()

    def train(self, dataset: str, train_path: str, val_path: str, model_config):

        num_epochs = model_config.get('epochs', 300)
        batch_size = model_config.get('batch_size', 4)
        lr = model_config.get('lr', 1e-4)
        visualize_steps: bool = model_config.get("visualize_steps", False)
        VISUALIZATION_FREQ: int = model_config.get("visualize_frequency", 10)
        visualize_dir = os.path.join(dataset, "swin_mae_reconstructions")

        model_dir = os.path.join(dataset, "swin_mae_checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        encoder_save_path = os.path.join(model_dir, "swin_mae_encoder_for_segmentation.pth")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_size = 224

        model = SwinMae().cuda()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        criterion = nn.L1Loss(reduction='none') # 'none' allows us to multiply by the mask later

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])

        shuffle = True
        collate_fn = lambda x: tuple(zip(*x)) 
        best_loss = float('inf')

        print("Loading datasets...")
        train_dataset = YOLOSegmentationDataset(dataset, image_size, transform, split=train_path)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

        val_dataset = YOLOSegmentationDataset(dataset, image_size, transform, split=val_path) # Adiciona split
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = model.to(device)

        print("Starting train...")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0

            for images, _ in loader:
                optimizer.zero_grad()

                imgs_batch = torch.stack(images).to(device)
                reconstructed, mask = model(imgs_batch)
                
                # Calculate L1 Loss only on the masked pixels
                loss_matrix = criterion(reconstructed, imgs_batch)
                loss = (loss_matrix * mask).sum() / (mask.sum() + 1e-5)
                
                # Backprop
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, avg_loss: {avg_loss}, loss: {loss.item():.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                encoder_state_dict = model.encoder.state_dict()
                torch.save(encoder_state_dict, encoder_save_path)

            if visualize_steps and (epoch + 1) % VISUALIZATION_FREQ == 0:
                self.visualize_mae_step(
                    model=model, 
                    val_loader=val_loader, 
                    epoch=epoch + 1, 
                    device=device,
                    output_dir=visualize_dir
                )

        # Save the full model after pre-training
        torch.save(model.state_dict(), os.path.join(model_dir, "swin_mae_pretrained_full.pth"))