import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinMae(nn.Module):
    def __init__(self, encoder_name='swin_base_patch4_window7_224', mask_ratio=0.6, mask_patch_size=32):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        
        # 1. The Backbone (Encoder)
        # features_only=True ensures it outputs the 4 scales we will need later for the FPN
        self.encoder = timm.create_model(encoder_name, pretrained=True, features_only=True)
        
        # The last scale of a Swin-Base has 1024 channels
        encoder_out_dim = 1024 
        
        # 2. The Learnable Mask Token
        # We apply this directly to the input image pixels where the mask is True
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        # 3. The Lightweight Decoder
        # Projects the deep 1024-d features back to pixel space using a 1x1 Conv + PixelShuffle
        # 32 is the total spatial reduction factor of the Swin architecture
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_out_dim, 3 * (32 ** 2), kernel_size=1),
            nn.PixelShuffle(32)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- INTERNAL MASK GENERATION ---
        grid_h = H // self.mask_patch_size
        grid_w = W // self.mask_patch_size
        num_patches = grid_h * grid_w
        num_mask = int(self.mask_ratio * num_patches)
        
        # Create random indices for masking
        noise = torch.rand(B, num_patches, device=x.device)
        mask_indices = noise.argsort(dim=1) < num_mask # Shape: (B, num_patches)
        
        # Reshape mask to grid and upsample to original image resolution
        mask_grid = mask_indices.view(B, 1, grid_h, grid_w).float()
        mask_upsampled = F.interpolate(mask_grid, size=(H, W), mode='nearest')
        
        # --- APPLY MASK ---
        # Replace original pixels with the learnable token where mask is 1
        x_masked = x * (1 - mask_upsampled) + self.mask_token * mask_upsampled
        
        # --- ENCODE & DECODE ---
        features = self.encoder(x_masked)
        last_feature_map = features[-1] # Take the deepest feature map
        last_feature_map = last_feature_map.permute(0, 3, 1, 2)
        reconstructed = self.decoder(last_feature_map)
        
        # We return the upsampled mask as well to compute the loss only on masked pixels
        return reconstructed, mask_upsampled