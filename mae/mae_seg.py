
import torchvision.transforms as T
import cv2
import torch
from .segmentation_model import load_mae_segmenter


IMAGE_SIZE = 224 # Tamanho que você usou no pré-treinamento do MAE


class MAE_Seg:

    def load_vitmae_seg(self, encoder_path, head_path):

        model = load_mae_segmenter(encoder_path).to(self.device)
        checkpoint = torch.load(head_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model
    
    def predict_from_image(self, frame):
        h, w = frame.shape[:2]

        # RGB e PIL
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = T.ToPILImage()(img_rgb)

        # Transformar para tensor normalizado
        x = self.transform(img_pil).unsqueeze(0).to(self.device)  # [1, C, H, W]

        # Forward
        outputs = self.model(x)  # [1, NUM_CLASSES, H, W]
        pred_mask_small = outputs.argmax(dim=1)[0].cpu().numpy()  # [H, W]
        pred_mask = cv2.resize(pred_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

        return pred_mask
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
