
import numpy as np
import torchvision.transforms as T
import cv2
import torch
from setr_pup.setr_pup_loader import SetrPupLoader


IMAGE_SIZE = 224 # Tamanho que você usou no pré-treinamento do MAE


class SetrPupInference:

    def load_vitmae_seg(self, encoder_path: str, head_path: str, num_classes: int, image_size: int = 224):

        loader = SetrPupLoader()
        model = loader.load_mae_segmenter(encoder_path, num_classes, image_size).to(self.device)
        checkpoint = torch.load(head_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model
    
    def predict(self, frame, confidence: float):
        h, w = frame.shape[:2]

        # RGB e PIL
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = T.ToPILImage()(img_rgb)

        # Transformar para tensor normalizado
        x = self.transform(img_pil).unsqueeze(0).to(self.device)  # [1, C, H, W]

        # Forward
        with torch.no_grad():
            outputs = self.model(x)  # [1, NUM_CLASSES, H, W]
            probs = torch.softmax(outputs, dim=1) # for confidence
            conf_map_tensor, pred_mask_tensor = torch.max(probs, dim=1)
            
            # pred_mask_small = outputs.argmax(dim=1)[0].cpu().numpy()  # [H, W]
            # pred_mask = cv2.resize(pred_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

            conf_map_small =conf_map_tensor[0].cpu().numpy()  # [H, W]
            pred_mask_small = pred_mask_tensor[0].cpu().numpy().astype(np.uint8)  # [H, W]

        pred_mask = cv2.resize(pred_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        conf_map = cv2.resize(conf_map_small, (w, h), interpolation=cv2.INTER_LINEAR)

        pred_mask_filtered = pred_mask.copy()
        pred_mask_filtered[conf_map < confidence] = 0

        classes_found = np.unique(pred_mask_filtered)
        results_confidence = {}

        # for every class
        for cls in classes_found:
            if cls == 0: continue # skip background

            # mean of pixels of mask class
            mean_conf = np.mean(conf_map[pred_mask_filtered == cls])
            results_confidence[int(cls)] = float(mean_conf)

        # return pred_mask, results_confidence
        return pred_mask
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
