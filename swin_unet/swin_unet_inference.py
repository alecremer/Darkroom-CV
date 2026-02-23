import numpy as np
import torchvision.transforms as T
import cv2
import torch
import segmentation_models_pytorch as smp

IMAGE_SIZE = 224

class SwinUnetInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Exatamente os mesmos transforms do SETR-PUP (padrão ImageNet)
        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.model = None

    def load_swin_unet(self, model_path: str, num_classes: int, image_size: int = 224):
        """
        Instancia a arquitetura smp.Unet e carrega os pesos do treinamento.
        """
        # 1. Recriar a exata mesma arquitetura usada no treino
        model = smp.Unet(
            encoder_name="tu-swin_base_patch4_window7_224",
            encoder_weights=None, # None porque vamos carregar os nossos próprios pesos treinados
            in_channels=3,
            classes=num_classes,
        ).to(self.device)

        # 2. Carregar o checkpoint salvo pelo SwinUnetTrain
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # 3. Injetar os pesos na arquitetura
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. Modo de avaliação
        model.eval()
        self.model = model

    def predict(self, frame, confidence: float):
        """
        Realiza a inferência em um frame BGR (padrão OpenCV).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_swin_unet first.")

        h, w = frame.shape[:2]

        # OpenCV BGR -> RGB e PIL
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = T.ToPILImage()(img_rgb)

        # Transformar para tensor normalizado [1, C, H, W]
        x = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Forward
        with torch.no_grad():
            outputs = self.model(x)  # [1, NUM_CLASSES, H, W]
            
            # O smp.Unet retorna logits crus. Precisamos do softmax para virar probabilidade (0 a 1)
            probs = torch.softmax(outputs, dim=1) 
            
            # Pegamos a classe com maior probabilidade e a confiança dela
            conf_map_tensor, pred_mask_tensor = torch.max(probs, dim=1)
            
            conf_map_small = conf_map_tensor[0].cpu().numpy()  # [H, W]
            pred_mask_small = pred_mask_tensor[0].cpu().numpy().astype(np.uint8)  # [H, W]

        # Redimensionar de volta para o tamanho original da imagem
        pred_mask = cv2.resize(pred_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        conf_map = cv2.resize(conf_map_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Aplicar o threshold de confiança
        pred_mask_filtered = pred_mask.copy()
        pred_mask_filtered[conf_map < confidence] = 0

        # Aqui mantive a mesma lógica do seu código original caso você 
        # precise descomentar aquele return de results_confidence depois
        classes_found = np.unique(pred_mask_filtered)
        results_confidence = {}

        for cls in classes_found:
            if cls == 0: continue # skip background
            mean_conf = np.mean(conf_map[pred_mask_filtered == cls])
            results_confidence[int(cls)] = float(mean_conf)

        return pred_mask