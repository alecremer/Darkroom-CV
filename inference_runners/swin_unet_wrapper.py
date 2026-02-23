import cv2
import numpy as np
from entities.entities import Point, PolygonalMask
from inference_runners.model_wrapper import ModelWrapper
# Ajuste o import abaixo para apontar para a pasta/arquivo correto do seu projeto
from swin_unet.swin_unet_inference import SwinUnetInference


class SwinUnetWrapper(ModelWrapper):

    def _load_swin_unet(self, model_path: str) -> SwinUnetInference:
        model = SwinUnetInference()
        # Inicializando com 2 classes (Fundo e Objeto) igual ao SETR-PUP
        model.load_swin_unet(model_path, num_classes=2) # TODO: multiclasses support
        return model
    
    def __init__(self, model_path: str):
        self.model = self._load_swin_unet(model_path)


    def inference(self, img, confidence: float, payload = None):
        
        predict_mask = self.model.predict(img, confidence)
        
        # Garante que a máscara fique no formato adequado para o OpenCV encontrar os contornos
        predict_mask_8bit = (predict_mask * 255).astype(np.uint8) if predict_mask.max() <= 1 else predict_mask.astype(np.uint8)

        contours_list, _ = cv2.findContours(predict_mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        masks = []

        for contour in contours_list:
            if len(contour) < 3:
                continue
            
            mask_points = []
            
            for point_wrapper in contour:
                x, y = point_wrapper[0]
                
                mask_points.append(Point((x), (y)))
            
            if len(mask_points) >= 3:
                masks.append(PolygonalMask("sample", mask_points, confidence)) #TODO: Get classes names/ids
        
        return masks