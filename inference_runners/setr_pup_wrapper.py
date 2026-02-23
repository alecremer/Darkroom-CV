
import cv2
import numpy as np
import torch
from entities.entities import Point, PolygonalMask
from inference_runners.model_wrapper import ModelWrapper
from setr_pup.setr_pup_inference import SetrPupInference


class SetrPupWrapper(ModelWrapper):

    def _load_vitmae_seg(self, encoder_path, head_path) -> SetrPupInference:

        model = SetrPupInference()
        model.load_vitmae_seg(encoder_path, head_path, 2) # TODO: multiclasses support

        return model
    
    def __init__(self, encoder_path, head_path):
        self.model = self._load_vitmae_seg(encoder_path, head_path)


    def inference(self, img, confidence, payload = None):
        
        predict_mask = self.model.predict(img, confidence)
        
        predict_mask_8bit = (predict_mask * 255).astype(np.uint8) if predict_mask.max() <= 1 else predict_mask.astype(np.uint8)

        contours_list, _ = cv2.findContours(predict_mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        masks = []

        # _, prediction_map = torch.max(predict_mask, dim=1)
        # prediction_map = prediction_map.squeeze().cpu().numpy().astype(np.uint8)
        # unique_classes = np.unique(prediction_map)

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