import cv2
from entities.entities import BoundingBox
import math

class AnnotationMapper:

    def result_to_bounding_box(self, detection_result, labels_to_annotate = None):
        detected = []
        for r in detection_result:
            for annotate_index, annotate_label in enumerate(labels_to_annotate):
                boxes = r.boxes
                class_indices = boxes.cls

                for i, class_index in enumerate(class_indices):
                    class_id = int(class_index)


                    if annotate_index == class_id:
                        label = annotate_label
                        box = boxes.xyxy[i]
                        
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # confidence
                        confidence = math.ceil((boxes.conf[i]*100))/100
                        bounding_box = BoundingBox(label, box, confidence)
                        detected.append(bounding_box)
        return detected

    
    def get_masks_from_result(self, result, frame_ref):
        masks = []
        for r in result:
            if r.masks is not None and len(r.masks.data) > 0:
                
                mask = r.masks.data[0].cpu().numpy()
                if mask is not None and mask.size > 0:
                    mask_resized = cv2.resize(mask, (frame_ref.shape[1], frame_ref.shape[0]))
                    masks.append(mask_resized)

        return masks