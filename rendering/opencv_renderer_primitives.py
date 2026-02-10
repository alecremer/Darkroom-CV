import cv2
import numpy as np
from types.entities import Rectangle

class OpencvRenderPrimitives:
    
    @staticmethod
    def draw_rectangle(img, rect: Rectangle, color = (0, 255, 0), thickness = 2):
        
        x0, y0, x, y = rect.to_coords()
        cv2.rectangle(img, (x0, y0), (x, y), color, thickness)
        OpencvRenderPrimitives.resize_and_show(img)

        
    @staticmethod
    def normalize_by_scale(x, y, scale) -> tuple[int, int]:
        x = int(x/scale)
        y = int(y/scale)

        return x, y
        
    @staticmethod
    def render_poly(poly, img, color = (128, 255, 0)):

        if len(poly) > 2:
            point_list = [(p.x, p.y) for p in poly]
            point_list = np.array(point_list, np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [point_list], (color[0], color[1], color[2], 0.5))
            cv2.polylines(overlay, [point_list], True, (0, 0, 0), 2)
            img_overlay = cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0)
            img = img_overlay
        for p in poly:
            cv2.circle(img, (p.x, p.y), 10, color, -1)
        return img
    
    
    
    @staticmethod
    def draw_btn(img, rect: Rectangle, color, text_color, text_origin_x, text_origin_y, text, font_size, text_thickness):
        x0, y0, x, y = rect.to_coords()
        cv2.rectangle(img, (x0, y0), (x, y), color, -1)
        cv2.putText(img, text, (text_origin_x, text_origin_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, text_thickness)

    
    @staticmethod
    def resize_and_show(img, res = (1080, 720)):
        #TODO find a best way to handle res
        scale_width = res[0] / img.shape[1]
        scale_height = res[1] / img.shape[0]
        scale = min(scale_width, scale_height, 1.0) 
        OpencvRenderPrimitives.resize_scale = scale

        if scale < 1.0:
            display_img = cv2.resize(img, None,
                                    fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA)
        else:
            display_img = img
        cv2.imshow('Annotation', display_img)