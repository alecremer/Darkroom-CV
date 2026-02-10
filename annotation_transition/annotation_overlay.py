import cv2

class AnnotationOverlay:

    def draw_guide_lines(self, img, x, y):
        h, w = img.shape[:2]
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)   # vertical
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)   # horizontal