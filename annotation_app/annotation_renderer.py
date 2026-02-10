from abc import ABC, abstractmethod

class AnnotationRender(ABC):

    @abstractmethod
    def draw_guidelines(self, x, y, h, w):
        pass