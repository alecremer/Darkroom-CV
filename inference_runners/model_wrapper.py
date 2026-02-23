from abc import ABC, abstractmethod

class ModelWrapper(ABC):

    @abstractmethod
    def inference(self, img, confidence, payload = None):
        pass