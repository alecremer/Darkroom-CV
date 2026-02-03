from abc import ABC, abstractmethod

class InferencePipeline(ABC):

    def inference(self, frame):
        pass