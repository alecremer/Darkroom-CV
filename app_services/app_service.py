from abc import ABC, abstractmethod

class AppService(ABC):

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def execute(self, config_path: str = "config.yaml"):
        pass
        