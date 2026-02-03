from abc import ABC, abstractmethod

class BaseCommand(ABC):

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def execute(self, config_path: str = "config.yaml"):
        pass
        