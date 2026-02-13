from engine import Engine
from configs.detect_model_config import DetectModelConfig
from config_resolver import ConfigResolver
from demo_tools import download_demo_data
from pathlib import Path
from configs.project_config import ProjectConfig

class Runner:


    def live(self, detect_cfg: DetectModelConfig, project_config: ProjectConfig):

        engine = Engine()
        engine.live(project_config.detect, detect_cfg)    

    def train(self, project_config: ProjectConfig):

        engine = Engine()
        engine.train(project_config.train)

    def annotate(self, img_path: str, file_config_or_path: str = "config.yaml", demo: bool = False):
        engine = Engine()

        if demo: 
            download_demo_data()
            base_path = Path("demo")
            img_path = str(base_path /  "PennFudanPed" / "PNGImages")
            file_config_or_path = str(base_path / "config.yaml")

        train_model_cfg, detect_model_cfg, annotate_cfg = ConfigResolver().parse_from_file(file_config_or_path)
        
        print("detect_model_cfg")
        print(annotate_cfg)
        # engine.annotate(img_path, annotate_cfg)
        engine.annotate(img_path, annotate_cfg)

    def test(self, file_config_or_path: str = "config.yaml"):
        engine = Engine()
        train_model_cfg, detect_model_cfg, annotate_cfg = ConfigResolver().parse_from_file(file_config_or_path)

        print(detect_model_cfg)
        for detect_cfg in detect_model_cfg:
            engine.test(detect_cfg.weights_path, detect_cfg.test_path, show_image=True)