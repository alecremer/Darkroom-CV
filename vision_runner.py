from vision import Vision, DetectModelConfig, TrainModelConfig, DetectConfig
from vision_runner_parser import VisionRunnerParser
from demo_tools import download_demo_data
from pathlib import Path

class VisionRunner:

    def live(self, detect_cfg: DetectConfig, file_config_or_path: str = "config.yaml"):

        model = Vision()
        train_model_cfg, detect_model_cfg, annotate_cfg = VisionRunnerParser().parse_from_file(file_config_or_path)
        model.live_detection(detect_model_cfg, detect_cfg)    

    def train(self, file_config_or_path: str = "config.yaml"):

        model = Vision()
        train_model_cfg, detect_model_cfg, annotate_cfg = VisionRunnerParser().parse_from_file(file_config_or_path)
        model.train(train_model_cfg)

    def annotate(self, img_path: str, file_config_or_path: str = "config.yaml", demo: bool = False):
        model = Vision()

        if demo: 
            download_demo_data()
            base_path = Path("demo")
            img_path = str(base_path /  "PennFudanPed" / "PNGImages")
            file_config_or_path = str(base_path / "config.yaml")

        train_model_cfg, detect_model_cfg, annotate_cfg = VisionRunnerParser().parse_from_file(file_config_or_path)
        
        model.annotate(img_path, annotate_cfg)

    def test(self, file_config_or_path: str = "config.yaml"):
        model = Vision()
        train_model_cfg, detect_model_cfg, annotate_cfg = VisionRunnerParser().parse_from_file(file_config_or_path)

        print(detect_model_cfg)
        for detect_cfg in detect_model_cfg:
            model.test(detect_cfg.weights_path, detect_cfg.test_path, show_image=True)