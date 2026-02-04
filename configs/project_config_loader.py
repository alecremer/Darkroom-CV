from project_config import ProjectConfig
from config_resolver import ConfigResolver


class ProjectConfigLoader:

    @staticmethod
    def load(path: str) -> ProjectConfig:
        train_model_cfg, detect_model_cfg, annotate_cfg = ConfigResolver().parse_from_file(path)
        return ProjectConfig(train_model_cfg, detect_model_cfg, annotate_cfg)
