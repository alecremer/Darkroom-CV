from app_services.app_service import AppService
from runner import Runner
from app_services.app_service import AppService
from project_config_loader import ProjectConfigLoader


class TrainAppService(AppService):
    def execute(self, config_path = "config.yaml"):
        project_cfg = ProjectConfigLoader.load(config_path)
        runner = Runner()
        runner.train(project_cfg)