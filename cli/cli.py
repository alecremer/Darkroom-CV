import multiprocessing
from typing import List
from engine import Engine, DetectModelConfig, TrainModelConfig, DetectConfig, AnnotateModelConfig
from runner import Runner
from cli.cli_parser import cli_parser
from metrics import Metrics
from config_resolver import ConfigResolver
from app_services.live_app_service import LiveAppService

class CLI:

    def __init__(self):
        
        self.args = cli_parser.parse()
    
    def set_metrics_active_ais(self, detect_config_list: List[DetectModelConfig], metrics: Metrics) -> Metrics:

        for d in detect_config_list:
            metrics.active_ias = metrics.active_ias + " - " + d.label
        return metrics
    
    def run_with_timeout(self, func, timeout):

        process = multiprocessing.Process(target=func)
        process.start()
        process.join(timeout)  

        if process.is_alive():
            print("Timeout reached. Terminating function...")
            process.terminate()
            process.join()


    
    def run(self, file_config_or_path: str = "config.yaml"):


        runner = Runner()
        run_mode = self.args.run_mode

        if len(run_mode) > 0:

            if self.args.run_mode == "test":
                runner.test(file_config_or_path)

            elif self.args.run_mode == "train":
                file_config_or_path = self.args.config_file_path
                runner.train( file_config_or_path)  

            
            elif self.args.run_mode == "annotate":
                img_path = self.args.path
                demo = self.args.demo

                runner.annotate(img_path, file_config_or_path, demo)
            
            elif self.args.run_mode == "live":

                live_command: LiveAppService = LiveAppService(self.args)
                live_command.execute(file_config_or_path)
                
                # else:
                #     runner.live(detect_cfg, file_config_or_path)    

