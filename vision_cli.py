import multiprocessing
from typing import List
from vision import Vision, DetectModelConfig, TrainModelConfig, DetectConfig, AnnotateModelConfig
from vision_runner import VisionRunner
from cli_parser import cli_parser
from metrics import Metrics
from vision_runner_parser import VisionRunnerParser


class VisionCLI:

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


        vision_runner = VisionRunner()
        run_mode = self.args.run_mode

        if len(run_mode) > 0:

            if self.args.run_mode == "test":
                # model.test(bottle_weight_path, bottle_dataset_path / "test/images")
                vision_runner.test(file_config_or_path)

            elif self.args.run_mode == "train":
                file_config_or_path = self.args.config_file_path
                vision_runner.train( file_config_or_path)  

            
            elif self.args.run_mode == "annotate":
                img_path = self.args.path
                demo = self.args.demo

                vision_runner.annotate(img_path, file_config_or_path, demo)
            
            elif self.args.run_mode == "live":

                

                show_video = not self.args.no_video
                performance_log = self.args.performance_log
                skip_frames = self.args.skip_frames
                capture_objects = self.args.capture_objects
                file_name = self.args.file
                record_name = self.args.record
                record = record_name != ""
                test = self.args.test


                source = "self"
                if(self.args.rtsp):
                    source = "rtsp"
                elif(self.args.file):
                    source = "file"
                
                detect_cfg = DetectConfig(show_video, capture_objects, performance_log, source, file_name, skip_frames, record, record_name)
                


                if performance_log:

                    metrics = Metrics()
                    metrics = self.set_metrics_active_ais(detect_cfg, metrics)
                    vision_runner.metrics = metrics
                    detect_cfg.loop_start_callback = lambda: metrics.log_performance()
                    vision_runner.live(detect_cfg, file_config_or_path)    


                elif test:

                    
                    show_video = False
                    source = "file"
                    skip_frames = 0
                    capture_objects = False
                    performance_log = True
                    file_name = "test.mkv"
                    # t = 70
                    t = 10

                    test_cfg = DetectConfig(show_video, capture_objects, performance_log, source, file_name, skip_frames, record, record_name)
                    test_cfg.loop_start_callback = lambda: metrics.log_performance()
                    detect_cfg_list = VisionRunnerParser().parse_from_file(file_config_or_path)[1]
                    detect_cfg_test = []

                    for idx in range(len(detect_cfg_list)):
                        metrics = Metrics()

                        cfg = detect_cfg_list[:idx+1]
                        metrics.capture_objects = False
                        metrics = self.set_metrics_active_ais(cfg, metrics)

                        # active_ias
                        # for d in cfg:
                            # active_ias = metrics.active_ias + " - " + d.label

                        
                        print("------------------------------------")
                        print("")
                        print("test for classes:")
                        print(f"cap: {capture_objects}")
                        print(metrics.active_ias)
                        self.run_with_timeout(lambda test_cfg=test_cfg: vision_runner.live(test_cfg),timeout=t)
                        print("finished")
                        
                        metrics.reinitialize()
                        capture_objects = True
                        metrics.capture_objects = True
                        test_cfg = DetectConfig(show_video, capture_objects, performance_log, source, file_name, skip_frames, record, record_name)
                        test_cfg.loop_start_callback = lambda: metrics.log_performance()
                        # metrics.capture_objects = capture_objects
                        print(f"for cap: {capture_objects}")
                        print(metrics.active_ias)
                        self.run_with_timeout(lambda test_cfg=test_cfg: vision_runner.live(test_cfg),timeout=t)
                        # self.run_with_timeout(lambda cfg=cfg: model.live_detection(cfg, capture_objects=capture_objects, file=str(file_name), show_video=show_video, loop_end_callback=lambda: metrics.log_performance(), source=source, ip="10.42.0.47:8080/h264_pcm.sdp", skip_frames=skip_frames), 
                                        # timeout=t)
                        print("finished")
            
                else:
                    vision_runner.live(detect_cfg, file_config_or_path)    

