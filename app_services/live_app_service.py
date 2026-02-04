from app_services.app_service import AppService
from live_config_factory import LiveConfigFactory
from project_config_loader import ProjectConfigLoader

class LiveAppService(AppService):
    
    

    def execute(self, config_path: str = "config.yaml"):
        factory = LiveConfigFactory()
        project_cfg = ProjectConfigLoader.load(config_path)
        (runner, detect_cfg) = factory._build_runner(self.args)
        runner.live(detect_cfg, project_cfg)

    # def test_loop(self):
    #     show_video = False
    #     source = "file"
    #     skip_frames = 0
    #     capture_objects = False
    #     performance_log = True
    #     file_name = "test.mkv"
    #     # t = 70
    #     t = 10

    #     test_cfg = DetectConfig(show_video, capture_objects, performance_log, source, file_name, skip_frames, record, record_name)
    #     test_cfg.loop_start_callback = lambda: metrics.log_performance()
    #     detect_cfg_list = ConfigResolver().parse_from_file(file_config_or_path)[1]
    #     detect_cfg_test = []

    #     for idx in range(len(detect_cfg_list)):
    #         metrics = Metrics()

    #         cfg = detect_cfg_list[:idx+1]
    #         metrics.capture_objects = False
    #         metrics = self.set_metrics_active_ais(cfg, metrics)

    #         # active_ias
    #         # for d in cfg:
    #             # active_ias = metrics.active_ias + " - " + d.label

            
    #         print("------------------------------------")
    #         print("")
    #         print("test for classes:")
    #         print(f"cap: {capture_objects}")
    #         print(metrics.active_ias)
    #         self.run_with_timeout(lambda test_cfg=test_cfg: runner.live(test_cfg),timeout=t)
    #         print("finished")
            
    #         metrics.reinitialize()
    #         capture_objects = True
    #         metrics.capture_objects = True
    #         test_cfg = DetectConfig(show_video, capture_objects, performance_log, source, file_name, skip_frames, record, record_name)
    #         test_cfg.loop_start_callback = lambda: metrics.log_performance()
    #         # metrics.capture_objects = capture_objects
    #         print(f"for cap: {capture_objects}")
    #         print(metrics.active_ias)
    #         self.run_with_timeout(lambda test_cfg=test_cfg: runner.live(test_cfg),timeout=t)
    #         # self.run_with_timeout(lambda cfg=cfg: model.live_detection(cfg, capture_objects=capture_objects, file=str(file_name), show_video=show_video, loop_end_callback=lambda: metrics.log_performance(), source=source, ip="10.42.0.47:8080/h264_pcm.sdp", skip_frames=skip_frames), 
    #                         # timeout=t)
    #         print("finished")

