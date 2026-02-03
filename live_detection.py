from typing import List
from model_types import TrainedModel
from configs.video_inference_config import VideoInferenceConfig
from configs.detect_model_config import DetectModelConfig
from enum import Enum
import cv2
from model_tasks import Task
from video_processor.inference_model import InferenceModel

class LiveDetection:

    
    
    def _multiple_inference_in_frame(self, models_for_inference: List[InferenceModel], frame, capture_objects: bool, show_video: bool):

        for m in models_for_inference:
            result = m.model(frame, stream=True, conf=m.confidence)                    

            if capture_objects:
                objects = []
                for r in result:
                    boxes = r.boxes
                    # for box in boxes:
                    #     objects.append(box.xyxy[0]) 
            if show_video:

                if m.task == Task.SEGMENTATION:
                    frame = self.create_masks_in_frames(result, frame, m.label)
                else:
                    self.create_bounding_box(result, frame, m.label)

        return frame
    
    def _pack_models_for_inference(self, models: List[TrainedModel], labels: List[str], models_for_seg_idxs: List[bool]) -> List[InferenceModel]:
        
        models_for_inference: List[InferenceModel] = []
        
        for model, label, confidences, task_is_seg in zip(models, labels, confidences, models_for_seg_idxs):
            _model = InferenceModel()
            _model.model = model
            _model.confidence = confidences
            _model.label = label
            _model.task = Task.SEGMENTATION if task_is_seg else Task.DETECTION

            models_for_inference.append(_model)

        return models_for_inference


    def _live_detection_loop(self, cam, confidences, labels, models_for_seg_idxs, models: List[TrainedModel], config: VideoInferenceConfig):

        result = None
        
        frame_count = 0

        process_frame = False
        frame_stream = []
        
        while True:
            
            frame_count, process_frame = self._handle_frame_count(frame_count, config.skip_frames)

            check, frame = cam.read()

            if process_frame:

                # invoke loop start callback
                if config.loop_start_callback:
                    config.loop_start_callback()

                # inference
                models_for_inference: List[InferenceModel] = self._pack_models_for_inference(models, labels, models_for_seg_idxs)
                frame = self._multiple_inference_in_frame(models_for_inference, frame, config.capture_objects, config.show_video)

                # show video
                if config.show_video:
                    frame_stream.append(frame)
                    cv2.imshow('video', frame)

                # invoke loop end callback
                if config.loop_end_callback:
                    config.loop_end_callback()


            # process exit
            key = cv2.waitKey(1)
            if key == 27: # esc 
                
                if(config.record):
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec de vídeo
                    output_file = config.record_file_name + ".mkv"  # Nome do arquivo de saída
                    fps = 10.0  # Quadros por segundo
                    frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)  # Cria o objeto de gravação de vídeo
                    for f in frame_stream:
                        out.write(f)

                    out.release()
                break

    def live_detection(self, detect_model_config: List[DetectModelConfig], config: VideoInferenceConfig):
        


        weight_paths = []
        labels = []
        confidence = []
        segmentation = []

        print("detection for classe: ")
        for model_cfg in detect_model_config:

            weight_paths.append(model_cfg.weights_path)
            labels.append(model_cfg.label)
            confidence.append(model_cfg.confidence)
            segmentation.append(model_cfg.segmentation)

            print(model_cfg.label)

        models_trained = self._set_trained_models(weight_paths)
        cam = self._select_cam_source_(config.source, config.ip, config.file)

        self._live_detection_loop(cam, confidence, labels, segmentation, models_trained, config)
        


        cam.release()

        cv2.destroyAllWindows()
