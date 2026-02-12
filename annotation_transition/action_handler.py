from typing import Any
from annotation_transition.annotation_data import AnnotationData
from annotation_transition.annotation_engine import AnnotationEngine
from annotation_transition.dataset_navigator import DatasetNavigator
from annotation_transition.engine_action import AnnotationEngineAction

class ActionHandler:

    def __init__(self, engine: AnnotationEngine, navigator: DatasetNavigator):
        self.engine = engine
        self.navigator = navigator

    def handle_navigation(self, action: AnnotationEngineAction, data: AnnotationData, payload: Any = None):
        if action is AnnotationEngineAction.NEXT_IMG:
            data.file_index = self.navigator.next_img()

        elif action is AnnotationEngineAction.PREVIOUS_IMG:
            data.file_index = self.navigator.previous_img()

    def handle(self, action: AnnotationEngineAction, data: AnnotationData, payload: Any = None):
        
        
        if action is AnnotationEngineAction.UPDATE:
            pass

        # if action is AnnotationEngineAction.ANNOTATE_BBOX:
        #     self.engine.annotate_bbox(payload, data.label, data.annotations, data.file_index)

        # elif action is AnnotationEngineAction.SELECT_LABEL:
        #     self.select_label(payload)

        # elif action is AnnotationEngineAction.START_CONSTRUCT_RECTANGLE:
        #     self.start_construct_rectangle(payload)

        # elif action is AnnotationEngineAction.EXCLUDE_CLICKED_ENTITY:
        #     self.exclude_box_from_annotation(payload)
        #     self.exclude_polygon_from_annotations(payload)