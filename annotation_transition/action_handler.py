from typing import Any
from annotation_transition.annotation_data import AnnotationData
from annotation_transition.annotation_engine import AnnotationEngine
from annotation_transition.annotation_repository import AnnotationRepository
from annotation_transition.dataset_navigator import DatasetNavigator
from annotation_transition.engine_action import AnnotationEngineAction

class ActionHandler:

    def __init__(self, engine: AnnotationEngine, navigator: DatasetNavigator, repo: AnnotationRepository):
        self.engine = engine
        self.navigator = navigator
        self.repo = repo

    def handle_navigation(self, action: AnnotationEngineAction, data: AnnotationData, payload: Any = None):
        if action is AnnotationEngineAction.NEXT_IMG:
            data.file_index = self.navigator.next_img()

        elif action is AnnotationEngineAction.PREVIOUS_IMG:
            data.file_index = self.navigator.previous_img()

    def handle(self, action: AnnotationEngineAction, data: AnnotationData, payload: Any = None):
        
        
        if action is AnnotationEngineAction.UPDATE:
            pass

        elif action is AnnotationEngineAction.ANNOTATE_BBOX:
            self.engine.annotate_bbox(payload, data.label, data.annotations, data.file_index)

        elif action is AnnotationEngineAction.ANNOTATE_MASK:
            self.engine.annotate_mask(payload, data.annotations, data.file_index)

        elif action is AnnotationEngineAction.RESET_ANNOTATION_CELL:
            self.engine.reset_annotation_cell(data.annotations, data.file_index)

        elif action is AnnotationEngineAction.SELECT_LABEL:
            data.label = self.engine.select_label(payload, data.labels, data.label)

        elif action is AnnotationEngineAction.SAVE_ANNOTATIONS:
            self.repo.save_annotations(data.annotations, data.labels, data.file_index)

        elif action is AnnotationEngineAction.EXCLUDE_CLICKED_ENTITY:
            self.engine.exclude_box_from_annotation(payload, data.annotations, data.file_index)
            self.engine.exclude_polygon_from_annotations(payload, data.annotations, data.file_index)