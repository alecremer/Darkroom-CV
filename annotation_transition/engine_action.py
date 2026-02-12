from enum import Enum

class AnnotationEngineAction(Enum):
    ANNOTATE_BBOX = "annotate_box"
    SELECT_LABEL = "select_label"
    EXCLUDE_CLICKED_ENTITY = "exclude_entity"
    NEXT_IMG = "next_img"
    PREVIOUS_IMG = "previous_img"
    RESET_ANNOTATION_CELL = "reset_annotation_cell"
    SAVE_ANNOTATIONS = "save_annotation"
    UNDO_MASK_POINT = "undo_mask_point"
    DONE_MASK = "done_mask"
    UPDATE = "update"
    # CREATE_BOX = "150",