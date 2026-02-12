from enum import Enum

class AnnotationAction(Enum):
    IDLE = 0
    START_CONSTRUCT_RECTANGLE = 5
    DRAW_CONSTRUCT_RECTANGLE = 10
    ANNOTATE_BBOX = 20
    CANCEL_CONSTRUCT_MASK = 30
    SELECT_LABEL = 40
    START_CONSTRUCT_MASK = 50
    EXCLUDE_CLICKED_ENTITY = 60
    NEXT_IMG = "next_img"
    PREVIOUS_IMG = "previous_img"
    RESET_ANNOTATION_CELL = 85
    TOGGLE_SHOW_UI = "toggle_show_ui"
    SAVE_ANNOTATIONS = "save_annotations"
    UNDO_MASK_POINT = "undo_mask_point"
    DONE_MASK = "done_mask"
    CREATE_MASK = "create_mask"
    QUIT = "quit"
    CREATE_BOX = "create_box"
    UPDATE = "update"