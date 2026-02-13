from annotation_transition.renderer.annotation_action import AnnotationAction


class ActionMapper:

    @staticmethod
    def can_map(action: AnnotationAction) -> bool:
        return action in {AnnotationAction.NEXT_IMG,
                          AnnotationAction.PREVIOUS_IMG,
                          AnnotationAction.ANNOTATE_BBOX,
                          AnnotationAction.ANNOTATE_MASK,
                          AnnotationAction.RESET_ANNOTATION_CELL,
                          AnnotationAction.QUIT,
                          AnnotationAction.EXCLUDE_CLICKED_ENTITY,
                          AnnotationAction.CREATE_MASK,
                          AnnotationAction.UPDATE}

    @staticmethod
    def map(action: AnnotationAction):

        if action is AnnotationAction.NEXT_IMG: return "next_img"
        elif action is AnnotationAction.PREVIOUS_IMG: return "previous_img"
        elif action is AnnotationAction.UPDATE: return "update"
        elif action is AnnotationAction.ANNOTATE_BBOX: return "annotate_box"
        elif action is AnnotationAction.ANNOTATE_MASK: return "annotate_mask"
        elif action is AnnotationAction.QUIT: return "quit"
        elif action is AnnotationAction.EXCLUDE_CLICKED_ENTITY: return "exclude_entity"
        elif action is AnnotationAction.CREATE_MASK: return "done_mask"
        elif action is AnnotationAction.RESET_ANNOTATION_CELL: return "reset_annotation_cell"