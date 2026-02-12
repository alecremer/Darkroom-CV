from annotation_transition.renderer.annotation_action import AnnotationAction


class ActionMapper:

    @staticmethod
    def can_map(action: AnnotationAction) -> bool:
        return action in {AnnotationAction.NEXT_IMG,
                          AnnotationAction.PREVIOUS_IMG,
                          AnnotationAction.ANNOTATE_BBOX,
                          AnnotationAction.UPDATE}

    @staticmethod
    def map(action: AnnotationAction):

        if action is AnnotationAction.NEXT_IMG: return "next_img"
        elif action is AnnotationAction.PREVIOUS_IMG: return "previous_img"
        elif action is AnnotationAction.UPDATE: return "update"
        elif action is AnnotationAction.ANNOTATE_BBOX: return "annotate_box"