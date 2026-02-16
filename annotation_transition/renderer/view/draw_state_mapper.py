from annotation_transition.renderer.draw_state import DrawState


class DrawStateMapper:

    @staticmethod
    def map(state: DrawState):

        text = str.lower(state.name)
        text.replace("_", " ")
        color = (55, 125, 222)

        if state is DrawState.IDLE:
            text = ""
            
        elif state is DrawState.DRAWING_RECTANGLE:
            color = (252, 102, 3)
            text = "Box"

        elif state is DrawState.DRAWING_MASK:
            color = (55, 125, 222)
            text = "Mask"

        elif state is DrawState.DRAWING_MASK_LASSO:
            color = (123, 179, 39)
            text = "Lasso"
            
        return text, color