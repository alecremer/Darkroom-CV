from annotation_transition.renderer.draw_state import DrawState


class DrawStateMapper:

    @staticmethod
    def map(state: DrawState):

        text = str.lower(state.name)
        text.replace("_", " ")
        color = (55, 125, 222)

        if state is DrawState.IDLE:
            text = "idle"
            
        elif state is DrawState.DRAWING_RECTANGLE:
            color = (252, 102, 3)
            text = "Box"

        elif state is DrawState.DRAWING_MASK:
            color = (55, 125, 222)
            text = "Mask"
            
        return text, color