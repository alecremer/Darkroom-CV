import re

class Tools:

    @staticmethod
    def natural_sort(l):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', l)]
