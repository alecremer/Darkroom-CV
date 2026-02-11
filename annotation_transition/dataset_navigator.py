class DatasetNavigator:

    def __init__(self, folder_list):
        self.file_index: int = 0
        self.folder_list = folder_list

    def next_img(self):
        if self.file_index +1 < len(self.folder_list):
            self.file_index = self.file_index + 1

    def previous_img(self):
        self.file_index = self.file_index - 1

        if self.file_index < 0:
            self.file_index = 0