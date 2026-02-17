from os import path, sep


class DatasetPathMapper:

    @staticmethod
    def dataset_to_images_and_labels(dataset_path: str, subpath: str) -> tuple[str, str]:
        path_joined = path.join(dataset_path, subpath)
        search_pattern = f"{sep}images{sep}"
        replace_pattern = f"{sep}labels{sep}"

        images_path = path_joined
        label_path = path_joined.replace(search_pattern, replace_pattern)

        return images_path, label_path