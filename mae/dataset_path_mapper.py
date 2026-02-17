import os


class DatasetPathMapper:

    @staticmethod
    def dataset_to_images_and_labels(dataset_path: str, subpath: str) -> tuple[str, str]:
        path_joined = os.path.join(dataset_path, subpath)
        
        sep = os.sep
        
        search_pattern = f"{sep}images{sep}"   # Ex: /images/
        replace_pattern = f"{sep}labels{sep}"  # Ex: /labels/

        images_path = path_joined
        
        label_path = path_joined.replace(search_pattern, replace_pattern)
        
        # FALLBACK: 
        if label_path == images_path and images_path.endswith("images"):
             label_path = images_path.replace("images", "labels")

        return images_path, label_path