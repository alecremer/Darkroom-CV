import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from os import path, listdir, sep
import numpy as np
from PIL import Image
import cv2

from setr_pup.dataset_path_mapper import DatasetPathMapper

class YOLODetectionDataset(Dataset):

    def __init__(self, dataset_path: str, transform = None):
        self.images_dir = path.join(dataset_path, "train", "images")
        self.labels_dir = path.join(dataset_path, "train", "labels")
        self.images_files = sorted(listdir(self.images_dir))
        self.transform = transform

        print(f"dataset path: {dataset_path}")
        
        if path.exists(self.images_dir):
            print(f"images path founded: {self.images_dir}")

        if path.exists(self.labels_dir):
            print(f"labels path founded: {self.labels_dir}")

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        img_path = path.join(self.images_dir, self.images_files[idx])
        label_name = self.images_files[idx].split('.')[0] + '.txt'
        label_path = path.join(self.labels_dir, label_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # por que???

        boxes = []
        labels = []
        
        if path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    labels.append(int(cls))
                    boxes.append([x, y, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img = self.transform(img)

        return img, target



class YOLOSegmentationDataset(Dataset):

    def __init__(self, dataset_path: str, image_size: int, transform = None, split: str = "train"):
        # path_joined = path.join(dataset_path, split)
        # search_pattern = f"{sep}images{sep}"
        # replace_pattern = f"{sep}labels{sep}"

        # label_path = path_joined.replace(search_pattern, replace_pattern)

        image_path, label_path = DatasetPathMapper.dataset_to_images_and_labels(dataset_path, split)
        self.images_dir = image_path
        self.labels_dir = label_path
        self.images_files = sorted(listdir(self.images_dir))

        self.transform = transform
        self.image_size = image_size

        
        if path.exists(self.images_dir):
            print(f"images path founded: {self.images_dir}")

        if path.exists(self.labels_dir):
            print(f"labels path founded: {self.labels_dir}")

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        img_path = path.join(self.images_dir, self.images_files[idx])
        label_name = self.images_files[idx].split('.')[0] + '.txt'
        label_path = path.join(self.labels_dir, label_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        H, W = img.shape[:2]

        mask = np.zeros((H, W), dtype=np.uint8)
        
        if path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    cls = vals[0]
                    poly = np.array(vals[5:], dtype=np.float32).reshape(-1, 2)
                    poly[:, 0] *= W
                    poly[:, 1] *= H
                    cv2.fillPoly(mask, [poly.astype(np.int32)], color=cls + 1)

        if self.transform:
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil)
            mask = torch.tensor(cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST), dtype=torch.long)
        else:
            img_tensor = transforms.ToTensor()(cv2.resize(img, (self.image_size, self.image_size)))
            mask = torch.tensor(cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST), dtype=torch.long)

        return img_tensor, mask



