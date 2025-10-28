from tqdm import tqdm
import ultralytics
from ultralytics import YOLO
from ultralytics import SAM
from ultralytics.data import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from pathlib import Path
import cv2
import numpy as np
import os
import yaml


config = {}

with open("config.yaml") as file_name:
    try:
        config = yaml.safe_load(file_name)
    except:
        raise("cannot open config file")

raw_path = config["raw_path"]
save_path = config["save_path"]
save_path_prefix = config["save_path_prefix"]
epochs = config["epochs"]

if save_path_prefix:
    save_path = os.path.join(save_path, save_path_prefix)



def yolo_bbox2segment(im_dir, save_dir=None, sam_model="sam_b.pt"):
    """
    Converts existing object detection dataset (bounding boxes) to segmentation dataset or oriented bounding box (OBB)
    in YOLO format. Generates segmentation data using SAM auto-annotator as needed.

    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path): Path to save the generated labels, labels will be saved
            into `labels-segment` in the same directory level of `im_dir` if save_dir is None. Default: None.
        sam_model (str): Segmentation model to use for intermediate segmentation data; optional.

    """

    # NOTE: add placeholder to pass class index check
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000))))

    if len(dataset.labels[0]["segments"]) > 0:  # if it's segment data
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return
    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model).to("cuda:0")
    
    # create save image directory
    save_masked_img_dir = Path(save_dir + "/images_masked") if save_dir else Path(im_dir).parent / "labels-segment-images"
    save_masked_img_dir.mkdir(parents=True, exist_ok=True)

    save_img_dir = Path(save_dir + "/images") if save_dir else Path(im_dir).parent / "labels-segment-images"
    save_img_dir.mkdir(parents=True, exist_ok=True)
    
    #process YOLO labels and generate segmentation masks using the SAM model
    for l in tqdm(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):
        h, w = l["shape"]
        boxes = l["bboxes"]
        if len(boxes) == 0:  # skip empty labels
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(l["im_file"])
        im = cv2.resize(im, (512, 512)) 
        # cv2.imshow(im)
        # sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, show=True)
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False)
        # waitkey = cv2.waitKey(0)
        for result in sam_results:
            result.save(save_masked_img_dir / Path(l["im_file"]).name)
            cv2.imwrite(save_img_dir / Path(l["im_file"]).name, result.orig_img)

        l["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir + "/labels") if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Saves segmentation masks and class labels to text files for each image
    for l in dataset.labels:
        texts = []
        lb_name = Path(l["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = l["cls"]
        for i, s in enumerate(l["segments"]):
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        if texts:
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    LOGGER.info(f"Generated segment labels saved in {save_dir}")

def train():
    model = YOLO("models/" + "yolo11n-seg.pt")
    path = "/path_to_config/config.yaml"
    results = model.train(data=path, epochs=epochs, imgsz=640)
    model.val()

# train()
yolo_bbox2segment(raw_path, save_path)