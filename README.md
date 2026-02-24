# Darkroom CV
> **An End-to-End, Model-Agnostic Computer Vision Framework**

## Overview
**Darkroom CV** is a comprehensive computer vision framework designed to manage the entire Machine Learning lifecycle—from dataset annotation and model training to multi-source inference and hardware profiling. 

Built as a robust experimentation sandbox, it aims to solve the fragmentation of CV workflows by unifying them into a single, purely local, and scalable environment.

### Architecture & Design Philosophy
This project was built with a focus on maintainability, scalability, and practical utility:

* **Model-Agnostic Core:** The framework's core relies on abstractions and lists, decoupling the logic from specific architectures. This allows seamless integration of new models, enabling side-by-side performance comparisons and A/B testing.

* **Full ML Lifecycle Ownership:** By handling annotation (with Active Learning), training, and deployment (inference), it eliminates the need to jump between disjointed tools, demonstrating a complete understanding of MLOps pipelines.

* **Local-First & Unrestricted:** Developed to bypass the limitations of cloud-based proprietary tools (like image caps or OS restrictions), it provides a powerful, uncapped local environment for researchers and engineers.

---

## Demos

Bounging boxes annotation:

https://github.com/user-attachments/assets/784cf68f-b737-4a60-bbda-e79808b6d954

Masks annotation:

https://github.com/user-attachments/assets/7e74a936-5892-43cb-8388-97bcfac533f0


---

## Core Capabilities

### AI-Assisted Annotation Workflow
A custom-built annotation tool designed to accelerate dataset creation through active learning:
* **Custom AI Assistance:** Seamlessly integrates your custom-trained models (YOLO Seg, YOLO Det, and SETR-PUP) directly into the UI to auto-generate bounding boxes and polygon masks. Train the models in the framework, use them to assist your annotations.
* **Granular Control:** Features a mouse-wheel-sensitive lasso tool for masks, confidence thresholding, multiclass annotation support, and complete undo history (including specific polygon point rollbacks).
* **Flexibility:** Toggle mask/bbox validity, delete-all per frame, and disable UI for streamlined workflows. Includes a built-in Penn-Fudan demo (no AI required) to test the tools out-of-the-box.

### Scalable Training & Multi-Model Inference
* **Stack Execution:** Train multiple AI models sequentially within a single run.
* **Versatile Inference:** Native support for various input streams, including local webcams, video files, and RTSP server feeds.

### Performance Profiling & Hardware Stress Testing
Built-in tools to measure and visualize system impact during inference:
* Automatically logs Cycle Time, RAM, GPU, and CPU usage alongside Active AIs and Captured Objects data.
* Includes a data processing script to generate analytical plots directly from logs.

---

## Usage & CLI

Currently, the framework is executed via the main python script. System parameters and active models are controlled via `config.yaml`.


```python3 main.py [mode] [-args]```

## modes
### 1. Live
run real-time detection<br>
```
python3 main.py live [-args]
```
<br>

__args:__

- ```-nv``` no video

- ```-cap``` capture objects (process image)

- ```-pl``` create [performance logs ](#Performance-Logs)

- ```-rtsp``` live from video of rtsp server

- ```-sf [frame_number]``` skip frames for better performance

- ```-rec [output_file_name]``` record

- ```-f [file_path]``` live from file
<br><br>

### 2. Train:

```
python3 main.py train -p [path_to_config_file]
```

> [!NOTE]
> If segmentation IA, run [segmentation from boxes](#segmentation-from-boxes)

<br>

### 3. Data processing:
Generate plots from logs, separating _capture objects_ from _non capture objects_.<br>
```
python3 data_processing.py
```
Processed logs are saved in the ```logs_processed``` folder
<br>

### 4. Annotation:
run annotation call:<br>
```
python3 main.py annotate -p [path_to_config_file]
```
run demo:<br>
```
python3 main.py annotate -demo
```


<br>

## Performance Logs
Logs include time, ram usage, gpu usage, cpu usage, cycle time, active AIs and captre objects data.

> [!IMPORTANT]
> Logs are always saved on ```logs``` folder


## Configuration file

All AI settings are defined in the ```config.yaml``` file.

### Parameters:
- ```dataset:``` dataset path

- ```weights:``` weights path for detection

- ```confidence:``` confidence for detection

- ```labels:``` class labels

- ```device:``` cpu or gpu (cuda)

- ```result folder name:``` name of folder for train results

- ```model:``` AI model (vit-mae, YOLO model, setr-pup, pup-head, swin-mae, swin-unet)

- ```detect:``` true or false, activate detection for this AI

- ```train:``` true or false, activate train for this AI

- ```segmentation:``` true or false, set true if segmentation AI 

## Segmentation from boxes
Create segmentation dataset from bounding boxes dataset

1. Configure paths in segmentation [config file](#segmentation-configuration-file)

2. In seg_from_boxes folder, run ```python3 seg.py```

### Segmentation configuration file

__parameters:__ 

- ```raw_path:``` path to folder containing raw images folder and labels folder

- ```save_path:``` path to save segmentation dataset

- ```save_path_prefix:``` create parent folder for save path, can be empty

- ```epochs:``` train epochs -->
