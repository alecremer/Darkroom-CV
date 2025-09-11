import os

def some(labels_path: str):
    label_list = [
        f for f in os.listdir(labels_path)
        if os.path.splitext(f)[1].lower() in {".txt"}
    ]

    os.makedirs(labels_path + "/labels_yolo", exist_ok=True)

    label_yolo = []

    for original_label in label_list:
        filename = original_label
        
        with open(labels_path + "/" + filename) as f:
            if filename != "classes.txt": 
                data = f.readlines()
                yolo_boxes = []
                for d in data:
                    (c, x1, y1, x2, y2) = d.strip().split(" ")
                    
                    x1 = float(x1)
                    x2 = float(x2)
                    y1 = float(y1)
                    y2 = float(y2)

                    x1 = 1.0 if x1 > 1.0 else x1
                    x1 = 0.0 if x1 < 0.0 else x1

                    x2 = 1.0 if x2 > 1.0 else x2
                    x2 = 0.0 if x2 < 0.0 else x2

                    y1 = 1.0 if y1 > 1.0 else y1
                    y1 = 0.0 if y1 < 0.0 else y1

                    y2 = 1.0 if y2 > 1.0 else y2
                    y2 = 0.0 if y2 < 0.0 else y2


                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width/2
                    y_center = y1 + height/2

                    data_yolo = f"{c} {x_center} {y_center} {width} {height}"
                    yolo_boxes.append(data_yolo)
                
                label_yolo.append({"name": filename, "data": yolo_boxes})
    
    for label in label_yolo:
        with open(labels_path + f"/labels_yolo/" + label["name"], "w") as f:
            for data in label["data"]:
                f.write(data)
                f.write("\n")

            

if __name__ == "__main__":
    some("/home/ale/Downloads/PI2/dataset/valid/labels")