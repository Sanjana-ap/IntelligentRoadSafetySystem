import os
import xml.etree.ElementTree as ET

# ⚡ Dataset root
dataset_root = r"C:\Users\devap\OneDrive\Desktop\MajorProject\intelligent_road_safety\datasets\IDD_FGVD"

# Final YOLO classes
classes = ["car", "bus", "truck", "motorcycle", "bicycle", "autorickshaw", "scooter", "minibus"]

def map_class(label):
    """Map fine-grained dataset labels to coarse YOLO classes"""
    if label.startswith("car_"):
        return "car"
    elif label.startswith("truck_"):
        return "truck"
    elif label.startswith("motorcycle_"):
        return "motorcycle"
    elif label.startswith("scooter_"):
        return "scooter"
    elif label.startswith("autorickshaw_"):
        return "autorickshaw"
    elif label.startswith("mini-bus_") or label == "bus":
        return "bus"
    elif label == "bicycle":
        return "bicycle"
    else:
        return None  # ignore irrelevant classes

def convert_annotation(xml_file, txt_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    if size is None:  # skip broken files
        return
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(txt_file, "w") as out_file:
        for obj in root.iter("object"):
            raw_cls = obj.find("name").text.strip()
            cls = map_class(raw_cls)

            if cls is None:
                continue  # skip unwanted classes

            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")
            xmin = float(xmlbox.find("xmin").text)
            ymin = float(xmlbox.find("ymin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymax = float(xmlbox.find("ymax").text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bbox_width = (xmax - xmin) / w
            bbox_height = (ymax - ymin) / h

            out_file.write(f"{cls_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

# Walk through train/val/test
for split in ["train", "val", "test"]:
    ann_dir = os.path.join(dataset_root, split, "annos")
    label_dir = os.path.join(dataset_root, "labels", split)
    os.makedirs(label_dir, exist_ok=True)

    for xml_file in os.listdir(ann_dir):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(ann_dir, xml_file)
        txt_path = os.path.join(label_dir, xml_file.replace(".xml", ".txt"))
        convert_annotation(xml_path, txt_path)

print("✅ Conversion completed! YOLO labels saved in labels/ folders.")
