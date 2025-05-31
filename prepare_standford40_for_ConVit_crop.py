import os
import shutil
from glob import glob
import xml.etree.ElementTree as ET
from PIL import Image

# Paths (update these to match your setup)
ROOT_DIR = r"C:\Users\andin\OneDrive\Desktop\Project Phase1 VDT2025\Data\Stanford40"  # root of the dataset
IMAGES_DIR = os.path.join(ROOT_DIR, "JPEGImages")
SPLITS_DIR = os.path.join(ROOT_DIR, "ImageSplits")
XML_DIR = os.path.join(ROOT_DIR, "XMLAnnotations")
OUTPUT_DIR = os.path.join(ROOT_DIR, "organized_cropped_data")

# Ensure output directories exist
for phase in ["train", "test"]:
    phase_dir = os.path.join(OUTPUT_DIR, phase)
    os.makedirs(phase_dir, exist_ok=True)

# Helper: parse split files and return list of image basenames
def load_split(split_file_path):
    images = []
    with open(split_file_path, 'r') as f:
        for line in f:
            name = line.strip()
            if not name.lower().endswith('.jpg'):
                name = name + '.jpg'
            images.append(name)
    return images

# Helper: parse XML to get bounding boxes for objects
# Returns list of (label, (xmin, ymin, xmax, ymax))
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objs.append((label, (xmin, ymin, xmax, ymax)))
    return objs

# Identify split files
split_files = os.listdir(SPLITS_DIR)
train_files = [f for f in split_files if 'train' in f.lower()]
test_files = [f for f in split_files if 'test' in f.lower()]

# Load images for each phase
train_images = []
for f in train_files:
    train_images += load_split(os.path.join(SPLITS_DIR, f))
test_images = []
for f in test_files:
    test_images += load_split(os.path.join(SPLITS_DIR, f))

# Crop ROI and organize into folders by class label
for phase, image_list in [('train', train_images), ('test', test_images)]:
    for img_name in image_list:
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMAGES_DIR, img_name)
        xml_path = os.path.join(XML_DIR, img_id + '.xml')
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            print(f"Missing file for {img_name}")
            continue

        # Open image and parse bounding boxes
        image = Image.open(img_path).convert('RGB')
        objs = parse_xml(xml_path)
        # For Stanford40, assume single object per image; pick first
        label, bbox = objs[0]
        xmin, ymin, xmax, ymax = bbox
        cropped = image.crop((xmin, ymin, xmax, ymax))

        # Save cropped ROI into organized folder
        dst_dir = os.path.join(OUTPUT_DIR, phase, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, img_name)
        cropped.save(dst_path)

print("Cropped ROI organization complete.")
