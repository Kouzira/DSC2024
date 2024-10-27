import os
import json
import torch
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoTokenizer

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Extract data from JSON
def extract_data_from_json(json_data, image_folder):
    images = []
    captions = []
    labels = []
    extracted = []
    ids = []

    for key, item in json_data.items():
        img_path = os.path.join(image_folder, item['image'])
        images.append(img_path)
        captions.append(item['caption'])
        labels.append(item['label'])
        extracted_text = item.get('extracted_text', "")
        extracted.append(extracted_text)
        ids.append(key)  # Store the ID

    return images, captions, labels, extracted, ids

# Load and process data
def load_data(images_path, ocr_texts, captions, labels):
    # Initialize image processor and tokenizer
    image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    # Process images
    images_tensor = []
    for img_path in images_path:
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format
        image_tensor = image_processor(image, return_tensors="pt")["pixel_values"]
        images_tensor.append(image_tensor)
    images_tensor = torch.cat(images_tensor, dim=0)

    # Process OCR texts
    ocr_text_ids = []
    for ocr_text in ocr_texts:
        input_ids = tokenizer.encode(ocr_text, max_length=191, padding="max_length", truncation=True, return_tensors="pt")
        ocr_text_ids.append(input_ids)
    ocr_text_ids = torch.cat(ocr_text_ids, dim=0)

    # Process captions
    caption_ids = []
    for caption in captions:
        input_ids = tokenizer.encode(caption, max_length=224, padding="max_length", truncation=True, return_tensors="pt")
        caption_ids.append(input_ids)
    caption_ids = torch.cat(caption_ids, dim=0)

    # Encode labels as numerical values
    label_map = {
        "multi-sarcasm": 0,
        "text-sarcasm": 1,
        "image-sarcasm": 2,
        "not-sarcasm": 3
    }
    labels_tensor = torch.tensor([label_map[label] for label in labels])
    
    return images_tensor, ocr_text_ids, caption_ids, labels_tensor

# Save features to .pt files in label-specific folders
def save_features(features, ids, labels, base_folder_path):
    for feature, id, label in zip(features, ids, labels):
        # Create a folder for the label if it doesn't exist
        label_folder_path = os.path.join(base_folder_path, label)
        os.makedirs(label_folder_path, exist_ok=True)

        feature_file_path = os.path.join(label_folder_path, f"{id}.pt")
        torch.save(feature, feature_file_path)
        print(f"Saved feature to {feature_file_path}")

# Load features from .pt files
def load_features(folder_path):
    loaded_features = {}
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                if file_name.endswith('.pt'):
                    feature_id = file_name[:-3]  # Remove '.pt' to get the ID
                    feature_tensor = torch.load(os.path.join(label_path, file_name))
                    loaded_features[feature_id] = feature_tensor
                    print(f"Loaded feature from {os.path.join(label_path, file_name)}")
    return loaded_features

# Main code
# File JSON chứa dữ liệu
json_file = "extracted_text_train.json"
image_folder = "train-images"  # Đường dẫn đến thư mục chứa ảnh
output_folder = "features"  # Folder to save features

# Read and extract data from JSON
json_data = load_json(json_file)
images, captions, labels, extracted_text, ids = extract_data_from_json(json_data, image_folder)

# Load and process data
images_tensor, ocr_text_ids, caption_ids, labels_tensor = load_data(images, extracted_text, captions, labels)

# Save features to .pt files in label-specific folders
save_features(images_tensor, ids, labels, os.path.join(output_folder, "images"))
save_features(ocr_text_ids, ids, labels, os.path.join(output_folder, "ocr_texts"))
save_features(caption_ids, ids, labels, os.path.join(output_folder, "captions"))
save_features(labels_tensor.unsqueeze(1), ids, labels, os.path.join(output_folder, "labels"))  # Unsqueeze to match the dimension

# Load features from .pt files
loaded_images = load_features(os.path.join(output_folder, "images"))
loaded_ocr_texts = load_features(os.path.join(output_folder, "ocr_texts"))
loaded_captions = load_features(os.path.join(output_folder, "captions"))
loaded_labels = load_features(os.path.join(output_folder, "labels"))
