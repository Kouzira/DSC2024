from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoTokenizer
import torch
from PIL import Image

class MultiMediaDataset(Dataset):
    def __init__(self, images, ocr_text_ids, caption_ids, labels):
        self.images = images
        self.ocr_text = ocr_text_ids
        self.caption_ids = caption_ids
        self.labels = labels
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.ocr_text_ids[index], self.caption_ids[index], self.labels[index]
        
def load_data(images_path, ocr_texts, captions, labels):
    r'''
    return images, ocr_text_ids, caption_ids, labels
    in tensor type
    '''

    # ocr_text max tokens = 191
    # caption max tokens = 224


    # examples:
    # image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")
    # miage_tensor = image_processor(image, return_tensors="pt")["pixel_values"]

    # Phải chạy word segmentation trước
    # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    # sentence = "Chúng_tôi học sinh_học. "" * 300
    # input_ids = torch.tensor([tokenizer.encode(sentence, max_length=256)])
    # print(input_ids.shape)

    
    # Initialize image processor and tokenizer
    # Initialize image processor and tokenizer
    image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    # Process images
    images_tensor = []
    for img_path in images_path:
        image = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
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
        "non-sarcasm": 3
    }
    labels_tensor = torch.tensor([label_map[label] for label in labels])
    
    return images_tensor, ocr_text_ids, caption_ids, labels_tensor
