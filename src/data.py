from torch.utils.data import Dataset
from transformers import AutoImageProcessor
import torch
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

class MultiMediaDataset(Dataset):
    def __init__(self, rootdir: str):
        self.encode_label = {"multi-sarcasm": 0,
                            "not-sarcasm": 1,
                            "image-sarcasm": 2,
                            "text-sarcasm": 3}
        # get size
        self.size = 0
        for class_name in os.listdir(rootdir):
            class_path = os.path.join(rootdir, class_name)
            if (not os.path.isdir(class_path)):
                continue
            for sample_id in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample_id)
                if (os.path.isdir(sample_path)):
                    self.size += 1

        # init
        self.image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")
        self.image_paths = [0 for i in range(self.size)]
        self.ocr_paths = [0 for i in range(self.size)]
        self.caption_paths = [0 for i in range(self.size)]
        self.labels = [0 for i in range(self.size)]

        # load sample paths
        for class_name in os.listdir(rootdir):
            class_path = os.path.join(rootdir, class_name)
            if (not os.path.isdir(class_path)):
                continue
            
            for sample_id in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample_id)
                if (not os.path.isdir(sample_path)):
                    continue

                sample_idx = int(sample_id)
                self.image_paths[sample_idx] = os.path.join(sample_path, f"image.jpg")
                self.ocr_paths[sample_idx] = os.path.join(sample_path, f"ocr.pt")
                self.caption_paths[sample_idx] = os.path.join(sample_path, f"caption.pt")
                self.labels[sample_idx] = self.encode_label[class_name]

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index]).convert('RGB')
        image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        return image_tensor,\
            torch.load(self.ocr_paths[index])[0],\
            torch.load(self.caption_paths[index])[0],\
            self.labels[index]