from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoTokenizer
# import 

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
        
def load_data():
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

    TODO()
    return None