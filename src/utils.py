# !pip install py_vncorenlp
# !apt install openjdk-21-jdk openjdk-21-jre -y
import os
import json
import torch
import easyocr
import kagglehub
import py_vncorenlp
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    SequentialLR, 
    LinearLR, 
    ExponentialLR
)
from model import MultiModalClassifier


label_names = ["multi-sarcasm", 
               "not-sarcasm", 
               "image-sarcasm", 
               "text-sarcasm"]

vncorenlp_path = '/tmp/vncorenlp'
if (not os.path.exists(vncorenlp_path)):
    os.mkdir(vncorenlp_path)
py_vncorenlp.download_model(save_dir=vncorenlp_path)
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")
OCRreader = easyocr.Reader(['vi'], gpu=True)


def predict_on_test(
    model: MultiModalClassifier,
    testdata_dir: str,
    output_path: str,
    device
):
    model.eval()
    model = model.to(device)

    json_input_path = f"{testdata_dir}/vimmsd-public-test.json"
    image_input_folder = f"{testdata_dir}/public-test-images/dev-images"

    with open(json_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {"results": {}, "phase": "dev"}
    
    for sample_cnt, (key, value) in enumerate(data.items()):
        image_name = value['image']
        image_path = os.path.join(image_input_folder, image_name)

        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image_tensor = image_processor(image, return_tensors="pt")["pixel_values"]

            # OCR
            result = OCRreader.readtext(image_path, detail=0)
            text = "\n".join(result)
            segmented_ocr_text = "\n".join(rdrsegmenter.word_segment(text))
            ocr_ids = tokenizer.encode(segmented_ocr_text, max_length=191, padding="max_length", truncation=True, return_tensors="pt")
            
            # Caption
            segmented_caption_text = "\n".join(rdrsegmenter.word_segment(value['caption']))
            caption_ids = tokenizer.encode(segmented_caption_text, max_length=224, padding="max_length", truncation=True, return_tensors="pt")

            image_tensor = image_tensor.to(device)
            ocr_ids = ocr_ids.to(device)
            caption_ids = ocr_ids.to(device)
            with torch.no_grad():
                pred = model(image_tensor, ocr_ids, caption_ids)
                pred_label = torch.argmax(pred, dim=1).item()
                label_name = label_names[pred_label]
                results["results"][key] = label_name
            print(f"{sample_cnt + 1}/{len(data)}", end='\r')
        else:
            notfoundcnt += 1
            print(f"\n{image_name} not found ({notfoundcnt}).")
    with open(output_path, 'w') as fout:
        json.dump(results, fout)


def get_optimizer(model, lr_for_pretrained, lr_for_untrained):
    # optimizer with different lr for pretrained and untrained modules    
    weight_decay = 1e-5
    optimizer = AdamW(
        [
            {"params": model.params["pretrained"].parameters(), "lr": lr_for_pretrained},
            {"params": model.params["untrained"].parameters(), "lr": lr_for_untrained}
        ], 
        lr=lr_for_pretrained, weight_decay=weight_decay)
    return optimizer

def get_lr_scheduler(optimizer, decay_rate, warmup_epochs, batchs):
    warmup_steps = warmup_epochs * batchs
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    decay_scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    # combine the warm-up and decay schedulers with SequentialLR
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])

def download_dataset():
    trainset_path = kagglehub.dataset_download("tmaitn/uitdsc24-train-dataset")
    testset_path = kagglehub.dataset_download("longnguynvhong/dsc2024-public-test")
    trainset_path = os.path.join(trainset_path, "uitdsc_train_dataset")
    print("Path to trainset files:", trainset_path)
    print("Path to testset files:", testset_path)
    return trainset_path, testset_path


def load_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])


def load_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler, history):   
    state_dict_path = os.path.join(checkpoint_dir, "checkpoint_0.pth")
    history_path = os.path.join(checkpoint_dir, "history.json") 

    checkpoint = torch.load(state_dict_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if (os.path.exists(history_path)):
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            history[0] = data[0]
            history[1] = data[1]


def save_checkpoint(checkpoint_dir, model, optimizer, scheduler, epoch, history):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict_path = os.path.join(checkpoint_dir, "checkpoint_0.pth")
    history_path = os.path.join(checkpoint_dir, "history.json")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, state_dict_path)
    with open(history_path, 'w') as fout:
        json.dump(history, fout)
    print("Checkpoint saved.")


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time the validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
