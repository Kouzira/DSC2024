import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from .model import MultiModalClassifier
from .data import MultiMediaDataset, load_data


warmup_epochs = 5
def warmup_lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

def train(
    model: MultiModalClassifier,
    train_loader: DataLoader,
    optimizer: optim,
    base_lr_scheduler: torch.optim.lr_scheduler,
    warmup_lr_scheduler: torch.optim.lr_scheduler,
    epochs: int,
    device: torch.device = None
):
    model.train()
    if (device is not None):
        model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_count, (batch_image, batch_ocr_text_ids, batch_caption_ids, batch_labels) in enumerate(train_loader):
            batch_image = batch_image.to(device)
            batch_ocr_text_ids = batch_ocr_text_ids.to(device)
            batch_caption_ids = batch_caption_ids.to(device)
            batch_labels = batch_labels.to(device)

            pred = model(batch_image, batch_ocr_text_ids, batch_caption_ids)
            batch_loss = nn.CrossEntropyLoss(pred, batch_labels)
            total_loss += batch_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print(f"{batch_count + 1}/{len(train_loader)}: loss {total_loss / (batch_count + 1)}", sep='\r')

        print('\n')
        if epoch < warmup_epochs:
            warmup_lr_scheduler.step()
            print(f"lr: {warmup_lr_scheduler.get_last_lr()}")
        else:
            base_lr_scheduler.step()
            print(f"lr: {base_lr_scheduler.get_last_lr()}")
            

def evaluate(
    model: MultiModalClassifier,
    val_loader: DataLoader,
    device: torch.device = None
):
    model.eval()
    if (device is not None):
        model.to(device)

    total_loss = 0
    with torch.no_grad():
        for batch_count, (batch_image, batch_ocr_text_ids, batch_caption_ids, batch_labels) in enumerate(val_loader):
            batch_image = batch_image.to(device)
            batch_ocr_text_ids = batch_ocr_text_ids.to(device)
            batch_caption_ids = batch_caption_ids.to(device)
            batch_labels = batch_labels.to(device)

            pred = model(batch_image, batch_ocr_text_ids, batch_caption_ids)
            batch_loss = nn.CrossEntropyLoss(pred, batch_labels)
            total_loss += batch_loss

            print(f"{batch_count + 1}/{len(val_loader)}: loss {total_loss / (batch_count + 1)}", sep='\r')
        print('\n')
    
    return total_loss / len(val_loader)

def k_fold_cross_validation(
    k: int,
    model: MultiModalClassifier,
    dataset: MultiMediaDataset,
    batch_size: int,
    epochs: int,
    optimizer: optim,
    base_lr_scheduler: torch.optim.lr_scheduler,
    warmup_lr_scheduler: torch.optim.lr_scheduler,
    device: torch.device = None
):
    kfold = KFold(n_splits=k, shuffle=True)

    total_val_loss = 0
    for fold_count, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print("=" * 10 + f"\nFold {fold_count + 1}/{k}")
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size, num_workers=4)

        train(model, 
              train_loader, 
              optimizer,
              base_lr_scheduler,
              warmup_lr_scheduler,
              epochs,
              device)
        
        total_val_loss += evaluate(model, val_loader, device)

    print(f"\n== AVG val loss: {total_val_loss / k} ==\n")


if __name__ == "__main__":
    images, ocr_text_ids, caption_ids, labels = load_data()
    dataset = MultiMediaDataset(images, ocr_text_ids, caption_ids, labels)
    batch_size = 128
    epochs = 50
    
    model = MultiModalClassifier()

    # optimizer with different lr for pretrained and untrained modules    
    lr_for_pretrained = 1e-5
    lr_for_untrained = 1e3
    weight_decay = 1e-3
    optimizer = optim.adamw.AdamW(
        [
            {"params": model.params["pretrained"].parameters(), "lr": lr_for_pretrained},
            {"params": model.params["untrained"].parameters(), "lr": lr_for_untrained}
        ], 
        lr=lr_for_pretrained, weight_decay=weight_decay)

    # lr scheduler
    warmup_scheduler = nn.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    base_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    k_fold_cross_validation(k = 5, 
                            model=model, 
                            dataset=dataset, 
                            batch_size=batch_size,
                            epochs=epochs,
                            optimizer=optimizer,
                            base_lr_scheduler=base_lr_scheduler,
                            warmup_lr_scheduler=warmup_scheduler)
