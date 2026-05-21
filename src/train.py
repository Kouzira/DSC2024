import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data import MultiMediaDataset
from model import MultiModalClassifier
from utils import (
    get_optimizer,
    save_checkpoint,
    save_best_checkpoint,
    load_checkpoint,
    download_dataset,
    predict_on_test,
    get_lr_scheduler,
    EarlyStopping,
    compute_metrics,
    format_metrics,
)



def train(
    model: MultiModalClassifier,
    train_loader: DataLoader,
    optimizer: optim,
    lr_scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    loss_fn,
    device,
):
    model.train()
    model = model.to(device)
    total_loss = 0
    for batch_count, (batch_image, batch_ocr_text_ids, batch_caption_ids, batch_labels) in enumerate(train_loader):
        batch_image = batch_image.to(device)
        batch_ocr_text_ids = batch_ocr_text_ids.to(device)
        batch_caption_ids = batch_caption_ids.to(device)
        batch_labels = batch_labels.to(device)

        with torch.amp.autocast("cuda"):
            pred = model(batch_image, batch_ocr_text_ids, batch_caption_ids)
            batch_loss = loss_fn(pred, batch_labels)
        total_loss += batch_loss.item()

        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        print(f"{batch_count + 1}/{len(train_loader)}: loss {total_loss / (batch_count + 1)}| lr: {lr_scheduler.get_last_lr()}" + " "*40, end='\r')
    print('')
    return total_loss / len(train_loader)


def evaluate(
    model: MultiModalClassifier,
    val_loader: DataLoader,
    loss_fn,
    device,
):
    """Evaluate model on val_loader. Returns (avg_loss, metrics_dict)."""
    model.eval()
    model = model.to(device)

    total_loss = 0
    all_preds = []
    all_labels = []
    for batch_count, (batch_image, batch_ocr_text_ids, batch_caption_ids, batch_labels) in enumerate(val_loader):
        batch_image = batch_image.to(device)
        batch_ocr_text_ids = batch_ocr_text_ids.to(device)
        batch_caption_ids = batch_caption_ids.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            logits = model(batch_image, batch_ocr_text_ids, batch_caption_ids)
            batch_loss = loss_fn(logits, batch_labels)

        total_loss += batch_loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(batch_labels.cpu().tolist())

        print(f"{batch_count + 1}/{len(val_loader)}: loss {total_loss / (batch_count + 1)}" + " "*40, end='\r')
    print('\n')

    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = 20
    epochs = 100
    warmup_epochs = 5

    # split dataset (was 0.5/0.5 — now 0.85/0.15 to leave more data for training)
    dataset_path, testset_path = download_dataset()
    dataset = MultiMediaDataset(dataset_path)
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [0.85, 0.15], generator=split_generator)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

    loss_fn = torch.nn.CrossEntropyLoss()  # no need to .to(device); CE has no learnable params

    # init
    lr_for_pretrained = 5e-6
    lr_for_untrained = 5e-5
    batchs = len(train_loader)
    model = MultiModalClassifier().to(device)
    optimizer = get_optimizer(model, lr_for_pretrained, lr_for_untrained)
    lr_scheduler = get_lr_scheduler(optimizer, 0.96, warmup_epochs, batchs)
    early_stopping = EarlyStopping(patience=10)
    scaler = torch.amp.GradScaler("cuda") # mixed precision
    history = [[], []]
    last_epoch = -1
    

    # load
    loading = False
    checkpoint_dir = os.path.join(os.path.expanduser("~"), "DSC2024", "checkpoint")
    if loading:
        last_epoch = load_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler, history)
        print(f"Load from checkpoint. Last epoch: {last_epoch}, last val loss: {history[1][-1]}")


    train_history = history[0]
    val_history = history[1]
    metrics_history = []
    best_f1 = -1.0
    for epoch in range(last_epoch + 1, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # train
        train_loss = train(model, train_loader, optimizer, lr_scheduler, scaler, loss_fn, device)
        train_history.append(train_loss)

        # val
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_history.append(val_loss)
        val_metrics["epoch"] = epoch
        val_metrics["val_loss"] = val_loss
        val_metrics["train_loss"] = train_loss
        metrics_history.append(val_metrics)
        print(f"Val loss: {val_loss:.4f}")
        print(format_metrics(val_metrics))

        history = [train_history, val_history]
        save_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler, epoch, history, metrics_history)

        # Track best by F1 macro (more informative than loss for imbalanced classes).
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_best_checkpoint(checkpoint_dir, model, epoch, val_metrics)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    result_path = os.path.join(os.path.expanduser("~"), "DSC2024", "results.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    predict_on_test(model, testset_path, result_path, device)