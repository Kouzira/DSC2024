import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data import MultiMediaDataset
from model import MultiModalClassifier
from utils import get_optimizer, save_checkpoint, load_checkpoint, download_dataset



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
        lr_scheduler.step()
        batch_image = batch_image.to(device)
        batch_ocr_text_ids = batch_ocr_text_ids.to(device)
        batch_caption_ids = batch_caption_ids.to(device)
        batch_labels = batch_labels.to(device)

        with torch.amp.autocast("cuda"):
            pred = model(batch_image, batch_ocr_text_ids, batch_caption_ids)
            batch_loss = loss_fn(pred, batch_labels)
        total_loss += batch_loss

        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"{batch_count + 1}/{len(train_loader)}: loss {total_loss / (batch_count + 1)}| lr: {lr_scheduler.get_last_lr()}" + " "*40, end='\r')
    print('\n')
    return total_loss / len(train_loader)


def evaluate(
    model: MultiModalClassifier,
    val_loader: DataLoader,
    loss_fn,
    device,
):
    model.eval()
    model = model.to(device)

    total_loss = 0
    for batch_count, (batch_image, batch_ocr_text_ids, batch_caption_ids, batch_labels) in enumerate(val_loader):
        batch_image = batch_image.to(device)
        batch_ocr_text_ids = batch_ocr_text_ids.to(device)
        batch_caption_ids = batch_caption_ids.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            pred = model(batch_image, batch_ocr_text_ids, batch_caption_ids)
            batch_loss = loss_fn(pred, batch_labels)
        total_loss += batch_loss
        print(f"{batch_count + 1}/{len(val_loader)}: loss {total_loss / (batch_count + 1)}" + " "*40, end='\r')
    print('\n')
    return total_loss / len(val_loader)

if __name__ == "__main__":   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    
    batch_size = 12
    epochs = 40
    warmup_epochs = 5

    # split dataset
    dataset_path, testset_path = download_dataset()
    dataset = MultiMediaDataset(dataset_path)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model = MultiModalClassifier().to(device)
    optimizer, lr_scheduler = get_optimizer(model, epochs, warmup_epochs, len(train_loader))

    # mixed precision
    scaler = torch.amp.GradScaler("cuda")

    
    last_epoch = -1
    history = [[], []]
    loading = False
    checkpoint_dir = "/kaggle/working/checkpoint/"
    if loading:
        load_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler, history)
        print(f"Load from checkpoint. Last epoch: {len(history[0])}, last loss: {history[1][-1]}")


    train_history = history[0]
    val_history = history[1]
    for epoch in range(last_epoch + 1, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # train
        train_loss = train(
            model, train_loader, optimizer, 
            lr_scheduler, scaler, loss_fn, device
        )
        train_history.append(train_loss.item())

        # val
        val_loss = evaluate(model, val_loader, loss_fn, device)
        val_history.append(val_loss)

        history = [train_history, val_history]
        save_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler, epoch, history)

    # predict_on_test(model, epoch, device)