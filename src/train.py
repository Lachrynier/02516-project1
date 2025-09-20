import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from IPython.display import clear_output

from src import data
from src.utils import DEVICE
from src.logger import setup_logger
from src.checkpoints import load_checkpoint, save_checkpoint, clear_checkpoints

LOSS_FUN = nn.BCEWithLogitsLoss()


def train_one_epoch(model, train_loader, optimizer):
    model.train()
    running_loss, correct = [], 0

    for data_batch, target in tqdm(train_loader, total=len(train_loader), leave=False):
        data_batch, target = data_batch.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data_batch).squeeze()
        loss = LOSS_FUN(output, target.float())
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        predicted = torch.sigmoid(output) > 0.5
        correct += (target.bool() == predicted).sum().cpu().item()

    return np.mean(running_loss), correct


def evaluate(model, loader, dataset):
    model.eval()
    losses, correct = [], 0
    import torch
    with torch.no_grad():
        for data_batch, target in loader:
            data_batch, target = data_batch.to(DEVICE), target.to(DEVICE)
            output = model(data_batch).squeeze()

            losses.append(LOSS_FUN(output, target.float()).cpu().item())
            predicted = torch.sigmoid(output) > 0.5
            correct += (target.bool() == predicted).sum().cpu().item()

    return np.mean(losses), correct / len(dataset)


def plot_history(history, title=None, filename=None):
    clear_output(wait=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(epochs, history['train_loss'], label="train")
    ax[0].plot(epochs, history['test_loss'], label="test")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(epochs, history['train_acc'], label="train")
    ax[1].plot(epochs, history['test_acc'], label="test")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    ax[1].grid(True)

    if title:
        fig.suptitle(title)

    if filename:
        plot_path = os.path.join("./figs", f"{filename}.pdf")
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")

    plt.show()
    plt.pause(0.001)


def train(model, optimizer, num_epochs, plot=False, save=False, restart=False, transform=None):
    trainset, testset, train_loader, test_loader = data.make_datasets_and_dataloaders(transform=transform)
    model.to(DEVICE)

    model_name = model.__class__.__name__
    log = setup_logger(model_name)

    if restart:
        clear_checkpoints(model)
        start_epoch, history = 0, {'train_acc': [], 'test_acc': [],
                                   'train_loss': [], 'test_loss': []}
    else:
        start_epoch, history = load_checkpoint(model, optimizer)

    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs), unit="epoch"):
        train_loss, train_correct = train_one_epoch(model, train_loader, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, testset)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_correct / len(trainset))
        history['test_acc'].append(test_acc)

        if plot:
            plot_history(history, title=model_name, filename=model_name)
        
        log.info(
            f"Epoch {epoch+1}/{start_epoch + num_epochs} | "
            f"Loss train: {train_loss:.3f} test: {test_loss:.3f} | "
            f"Acc train: {history['train_acc'][-1]*100:.1f}% "
            f"test: {test_acc*100:.1f}%"
        )
    
    if save:
        save_checkpoint(model, optimizer, start_epoch + num_epochs, history)

