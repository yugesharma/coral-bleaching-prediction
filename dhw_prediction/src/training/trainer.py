import torch
import torch.nn as nn
import numpy as np
import json
import os


def train_epoch(model, loader, optimizer, loss_fn, device, model_type):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        if model_type == 'mlp':
            x_tab, y = batch
            x_tab, y = x_tab.to(device), y.to(device)
            preds = model(x_tab)

        elif model_type == 'cnn':
            x_seq, x_tab, y = batch
            x_seq  = x_seq.to(device)
            x_tab  = x_tab.to(device)
            y      = y.to(device)
            preds, _ = model(x_seq, x_tab)

        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, model_type):
    model.eval()
    total_loss = 0
    all_probs  = []
    all_labels = []

    for batch in loader:
        if model_type == 'mlp':
            x_tab, y = batch
            x_tab, y = x_tab.to(device), y.to(device)
            preds = model(x_tab)

        elif model_type == 'cnn':
            x_seq, x_tab, y = batch
            x_seq  = x_seq.to(device)
            x_tab  = x_tab.to(device)
            y      = y.to(device)
            preds, _ = model(x_seq, x_tab)

        loss = loss_fn(preds, y)
        total_loss += loss.item()

        all_probs.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    return total_loss / len(loader), probs, labels


def train(model, train_loader, val_loader, optimizer, loss_fn,
          n_epochs, patience, device, model_type, save_path):

    best_val_loss = np.inf
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, model_type
        )
        val_loss, val_probs, val_labels = eval_epoch(
            model, val_loader, loss_fn, device, model_type
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"           ↳ Best model saved")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path))

    return model, history


def save_history(history, output_dir):
    path = os.path.join(output_dir, 'history.json')
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {path}")
