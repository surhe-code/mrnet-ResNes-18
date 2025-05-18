import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import numpy as np
import argparse
import os
import time
import random

# Import from local files
from model import MRNet_ResNet18 # MODIFIED: Import MRNet_ResNet18
from dataloader import get_dataloaders, pad_collate_fn # Ensure pad_collate_fn is importable if get_dataloaders doesn't return it

def evaluate_model(model, dataloader, criterion, device):
    """Helper function to evaluate model on a given dataloader."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data['volume'].to(device)
            labels = batch_data['label'].to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int)) # Standard threshold for other metrics

    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Ensure there are both classes present for AUC, otherwise it's ill-defined or 0.5
    unique_labels = np.unique(np.array(all_labels).flatten())
    if len(unique_labels) < 2:
        epoch_auc = 0.5 # Or handle as an error/warning, common for very small/imbalanced val sets
        print(f"Warning: Only one class ({unique_labels}) present in labels. AUC set to 0.5.")
    else:
        epoch_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
    
    accuracy = accuracy_score(np.array(all_labels), np.array(all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(np.array(all_labels), np.array(all_preds), average='binary', zero_division=0)

    print(f"Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}, Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
    return epoch_loss, epoch_auc, accuracy, precision, recall, f1


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True # Important for reproducibility
        torch.backends.cudnn.benchmark = False    # Set to False for reproducibility if input sizes vary, True if fixed for speed

    train_loader, valid_loader, train_dataset = get_dataloaders(
        root_dir=args.dataset_root,
        task=args.task,
        plane=args.plane,
        batch_size=args.batch_size,
        num_workers=args.num_workers
        # collate_fn=pad_collate_fn # This is now set inside get_dataloaders
    )
    print(f"Training on {args.task} task, {args.plane} plane using ResNet18.")

    model = MRNet_ResNet18(n_classes=1, pretrained=True) # MODIFIED: Instantiate MRNet_ResNet18
    model = model.to(device)

    labels_for_weight = np.array(train_dataset.data_info['label'].tolist())
    num_positive = np.sum(labels_for_weight)
    num_negative = len(labels_for_weight) - num_positive
    pos_weight_val = num_negative / num_positive if num_positive > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
    print(f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW([
        {'params': model.base_model.parameters(), 'lr': args.lr_backbone},
        {'params': model.fc.parameters(), 'lr': args.lr_fc}
    ], weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=args.patience, verbose=True) # Maximize AUC

    best_val_auc = 0.0
    epochs_no_improve = 0
    
    save_dir = os.path.join(args.save_model_dir, args.task, args.plane)
    os.makedirs(save_dir, exist_ok=True)
    # MODIFIED: Model name includes "resnet18"
    best_model_filename = f"best_model_resnet18_{args.task}_{args.plane}.pth"
    best_model_path = os.path.join(save_dir, best_model_filename)

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        running_train_loss = 0.0
        train_labels_all, train_preds_all, train_probs_all = [], [], []

        for i, batch_data in enumerate(train_loader):
            inputs = batch_data['volume'].to(device)
            labels = batch_data['label'].to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            train_labels_all.extend(labels.cpu().numpy())
            train_probs_all.extend(torch.sigmoid(outputs).detach().cpu().numpy())

            if (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_auc = roc_auc_score(np.array(train_labels_all), np.array(train_probs_all)) if len(set(np.array(train_labels_all).flatten())) > 1 else 0.5
        
        # Validation Phase
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Validation:")
        val_loss, val_auc, val_acc, val_p, val_r, val_f1 = evaluate_model(model, valid_loader, criterion, device)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] completed in {epoch_duration:.2f}s")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train AUC: {epoch_train_auc:.4f}")
        # Validation metrics are printed by evaluate_model

        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            print(f"Validation AUC improved from {best_val_auc:.4f} to {val_auc:.4f}. Saving model to {best_model_path}")
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation AUC ({val_auc:.4f}) did not improve from best ({best_val_auc:.4f}). Epochs without improvement: {epochs_no_improve}")

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement.")
            break
            
    print("\nTraining finished.")
    print(f"Best Validation AUC achieved during training: {best_val_auc:.4f}")
    print(f"Best model saved to: {best_model_path}")

    # ADDED: Final evaluation of the best saved model on the validation set
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final evaluation on validation set...")
        # Re-initialize model architecture and load state_dict
        final_model = MRNet_ResNet18(n_classes=1, pretrained=False) # pretrained=False as we are loading weights
        final_model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_model = final_model.to(device)
        print("Final evaluation metrics on the validation set (using best saved model):")
        evaluate_model(final_model, valid_loader, criterion, device)
    else:
        print(f"Warning: Best model path {best_model_path} not found. Skipping final evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MRNet with ResNet18 backbone.") # MODIFIED description
    
    parser.add_argument('--dataset_root', type=str, default='MRNet-v1.0/', help="Path to the MRNet dataset root.")
    parser.add_argument('--task', type=str, required=True, choices=['abnormal', 'acl', 'meniscus'], help="Task.")
    parser.add_argument('--plane', type=str, required=True, choices=['sagittal', 'coronal', 'axial'], help="MRI plane.")
    # MODIFIED: Default save directory to reflect ResNet18
    parser.add_argument('--save_model_dir', type=str, default='./models_resnet18', help="Directory to save models.")

    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    # Batch size might need to be adjusted for ResNet18 vs ResNet50, but usually ResNet18 allows larger batch sizes
    # For now, keeping it the same, user might need to adjust based on their GPU for ResNet18
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size.")
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help="LR for ResNet18 backbone.")
    parser.add_argument('--lr_fc', type=float, default=1e-4, help="LR for the new FC layer.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay.")
    parser.add_argument('--patience', type=int, default=5, help="Patience for ReduceLROnPlateau.")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help="Patience for early stopping.")
    
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--log_interval', type=int, default=10, help="Log interval (batches).")

    args = parser.parse_args()
    
    # Example run command:
    # python train.py --task acl --plane sagittal --dataset_root /path/to/MRNet-v1.0/ --epochs 30 --batch_size 16
    # Note: Batch size can likely be increased for ResNet18 compared to ResNet50 on the same GPU.
    
    train_model(args)
