#!/usr/bin/env python3
"""
Test Script for Quantum Vision Transformer on MNIST and CIFAR-10

This script trains and evaluates the three quantum vision transformer variants:
1. Orthogonal Patch-wise Network
2. Quantum Orthogonal Transformer
3. Compound Transformer

Usage:
    # Train on MNIST with patchwise model
    python test_quantum_vision_transformer.py --dataset mnist --model patchwise --n-layers 2

    # Train on CIFAR-10 with compound model
    python test_quantum_vision_transformer.py --dataset cifar10 --model compound --n-layers 3

    # Use butterfly circuit
    python test_quantum_vision_transformer.py --dataset mnist --circuit-type butterfly

    # Quick test with mini dataset
    python test_quantum_vision_transformer.py --dataset mnist --mini --n-epochs 5
"""

import argparse
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns

from QuantumVisionTransformer import (
    OrthogonalPatchwise,
    QuantumOrthogonalTransformer,
    CompoundTransformer,
    create_quantum_vision_transformer
)


# ============================================================================
# Data Loading
# ============================================================================

def load_mnist(batch_size: int = 32, mini: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset.

    Args:
        batch_size: Batch size
        mini: If True, use small subset for quick testing

    Returns:
        train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create validation split from training data
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    if mini:
        train_dataset = Subset(train_dataset, range(min(500, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(100, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(200, len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def load_cifar10(batch_size: int = 32, mini: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset.

    Args:
        batch_size: Batch size
        mini: If True, use small subset for quick testing

    Returns:
        train_loader, val_loader, test_loader
    """
    transform_train = transforms.Compose([
        transforms.Resize(28),  # Resize to 28x28 to match MNIST
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # Create validation split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    if mini:
        train_dataset = Subset(train_dataset, range(min(500, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(100, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(200, len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number

    Returns:
        Average loss and accuracy
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval"
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on data.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to use
        desc: Description for progress bar

    Returns:
        Average loss, accuracy, predictions, labels
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ============================================================================
# Visualization
# ============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str
):
    """
    Plot training and validation curves.

    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_accs: Training accuracies
        val_accs: Validation accuracies
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


# ============================================================================
# Main Training Function
# ============================================================================

def main(args):
    """Main training and evaluation function."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading {args.dataset.upper()} dataset...")
    if args.dataset == "mnist":
        train_loader, val_loader, test_loader = load_mnist(args.batch_size, args.mini)
        img_size = 28
        n_classes = 10
        class_names = [str(i) for i in range(10)]
    elif args.dataset == "cifar10":
        train_loader, val_loader, test_loader = load_cifar10(args.batch_size, args.mini)
        img_size = 28  # Resized
        n_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    # Create model
    print(f"\nCreating {args.model} model...")
    print(f"  Circuit type: {args.circuit_type}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Number of layers: {args.n_layers}")

    model = create_quantum_vision_transformer(
        model_type=args.model,
        img_size=img_size,
        patch_size=args.patch_size,
        n_layers=args.n_layers,
        n_classes=n_classes,
        circuit_type=args.circuit_type
    )
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, args.n_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, desc=f"Epoch {epoch} [Val]"
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.n_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_path = output_dir / f"best_model_{args.dataset}_{args.model}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Plot training curves
    plot_path = output_dir / f"training_curves_{args.dataset}_{args.model}.png"
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, str(plot_path))

    # Load best model for final evaluation
    print("\n" + "="*70)
    print("Final Evaluation on Test Set")
    print("="*70)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    cm_path = output_dir / f"confusion_matrix_{args.dataset}_{args.model}.png"
    plot_confusion_matrix(test_labels, test_preds, class_names, str(cm_path))

    # Save results summary
    results = {
        'dataset': args.dataset,
        'model': args.model,
        'circuit_type': args.circuit_type,
        'n_layers': args.n_layers,
        'patch_size': args.patch_size,
        'n_params': n_params,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time_minutes': training_time / 60
    }

    results_path = output_dir / f"results_{args.dataset}_{args.model}.txt"
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"\nResults saved to {results_path}")

    print("\n" + "="*70)
    print("Experiment Completed!")
    print("="*70)


# ============================================================================
# Command Line Arguments
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Quantum Vision Transformer on MNIST and CIFAR-10"
    )

    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'cifar10'],
        help='Dataset to use (default: mnist)'
    )
    parser.add_argument(
        '--mini',
        action='store_true',
        help='Use mini dataset for quick testing'
    )

    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='patchwise',
        choices=['patchwise', 'orthogonal', 'compound'],
        help='Model type (default: patchwise)'
    )
    parser.add_argument(
        '--circuit-type',
        type=str,
        default='butterfly',
        choices=['butterfly', 'pyramid', 'x'],
        help='Quantum circuit type (default: butterfly)'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=2,
        help='Number of quantum layers (default: 2)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=7,
        help='Patch size (default: 7)'
    )

    # Training
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )

    # System
    parser.add_argument(
        '--seed',
        type=int,
        default=2025,
        help='Random seed (default: 2025)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage even if CUDA is available'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./qvit_results',
        help='Output directory for results (default: ./qvit_results)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("="*70)
    print("Quantum Vision Transformer Testing")
    print("="*70)
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*70)

    main(args)
