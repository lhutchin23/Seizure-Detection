import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from Dataloader import preprocess
from modelarchitecture import EEG_CNN_Learnable
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
NUM_SCALES = 64
SIGNAL_LENGTH = 178

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading and preprocessing data...")
X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

model = EEG_CNN_Learnable(
    num_scales=NUM_SCALES, signal_length=SIGNAL_LENGTH, dropout_rate=DROPOUT_RATE
).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_acc = 0.0
best_epoch = 0

# print("Training")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    train_preds = []
    train_labels = []

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(batch_y.cpu().numpy())

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    # Calculate training metrics
    avg_train_loss = epoch_loss / len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(batch_y.cpu().numpy())

    # Calculate validation metrics
    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(f"\n{'='*70}")
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_acc:.4f}")

    # Get learned wavelet parameters
    learned_params = model.get_learned_params()
    print(f"Centre Freq: {learned_params['centre_freq']:.4f}")
    print(f"Bandwidth Freq: {learned_params['bandwidth_freq']:.4f}")
    print(f"Threshold: {learned_params['threshold']:.4f}")

    # Save checkpoint for this epoch
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learned_params": learned_params,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    # Track best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "learned_params": learned_params,
            },
            best_model_path,
        )

print(f"Best Validation Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}")

best_checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"))
model.load_state_dict(best_checkpoint["model_state_dict"])
model.eval()

val_preds = []
val_labels = []
val_probs = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(batch_y.cpu().numpy())
        val_probs.extend(probs.cpu().numpy())

val_preds = np.array(val_preds)
val_labels = np.array(val_labels)
val_probs = np.array(val_probs)

accuracy = accuracy_score(val_labels, val_preds)
f1 = f1_score(val_labels, val_preds, average="binary")
precision = precision_score(val_labels, val_preds, average="binary")
recall = recall_score(val_labels, val_preds, average="binary")

print("Best Model Performance on Validation Set:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Print classification report
print(
    classification_report(
        val_labels, val_preds, target_names=["Non-Seizure", "Seizure"]
    )
)

# Generate Confusion Matrix
cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Non-Seizure", "Seizure"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title(
    f"Confusion Matrix - Learnable Wavelet Layer\nBest Epoch: {best_epoch} | Val Acc: {best_val_acc:.4f}"
)
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_best_epoch.png")
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nConfusion matrix saved to: {cm_path}")

# Generate F1 Score Matrix (per-class metrics)
f1_per_class = f1_score(val_labels, val_preds, average=None)
precision_per_class = precision_score(val_labels, val_preds, average=None)
recall_per_class = recall_score(val_labels, val_preds, average=None)

# Create a metrics table
metrics_data = np.array(
    [
        [precision_per_class[0], recall_per_class[0], f1_per_class[0]],
        [precision_per_class[1], recall_per_class[1], f1_per_class[1]],
    ]
)

plt.figure(figsize=(8, 4))
sns.heatmap(
    metrics_data,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    xticklabels=["Precision", "Recall", "F1-Score"],
    yticklabels=["Non-Seizure", "Seizure"],
)
plt.title(f"Per-Class Metrics - Learnable Wavelet Layer\nBest Epoch: {best_epoch}")
plt.tight_layout()
metrics_path = os.path.join(RESULTS_DIR, "f1_metrics_best_epoch.png")
plt.savefig(metrics_path, dpi=200, bbox_inches="tight")
plt.close()

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss", marker="o")
axes[0].plot(range(1, NUM_EPOCHS + 1), val_losses, label="Val Loss", marker="s")
axes[0].axvline(
    x=best_epoch, color="r", linestyle="--", label=f"Best Epoch ({best_epoch})"
)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(range(1, NUM_EPOCHS + 1), train_accs, label="Train Acc", marker="o")
axes[1].plot(range(1, NUM_EPOCHS + 1), val_accs, label="Val Acc", marker="s")
axes[1].axvline(
    x=best_epoch, color="r", linestyle="--", label=f"Best Epoch ({best_epoch})"
)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training and Validation Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
history_path = os.path.join(RESULTS_DIR, "training_history.png")
plt.savefig(history_path, dpi=200, bbox_inches="tight")
plt.close()

learned_params = best_checkpoint["learned_params"]
params_path = os.path.join(RESULTS_DIR, "learned_wavelet_params.txt")
with open(params_path, "w") as f:
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Validation Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Validation F1 Score: {f1:.4f}\n\n")
    f.write("Learned Wavelet Parameters:\n")
    f.write(f"Centre Frequency: {learned_params['centre_freq']:.6f}\n")
    f.write(f"Bandwidth Frequency: {learned_params['bandwidth_freq']:.6f}\n")
    f.write(f"Denoising Threshold: {learned_params['threshold']:.6f}\n")
    f.write(f"\nScale Range:\n")
    f.write(f"Min Scale: {learned_params['scales'].min():.6f}\n")
    f.write(f"Max Scale: {learned_params['scales'].max():.6f}\n")

print(f"\nCheckpoints directory: {CHECKPOINT_DIR}/")
print(f"Results directory: {RESULTS_DIR}/")
print(f"Confusion matrix: {cm_path}")
print(f"F1 metrics: {metrics_path}")
print(f"Training history: {history_path}")
print(f"Learned parameters: {params_path}")
