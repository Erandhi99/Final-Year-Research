import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# Load and clean dataset
filePath = "../data/raw/id-new.csv"
df = pd.read_csv(filePath)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace("'", '').str.replace('-', '_')
df = df.drop(['case_id', 'client_id'], axis=1)

print(df.isnull().sum())
df.dtypes
df.head()

# Check class distribution
sns.countplot(x=df['target'])
plt.title("Class Distribution")
plt.show()

# Encode target
df['target'] = df['target'].replace({-1: 0, 0: 0, 1: 1})

# Upsample minority class
df_majority = df[df.target == 0]
df_minority = df[df.target == 1]

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)

# Standardize features
features = df_balanced.columns.tolist()
features.remove('target')
scaler = StandardScaler()
df_balanced[features] = scaler.fit_transform(df_balanced[features])

# Split features and target
X = df_balanced[features]
y = df_balanced['target']

# Original first split is fine
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Just avoid stratify in the second split
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to tensors
def to_tensor_dataset(X, y):
    return TensorDataset(
        torch.tensor(X.to_numpy(), dtype=torch.float32),
        torch.tensor(y.to_numpy().astype(np.int64))  # Force conversion to int64
    )

# WeightedRandomSampler for DataLoader
class_counts = y_train.value_counts().to_dict()
class_weights = [1.0 / class_counts[c] for c in y_train]
sample_weights = torch.DoubleTensor(class_weights)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoaders
train_dataset = to_tensor_dataset(X_train, y_train)
val_dataset = to_tensor_dataset(X_val, y_val)
test_dataset = to_tensor_dataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Compute class weights for loss function
class_weight_values = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weight_values, dtype=torch.float32)

# Define the MLP model
class ASDClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ASDClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model, loss, optimizer
model = ASDClassifier(X.shape[1])
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with early stopping
train_losses, val_losses = [], []
epochs = 100
best_val_loss = float('inf')
early_stop_count = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracy = correct / total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = val_correct / val_total

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_count = 0
        best_model_state = model.state_dict()
    else:
        early_stop_count += 1
        if early_stop_count >= 10:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(best_model_state)

# Plot loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

# Final Evaluation
model.eval()
y_preds, y_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_preds.extend(predicted.numpy())
        y_true.extend(y_batch.numpy())

# Print results
print(f"\nTest Accuracy: {accuracy_score(y_true, y_preds):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_preds))
print(classification_report(y_true, y_preds, zero_division=0))

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_np = X.to_numpy()
y_np = y.to_numpy()

all_fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_np, y_np)):
    print(f"\nFold {fold + 1}")

    X_train, X_val = X_np[train_idx], X_np[val_idx]
    y_train, y_val = y_np[train_idx], y_np[val_idx]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = ASDClassifier(X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    early_stop = 0

    for epoch in range(50):  # reduced epochs for safety
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Early stopping check
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        y_preds, y_true = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()

                _, pred = torch.max(out, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                y_preds.extend(pred.numpy())
                y_true.extend(yb.numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}, Acc = {acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 7:
                print("Early stopping.")
                break

    print("Fold Accuracy:", accuracy_score(y_true, y_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_preds))
    print(classification_report(y_true, y_preds, zero_division=0))
    all_fold_accuracies.append(acc)

print(f"\nAverage Accuracy over 5 folds: {np.mean(all_fold_accuracies):.4f}")