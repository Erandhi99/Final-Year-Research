import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

# Load and clean dataset
filePath = "../data/raw/adhd-new.csv"
df = pd.read_csv(filePath)
df = df.drop(['ID'], axis=1)

df
print(df.isnull().sum())
df.dtypes
df.head()

# Check class distribution
sns.countplot(x=df['Class'])
plt.title("Class Distribution")
plt.show()

# Encode the target variable
df['Class'] = df['Class'].map({'ADHD': 1, 'Control': 0})   # 0 = Control, 1 = ADHD
df

#Select EEG features
eeg_features = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]

#Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df[eeg_features])

features_scaled_df = pd.DataFrame(features_scaled, columns=eeg_features)
features_scaled_df.head()

# Feature-target split
X = features_scaled_df
y = df['Class']

X.shape
y.shape

# Split data: train, validation, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Convert to tensors
def to_tensor_dataset(X, y):
    return TensorDataset(
        torch.tensor(X.to_numpy(copy=True), dtype=torch.float32),
        torch.tensor(y.to_numpy(copy=True), dtype=torch.long)
    )


# Create DataLoaders
train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=128)
test_loader = DataLoader(to_tensor_dataset(X_test, y_test), batch_size=128)

# Define the MLP model
class ADHDClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ADHDClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32, 2)
            
        )

    def forward(self, x):
        return self.net(x)

# Initialize model, loss function, and optimizer
model = ADHDClassifier(input_dim=X.shape[1])  # X from your preprocessed features

# Class weights
adhd_count = (y == 1).sum()
control_count = (y == 0).sum()
total = adhd_count + control_count

adhd_weight = total / (2 * adhd_count)
control_weight = total / (2 * control_count)
class_weights = torch.tensor([control_weight, adhd_weight], dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training with early stopping
train_losses, val_losses = [], []
epochs = 100
best_val_loss = float('inf')
early_stop_count = 0
patience = 20

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
        
         # Compute training accuracy for this batch
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
    scheduler.step()

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_count = 0
        best_model_state = model.state_dict()
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(best_model_state)

# Plot loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss (ADHD)')
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