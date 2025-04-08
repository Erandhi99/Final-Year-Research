import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

# Load and clean dataset
filePath = "../data/raw/asd-new.csv"
df = pd.read_csv(filePath)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace("'", '').str.replace('-', '_')
df = df.drop(['CASE_NO_PATIENTS'], axis=1)

print(df.isnull().sum())
df.dtypes
df.head()

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
categorical_cols.remove('ASD_traits')

# Fill missing values
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

print(df.isnull().sum())

# Convert ASD_traits to numeric
df['ASD_traits'] = df['ASD_traits'].map({'Yes': 1, 'No': 0}) 

# One-Hot Encode categorical columns
onehot = OneHotEncoder(sparse_output=False, drop='first')
categorical_encoded_data = onehot.fit_transform(df[categorical_cols])  # This is the actual array
categorical_encoded_names = onehot.get_feature_names_out(categorical_cols)  # Column names

# Standardize numerical columns
scaler = StandardScaler()
numerical_standardized_data = scaler.fit_transform(df[numerical_cols])  # This is the actual array

# Combine them
combined_data = np.concatenate([categorical_encoded_data, numerical_standardized_data], axis=1)
combined_column_names = np.concatenate([categorical_encoded_names, numerical_cols])

# Convert to DataFrame
feature_cols_df = pd.DataFrame(combined_data, columns=combined_column_names)
feature_cols_df.head()

# Final dataset
df = pd.concat([df['ASD_traits'].reset_index(drop=True), feature_cols_df], axis=1)
df.head()

# Feature-target split
X = feature_cols_df
y = df['ASD_traits']

# Split data: train, validation, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Convert to tensors
def to_tensor_dataset(X, y):
    return TensorDataset(
        torch.tensor(X.to_numpy(), dtype=torch.float32),
        torch.tensor(y.to_numpy().astype(np.int64))  # Force conversion to int64
    )

# Create DataLoaders
train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(to_tensor_dataset(X_test, y_test), batch_size=64)

# Define upgraded MLP model
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

model = ASDClassifier(X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train with early stopping
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