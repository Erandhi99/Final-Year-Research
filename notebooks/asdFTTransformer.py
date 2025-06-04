import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import rtdl
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and clean dataset
filePath = "../data/raw/asd-new.csv"
df = pd.read_csv(filePath)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace("'", '').str.replace('-', '_')
df = df.drop(['CASE_NO_PATIENTS'], axis=1)

# Target encoding
df['ASD_traits'] = df['ASD_traits'].map({'Yes': 1, 'No': 0})

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(exclude=['object']).drop('ASD_traits', axis=1).columns.tolist()

# Fill missing values
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Label encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Get cardinalities for categorical features
cat_cardinalities = [df[col].nunique() for col in categorical_cols]

# Standardize numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Feature-target split
X = df.drop(columns=['ASD_traits'])
y = df['ASD_traits']

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Prepare data for FT-Transformer
def prepare_ft_transformer_data(X, categorical_cols, numerical_cols):
    """Prepare data in the format expected by FT-Transformer"""
    X_cat = X[categorical_cols].values.astype(np.int64) if categorical_cols else None
    X_num = X[numerical_cols].values.astype(np.float32) if numerical_cols else None
    return X_cat, X_num

X_cat_train, X_num_train = prepare_ft_transformer_data(X_train, categorical_cols, numerical_cols)
X_cat_val, X_num_val = prepare_ft_transformer_data(X_val, categorical_cols, numerical_cols)
X_cat_test, X_num_test = prepare_ft_transformer_data(X_test, categorical_cols, numerical_cols)

y_train = y_train.values.astype(np.int64)
y_val = y_val.values.astype(np.int64)
y_test = y_test.values.astype(np.int64)

# Create data loaders
def create_dataloader(X_cat, X_num, y, batch_size=64, shuffle=True):
    """Create DataLoader for FT-Transformer"""
    tensors = []
    if X_cat is not None:
        tensors.append(torch.from_numpy(X_cat))
    if X_num is not None:
        tensors.append(torch.from_numpy(X_num))
    tensors.append(torch.from_numpy(y))
    
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_dataloader(X_cat_train, X_num_train, y_train, batch_size=64, shuffle=True)
val_loader = create_dataloader(X_cat_val, X_num_val, y_val, batch_size=64, shuffle=False)
test_loader = create_dataloader(X_cat_test, X_num_test, y_test, batch_size=64, shuffle=False)

# Create FT-Transformer model
model = rtdl.FTTransformer.make_default(
    n_num_features=len(numerical_cols) if numerical_cols else 0,
    cat_cardinalities=cat_cardinalities if categorical_cols else None,
    n_classes=2,  # Binary classification
    # Model architecture parameters
    d_token=96,  # Token dimension
    n_blocks=3,  # Number of transformer blocks
    attention_dropout=0.2,
    ffn_dropout=0.1,
    residual_dropout=0.0,
    activation='reglu',
    prenormalization=True,
    initialization='kaiming',
    # Token bias (helps with numerical features)
    token_bias=True,
    kv_compression=None,
    kv_compression_sharing=None,
)

model = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

# Training function
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        if len(batch) == 3:  # Both categorical and numerical features
            X_cat, X_num, y = [x.to(device) for x in batch]
            outputs = model(X_num, X_cat)
        elif X_cat_train is not None:  # Only categorical features
            X_cat, y = [x.to(device) for x in batch]
            outputs = model(None, X_cat)
        else:  # Only numerical features
            X_num, y = [x.to(device) for x in batch]
            outputs = model(X_num, None)
        
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:  # Both categorical and numerical features
                X_cat, X_num, y = [x.to(device) for x in batch]
                outputs = model(X_num, X_cat)
            elif X_cat_val is not None:  # Only categorical features
                X_cat, y = [x.to(device) for x in batch]
                outputs = model(None, X_cat)
            else:  # Only numerical features
                X_num, y = [x.to(device) for x in batch]
                outputs = model(X_num, None)
            
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

# Training loop
n_epochs = 100
best_val_acc = 0
patience = 15
patience_counter = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("Starting training...")
for epoch in range(n_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    scheduler.step()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_ft_transformer.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    print()

# Load best model
model.load_state_dict(torch.load('best_ft_transformer.pth'))

# Test evaluation
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # Both categorical and numerical features
                X_cat, X_num, y = [x.to(device) for x in batch]
                outputs = model(X_num, X_cat)
            elif X_cat_test is not None:  # Only categorical features
                X_cat, y = [x.to(device) for x in batch]
                outputs = model(None, X_cat)
            else:  # Only numerical features
                X_num, y = [x.to(device) for x in batch]
                outputs = model(X_num, None)
            
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    return np.array(predictions), np.array(actuals)

# Get predictions
preds, y_test_actual = predict(model, test_loader, device)

# Evaluation
print("Test Accuracy:", accuracy_score(y_test_actual, preds))
print("Classification Report:\n", classification_report(y_test_actual, preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_test_actual, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No ASD', 'ASD'], yticklabels=['No ASD', 'ASD'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('FT-Transformer Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Accuracy')
ax2.plot(val_accs, label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")