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