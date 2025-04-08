import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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