import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PyTorch Tabular
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import NodeConfig
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Fix for PyTorch 2.6+ security issue - Alternative comprehensive approach
import torch
import omegaconf

# Method 1: Add comprehensive OmegaConf support
try:
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata
    from omegaconf._utils import _get_value
    from omegaconf.nodes import StringNode, IntegerNode, FloatNode, BooleanNode, EnumNode
    from omegaconf.basecontainer import BaseContainer
    from omegaconf._impl import ConfigImpl
    
    # Add all necessary OmegaConf classes to safe globals
    torch.serialization.add_safe_globals([
        DictConfig,
        ListConfig,
        ContainerMetadata,
        _get_value,
        StringNode,
        IntegerNode,
        FloatNode,
        BooleanNode,
        EnumNode,
        BaseContainer,
        ConfigImpl
    ])
except ImportError as e:
    print(f"Some OmegaConf imports failed: {e}")
    # Fallback: just add the basic ones
    from omegaconf import DictConfig
    torch.serialization.add_safe_globals([DictConfig])

# Method 2: Alternative - Monkey patch the pytorch_tabular load function (if Method 1 doesn't work)
# Uncomment the lines below if you still get errors:
# import pytorch_tabular.utils.python_utils as pt_utils
# original_pl_load = pt_utils.pl_load
# def patched_pl_load(path_or_url, map_location=None):
#     fs = pt_utils.get_filesystem(path_or_url)
#     with fs.open(path_or_url, "rb") as f:
#         return torch.load(f, map_location=map_location, weights_only=False)
# pt_utils.pl_load = patched_pl_load

# Load and clean dataset
filePath = "../data/raw/asd-new.csv"
df = pd.read_csv(filePath)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace("'", '').str.replace('-', '_')
df = df.drop(['CASE_NO_PATIENTS'], axis=1)

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
categorical_cols.remove('ASD_traits')

# Fill missing values
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Convert target to numeric
df['ASD_traits'] = df['ASD_traits'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = encoder.fit_transform(df[categorical_cols])
categorical_encoded_names = encoder.get_feature_names_out(categorical_cols)

# Standardize numeric features
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df[numerical_cols])

# Combine features
X_data = np.concatenate([categorical_encoded, numerical_scaled], axis=1)
X_df = pd.DataFrame(X_data, columns=np.concatenate([categorical_encoded_names, numerical_cols]))
y_df = df['ASD_traits'].reset_index(drop=True)
final_df = pd.concat([y_df, X_df], axis=1)

# Split data: train, val, test
X_train, X_temp, y_train, y_temp = train_test_split(X_df, y_df, test_size=0.3, stratify=y_df, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Convert back to DataFrames
train_df = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
val_df = pd.concat([y_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1)
test_df = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

# Configuration for PyTorch Tabular
data_config = DataConfig(
    target=["ASD_traits"],
    continuous_cols=list(X_df.columns)
)

# Model config
model_config = NodeConfig(
    task="classification",
    num_layers=2,
    num_trees=512,
    depth=6,
    learning_rate=1e-3,
    choice_function="entmax15",
    bin_function="entmoid15",
    metrics=["accuracy"],
)

# Trainer config with correct early stopping monitor
trainer_config = TrainerConfig(
    auto_lr_find=False,
    batch_size=64,
    max_epochs=50,
    checkpoints="valid_accuracy",
    early_stopping="valid_accuracy",
    early_stopping_mode="max",
)

# Define the correct callback
early_stop_callback = EarlyStopping(
    monitor="valid_accuracy",
    mode="max",
    patience=5,
    verbose=True
)

# Optimizer config
optimizer_config = OptimizerConfig()

# Initialize model
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    trainer_config=trainer_config,
    optimizer_config=optimizer_config
)

# Train the model
tabular_model.fit(
    train=train_df,
    validation=val_df,
)

# Predictions
pred_df = tabular_model.predict(test_df)
y_pred = pred_df['prediction'].values
y_true = test_df['ASD_traits'].values

# Evaluation
print(f"\nTest Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred, zero_division=0))

# Plotting loss curves (optional if logs available)
try:
    metrics_df = tabular_model.trainer.logged_metrics
    if hasattr(metrics_df, 'keys') and "loss" in metrics_df:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label='Train Loss')
        plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs. Validation Loss")
        plt.legend()
        plt.show()
    else:
        print("Loss metrics not available for plotting")
except Exception as e:
    print(f"Could not plot metrics: {e}")