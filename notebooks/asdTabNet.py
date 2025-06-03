import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

# Extract cat_idxs and cat_dims for TabNet
cat_idxs = [df.columns.get_loc(col) for col in categorical_cols]
cat_dims = [df[col].nunique() for col in categorical_cols]

# Standardize numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Feature-target split
X = df.drop(columns=['ASD_traits'])
y = df['ASD_traits']

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Convert to float32 (TabNet requirement)
X_train = X_train.to_numpy().astype(np.float32)
X_val = X_val.to_numpy().astype(np.float32)
X_test = X_test.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(int)
y_val = y_val.to_numpy().astype(int)
y_test = y_test.to_numpy().astype(int)

# Create TabNet model
tabnet = TabNetClassifier(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=3,  # You can tune this value
    n_d=16, n_a=16, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    device_name='auto'
)

#Train the model
tabnet.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_val, y_val)],
    eval_name=['valid'],
    eval_metric=['accuracy'],
    max_epochs=200, patience=20,
    batch_size=64, virtual_batch_size=32,
    num_workers=0, drop_last=False
)

# Predictions and evaluation
preds = tabnet.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, preds))
print("Classification Report:\n",classification_report(y_test, preds))
conf_mat = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No ASD', 'ASD'],
            yticklabels=['No ASD', 'ASD'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()