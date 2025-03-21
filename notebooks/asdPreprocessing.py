import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

filePath = "../data/raw/asd-new.csv"
df = pd.read_csv(filePath)

df.info()

df.head()
