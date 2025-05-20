import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
csv1 = pd.read_csv("Housing.csv")

# Ensure the 'bedrooms' column is numeric
#csv1["bedrooms"] = pd.to_numeric(csv1["bedrooms"], errors="coerce")

# Reshape for sklearn (expects 2D array)
bedrooms_array = csv1["bedrooms"].values.reshape(-1, 1) # to convert it in bedroom array and also we will reshape this using reshape(-1,-1)

# Apply Min-Max Normalization
scaler = MinMaxScaler()
csv1["bedrooms"] = scaler.fit_transform(bedrooms_array)

# Print normalized column
print(csv1["bedrooms"])
