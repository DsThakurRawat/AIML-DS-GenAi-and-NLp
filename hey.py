import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
csv1 = pd.read_csv("Housing.csv")
csv1.bedrooms = pd.to_numeric(csv1.bedrooms, errors="coerce")

# Handling NaN values
csv1.bedrooms.fillna(csv1.bedrooms.median(), inplace=True)

# Reshape and normalize
bedrooms_array = csv1.bedrooms.values.reshape(-1, 1)
scaler = MinMaxScaler()
csv1.bedrooms = scaler.fit_transform(bedrooms_array)

# Define a function to classify sizes
def classify_size(value):
    if value < 0.33:
        return "Small"
    elif 0.33 <= value < 0.67:
        return "Medium"
    else:
        return "Large"

# Apply classification function to each row
csv1["Size_Category"] = csv1.bedrooms.apply(classify_size)

# Print results
print(csv1[["bedrooms", "Size_Category"]])








