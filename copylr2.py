import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("hotel_bookings.csv", low_memory=False)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Selecting relevant features for predicting 'adr'
selected_features = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children"]
df_selected = df[selected_features + ["adr"]].copy()

# Convert 'children' to numeric and fill missing values with the median
df_selected["children"] = pd.to_numeric(df_selected["children"], errors="coerce")
df_selected["children"].fillna(df_selected["children"].median(), inplace=True)

# Handle missing values in 'adr' if any
df_selected.dropna(inplace=True)

# Handle outliers by capping extreme values (Winsorization method)
for col in selected_features + ["adr"]:
    q1 = df_selected[col].quantile(0.01)  # 1st percentile
    q99 = df_selected[col].quantile(0.99)  # 99th percentile
    df_selected[col] = df_selected[col].clip(lower=q1, upper=q99)

# Normalize features using Min-Max Scaling
scaler = MinMaxScaler()
df_selected[selected_features] = scaler.fit_transform(df_selected[selected_features])

# Splitting into training (80%) and testing (20%) sets
X = df_selected[selected_features]  # Independent variables
y = df_selected["adr"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset shape
print("Training Set Shape:", X_train.shape, y_train.shape)
print("Testing Set Shape:", X_test.shape, y_test.shape)
