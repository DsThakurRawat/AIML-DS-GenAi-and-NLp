import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot  as mtp
from sklearn.preprocessing import MinMaxScaler

#to lad data 
csv1 = pd.read_csv("Housing.csv")
csv1.bedrooms = pd.to_numeric(csv1.bedrooms, errors="coerce")
# to hanlde NaN values 
csv1.bedrooms.fillna(csv1.bedrooms.median(), inplace = True)
# don,t forget to reshape 
bedrooms_array = csv1.bedrooms.values.reshape(-1, 1)
scaler = MinMaxScaler()
csv1.bedrooms = scaler.fit_transform(bedrooms_array)

def classify_size(value):
    if value < 0.33:
        return "small"
    elif 0.33 <= value < 0.67:
        return "Median"
    else:
        return "Large"
    
csv1.size_category = csv1.bedrooms.apply(classify_size)
print(csv1.bedrooms)


