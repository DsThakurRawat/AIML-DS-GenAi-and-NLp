# reading csv file and dispalying the first 5 rows
import pandas as pd # type: ignore
import matplotlib.pyplot  as mt
import seaborn as sb 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

csv1 = pd.read_csv("Housing.csv")
df = pd.DataFrame(csv1,index = [1,2,3,4,5]) # this index = [1,2,3,4,5] is used to display the first 5 rows
print(df)
#print(csv1.head())
#print(csv1.to_string()) # to print all data frame
#print(csv1.isnull().sum()) #clear # but here 
# this return false when data is not missing and return true when data is missing


#draw a line in diagram from pos 0,0 to 6,60
bdrooms = csv1.bedrooms
y=bdrooms
y.label=('Bedrooms')
mt.hist(bdrooms, bins = 20, color = "green") #
#mt.show()
"""
Xnorm = (X - X.min()) / (X.max() - X.min()) ( this is formula of normalization)

"""""
""""
Bdmin = csv1.bedrooms.min()
Bdmax = csv1.bedrooms.max()"""

"""def normalization(roomnumber):
     
      
      Xnorm = (roomnumber - Bdmin) / (Bdmax - Bdmin)
      return Xnorm
roomnumber = csv1.bedrooms
csv1.bedrooms = csv1.bedrooms.apply(normalization)
print(csv1.bedrooms)"""
 # using sklearn to print normalize daata


 # question--2 normalizing the whole data 
# Load data
csv1 = pd.read_csv("Housing.csv")

# Ensure the 'bedrooms' column is numeric
csv1["bedrooms"] = pd.to_numeric(csv1["bedrooms"], errors="coerce")

# Reshape for sklearn (expects 2D array)
bedrooms_array = csv1["bedrooms"].values.reshape(-1, 1)

# Apply Min-Max Normalization and preprocessing
scaler = MinMaxScaler()
csv1["bedrooms"] = scaler.fit_transform(bedrooms_array)

# Print normalized column
print(csv1["bedrooms"])


