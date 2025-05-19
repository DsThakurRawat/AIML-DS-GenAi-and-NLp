import pandas as pd 
import numpy as np
import matplotlib as pt
import matplotlib.pyplot as plt 
import seaborn as sns
#step -1 laoding and inspecting data
df = pd.read_csv("Reviews.csv")
#print(df.head(15))
# examine the score columns by printing it
print(df['Score'].value_counts()) # this will show how many customer gave 5 star 4 star 3 star 2 star

# step 2 categorize sentiment
"""
1 - 2 negative 
3 - neutral
4-5 - positive;
"""

def mapsentiment(Score):
    if Score <=2:
        return "Negative"
    elif Score == 3:
        return "Neutral"
    else: 
        return "Positive"


df["sentiment"] = df["Score"].apply(mapsentiment)


    
#visualising the sentiment analysis sentiment analysis means what are the sentiments of user accodrding to rating given by them
#step 3 visualising the sentiment distribution 
# we will using matplotlib and seaborn to create a bar plot or pie plot 

sns.countplot(x = "sentiment", data = df, palette = "pastel" )
plt.title("sentiment distribution of amazon food Reviews")
plt.xlabel("sentiment")
plt.ylabel("Number of Reviews Recieved")
plt.show()