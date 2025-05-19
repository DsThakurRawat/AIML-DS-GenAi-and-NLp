import pandas as pd
df = pd.read_csv("Reviews.csv")

print(df)


#We’ll measure review length using the Text column (the full review body). Use len() to count characters or split() to count words.
df["ReviewLenght"] = df["Text"].apply(len)

df["WordCount"] = df["Text"].apply(lambda x : len(str(x).split()))

# Step 3: Explore Average Length by Rating
# we will group by score and will  get average count
avglength = df.groupby("Score")["WordCount"].mean().reset_index()

print(avglength)

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data = avglength, x='Score', y='WordCount', palette='coolwarm')
plt.title("Average Review Word Count by Rating")
plt.xlabel("Rating Score")
plt.ylabel("Average Word Count")
plt.show()