import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Reviews.csv")
# Assume the dataset has a 'Time' column representing the timestamp of the review
df['Time'] = pd.to_datetime(df['Time'], unit='s')  # If the time is in Unix timestamp

# Extract the year
df['Year'] = df['Time'].dt.year

# Group by year: count of reviews
reviews_per_year = df.groupby('Year').size()

# Group by year: average rating
avg_rating_per_year = df.groupby('Year')['Score'].mean()

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Reviews', color=color)
ax1.plot(reviews_per_year.index, reviews_per_year.values, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Average Rating', color=color)
ax2.plot(avg_rating_per_year.index, avg_rating_per_year.values, color=color, marker='x')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Review Volume and Average Rating Over Years')
plt.show()
