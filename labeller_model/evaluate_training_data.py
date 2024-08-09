import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "10000_training_labels.csv"
df = pd.read_csv(file_path)

# List of tags to generate pie charts for
tags = df.columns.tolist()
tags.remove("uid")


# Function to plot pie charts for each tag
def plot_pie_charts(df, tags):
    for tag in tags:
        data = df[tag].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140)
        plt.title(f"Distribution of {tag}")
        plt.axis("equal")
        plt.show()


# Plot pie charts
plot_pie_charts(df, tags)
