import pandas as pd

# Load the CSV file
file_path = 'new_10000_training_labels.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)
# Replace 'None' strings with actual NaN values
df.replace('None', pd.NA, inplace=True)
# Remove rows where all columns except 'uid' contain 'None'
cleaned_df = df.dropna(how='all', subset=df.columns[1:])

# Save the cleaned data to a new CSV file
output_file_path = 'cleaned_new_10000_training_labels.csv'  # Replace with your desired output file path
cleaned_df.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")