import pandas as pd

# List of CSV files
csv_files = ['10000_density.csv', '10000_is_figure.csv', '10000_is_multi_object.csv', '10000_is_scene.csv', '10000_is_transparent.csv', '10000_is_weird.csv', '10000_relaxed_score.csv', '10000_style.csv']

# Set to store unique UIDs
unique_uids = set()

# Loop through each CSV file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Add the UIDs to the set
    unique_uids.update(df['UID'])

# Write the unique UIDs to a text file
with open('relabel_uid.txt', 'w') as f:
    for uid in unique_uids:
        f.write(f"{uid}\n")

print("Unique UIDs have been written to unique_uids.txt")