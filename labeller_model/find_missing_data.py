import csv
import os
import glob


def load_labels_from_csv(csv_file):
    labels = {}
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line, which is the header
        for row in reader:
            uid = row[0]
            labels[uid] = tuple(row[1:])  # Adjust the slicing if necessary
    return labels


def return_filtered_labels(csv_file_path):
    all_labels = load_labels_from_csv(csv_file_path)
    source_folder = "views"
    exclude_labels = set()
    for idx, (uid, labels) in enumerate(all_labels.items()):
        img_folder = os.path.join(source_folder, uid)
        # Check if the image folder exists
        if not os.path.exists(img_folder):
            # Handle the case where the image folder doesn't exist
            # print(uid)
            exclude_labels.add(uid)
            continue
        images = sorted(glob.glob(f"{img_folder}/*.png"))
        if len(images) != 40:
            # print(uid)
            exclude_labels.add(uid)
            continue
        # print(len(labels))
        if len(labels) != 8:
            # print(uid)
            exclude_labels.add(uid)
            continue

    true_labels = {}
    with open(csv_file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line, which is the header
        for row in reader:
            uid = row[0]
            if uid not in exclude_labels:
                true_labels[uid] = tuple(row[1:])  # Adjust the slicing if necessary
    # print("number of data: ", len(true_labels))
    return true_labels


true_labels = return_filtered_labels("training_label.csv")
# print(len(true_labels))
