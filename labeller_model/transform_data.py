import torch
import lmdb
import os
import glob
import pickle
from PIL import Image
import io
from tqdm import tqdm
import shutil
from find_missing_data import return_filtered_labels
import csv


def load_metadata_from_csv(csv_file):
    metadata = {}
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line, which is the header
        for row in reader:
            uid = row[0]
            metadata[uid] = tuple(row[1:])  # Adjust the slicing if necessary
            # print(metadata[uid])
    return metadata


def create_lmdb_dataset(source_folder, lmdb_path, all_labels, all_metadata):
    # If the database directory already exists, remove it
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)  # Use shutil.rmtree to delete an entire directory tree

    map_size = int(1e12)  # Size of the database, this is a maximum limit
    db = lmdb.open(lmdb_path, map_size=map_size)

    with db.begin(write=True) as txn:
        # Initialize progress bar with total steps equal to the number of items in labels
        pbar = tqdm(total=len(all_labels), desc="Creating LMDB Dataset", unit="item")
        for idx, (uid, labels) in enumerate(all_labels.items()):
            img_folder = os.path.join(source_folder, uid)
            # Check if the image folder exists
            if not os.path.exists(img_folder):
                # Handle the case where the image folder doesn't exist
                print(f"Image folder does not exist: {img_folder}")
                continue
            images = sorted(glob.glob(f"{img_folder}/*.png"))
            if len(images) != 40:
                print(f"No enough images: {img_folder}")
                continue
            img_list = []
            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_list.append(
                    img_byte_arr.getvalue()
                )  # Save image data as byte array
            key = uid.encode("ascii")  # Use uid as key
            labels_values = [int(label) for label in labels]
            if len(labels_values) != 8:
                print("bad data uid: ", uid)
                continue
            labels_tensor = torch.tensor(labels_values)
            metadata = all_metadata[uid]
            metadata_values = [int(m) for m in metadata]
            # print(metadata_values)
            metadata_tensor = torch.tensor(metadata_values)
            value = pickle.dumps((img_list, labels_tensor, metadata_tensor))
            txn.put(key, value)
            # Update the progress bar
            pbar.update(1)
        pbar.close()
    db.close()


# Example usage:
csv_file_path = "training_label.csv"
labels = return_filtered_labels(csv_file_path)
print("finish loading labels")
all_metadata = load_metadata_from_csv("model_metadata.csv")
# Instantiate your dataset
create_lmdb_dataset(
    source_folder="views",
    lmdb_path="./transformed_data",
    all_labels=labels,
    all_metadata=all_metadata,
)
print("finish creating lmdb")
# Assuming labels are stored as a tuple:
# (style, score, is_multi_object, is_weird, is_scene, is_figure, is_transparent, density)
