import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from dataset import LMDBDataset
from torchvision import transforms
from find_missing_data import return_filtered_labels


def get_loaders():
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Define your transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    csv_file_path = "training_label.csv"
    labels = return_filtered_labels(csv_file_path)
    # import itertools

    # out = dict(itertools.islice(labels.items(), 3))
    # uids = list(out.keys())
    uids = list(labels.keys())
    # Instantiate your dataset
    transformed_dataset = LMDBDataset("transformed_data", uids, transform)
    # Assuming 'transformed_dataset' is an instance of your dataset
    total_size = len(transformed_dataset)
    print("total number of training data: ", total_size)
    train_size = int(total_size * 0.8)  # 80% for training
    test_size = total_size - train_size  # Remaining for test
    print("training data: ", train_size)
    print("test data: ", test_size)

    train_dataset, test_dataset = random_split(
        transformed_dataset, [train_size, test_size]
    )

    # Now, you can use DataLoader to create loaders for training and test sets
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader
