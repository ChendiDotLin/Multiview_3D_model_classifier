import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from dataset import LMDBDataset


def get_loaders():
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Instantiate your dataset
    transformed_dataset = LMDBDataset("transformed_data")
    # Assuming 'transformed_dataset' is an instance of your dataset
    total_size = len(transformed_dataset)
    train_size = int(total_size * 0.8)  # 80% for training
    test_size = total_size - train_size  # Remaining for test

    train_dataset, test_dataset = random_split(
        transformed_dataset, [train_size, test_size]
    )

    # Now, you can use DataLoader to create loaders for training and test sets
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    return train_loader, test_loader
