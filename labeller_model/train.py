import torch.nn as nn
import torch
from model import MultiView3DModelClassifier
from data_loader import get_loaders
from tqdm import tqdm  # Import the tqdm function
import os
import multiprocessing as mp

model_save_path = "./saved_models"  # Define the directory to save the models
os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists

criterion_style_score = nn.CrossEntropyLoss()
criterion_binary = (
    nn.BCEWithLogitsLoss()
)  # This expects logits as input; sigmoid is included inside
num_epochs = 50


def combined_loss(outputs, labels):
    # print(outputs)
    # print(labels)
    (
        style,
        score,
        is_multi_object,
        is_weird,
        is_scene,
        is_figure,
        is_transparent,
        density,
    ) = outputs
    # Assuming labels are stored as a tuple:
    # (style, score, is_multi_object, is_weird, is_scene, is_figure, is_transparent, density)
    # Cast categorical labels to long and binary labels to float
    style_labels = labels[:, 0].long()  # Assuming first label is style
    score_labels = labels[:, 1].long()  # Assuming second label is score
    density_labels = labels[:, 7].long()  # Adjust index as necessary

    # Binary labels
    is_multi_object_labels = labels[:, 2].float()  # Adjust index as necessary
    is_weird_labels = labels[:, 3].float()
    is_scene_labels = labels[:, 4].float()
    is_figure_labels = labels[:, 5].float()
    is_transparent_labels = labels[:, 6].float()
    # print("style: ", style)
    # print("style labels: ", style_labels)
    loss_style = criterion_style_score(style, style_labels)
    loss_score = criterion_style_score(score, score_labels)
    loss_density = criterion_style_score(density, density_labels)
    loss_multi_object = criterion_binary(
        is_multi_object, is_multi_object_labels.view(-1, 1).float()
    )
    loss_weird = criterion_binary(is_weird, is_weird_labels.view(-1, 1).float())
    loss_scene = criterion_binary(is_scene, is_scene_labels.view(-1, 1).float())
    loss_figure = criterion_binary(is_figure, is_figure_labels.view(-1, 1).float())
    loss_transparent = criterion_binary(
        is_transparent, is_transparent_labels.view(-1, 1).float()
    )

    total_loss = (
        loss_style
        + loss_score
        + loss_density
        + loss_multi_object
        + loss_weird
        + loss_scene
        + loss_figure
        + loss_transparent
    )
    return total_loss


def train(train_loader):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )

        for images, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Optionally update the progress bar description to show the running loss
            progress_bar.set_description(
                f"Epoch {epoch+1} - Loss: {running_loss / (progress_bar.n + 1)}"
            )

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        # Save the model
        model_filename = f"model_epoch_{epoch+1}.pth"  # Naming the model file
        save_path = os.path.join(model_save_path, model_filename)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
    # Saving the model's state dictionary
    # torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    mp.freeze_support()  # Only necessary if the program might be frozen, e.g., with PyInstaller
    model = MultiView3DModelClassifier()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = get_loaders()
    train(train_loader)
