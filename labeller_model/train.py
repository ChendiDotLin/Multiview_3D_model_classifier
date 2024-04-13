import torch.nn as nn
import torch
from model import MultiView3DModelClassifier
from data_loader import get_loaders

criterion_style_score = nn.CrossEntropyLoss()
criterion_binary = (
    nn.BCEWithLogitsLoss()
)  # This expects logits as input; sigmoid is included inside
num_epochs = 20


def combined_loss(outputs, labels):
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
    (
        style_labels,
        score_labels,
        is_multi_object_labels,
        is_weird_labels,
        is_scene_labels,
        is_figure_labels,
        is_transparent_labels,
        density_labels,
    ) = labels

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


model = MultiView3DModelClassifier()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader = get_loaders()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Saving the model's state dictionary
torch.save(model.state_dict(), "model.pth")
