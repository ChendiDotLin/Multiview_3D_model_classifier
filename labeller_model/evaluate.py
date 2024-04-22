from model import MultiView3DModelClassifier
from data_loader import get_loaders
import torch
from tqdm import tqdm  # Import the tqdm function
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    output_labels = [
        "style",
        "score",
        "is_multi_object",
        "is_weird",
        "is_scene",
        "is_figure",
        "is_transparent",
        "density",
    ]
    categorical_labels = ["style", "score", "density"]
    binary_labels = [
        "is_multi_object",
        "is_weird",
        "is_scene",
        "is_figure",
        "is_transparent",
    ]

    # Initialize dictionaries to store predictions and actual labels
    predictions = {label: [] for label in output_labels}
    true_values = {label: [] for label in output_labels}

    # Assuming the get_loaders function can also be used here to get test_loader
    _, test_loader = get_loaders()

    model = MultiView3DModelClassifier()
    model.load_state_dict(torch.load("saved_models/model_epoch_33.pth"))
    model.eval()

    # No gradient calculation is needed
    progress_bar = tqdm(test_loader, desc=f"test progress", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            outputs = model(images)

            # Assuming labels is a tuple in the same order as output_labels
            for i, label_name in enumerate(output_labels):
                output = outputs[i]
                label = labels[:, i]
                # print(output)
                # print(label)
                if label_name in categorical_labels:
                    _, predicted = torch.max(
                        output.data, 1
                    )  # Get the index of the max log-probability
                    predictions[label_name].extend(predicted.cpu().numpy())
                    true_values[label_name].extend(label.cpu().numpy())
                elif label_name in binary_labels:
                    predicted = (
                        torch.sigmoid(output.data) > 0.5
                    ).int()  # Binary classification
                    predictions[label_name].extend(predicted.view(-1).cpu().numpy())
                    true_values[label_name].extend(label.view(-1).cpu().numpy())
            # Optionally update the progress bar description to show the running loss
            progress_bar.set_description(f"")
    # Calculate and print accuracy for each type of label
    for label_name in output_labels:
        accuracy = accuracy_score(true_values[label_name], predictions[label_name])
        print(f"Accuracy for {label_name}: {accuracy}")
        # print(f"Accuracy of the model on the test images: {100 * correct / total}%")
