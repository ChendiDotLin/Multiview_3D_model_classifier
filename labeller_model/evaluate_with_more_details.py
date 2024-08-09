from model import MultiView3DModelClassifier, MultiView3DModelClassifierWithAttention
from data_loader import get_loaders
import torch
from tqdm import tqdm  # Import the tqdm function
from sklearn.metrics import accuracy_score
import csv
# import boto3

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

    mismatch = {label: [] for label in output_labels}
    mismatch["relaxed_score"] = []
    # Assuming the get_loaders function can also be used here to get test_loader
    _, test_loader = get_loaders()

    model = MultiView3DModelClassifierWithAttention(num_layers=1)
    # modify the following
    model.load_state_dict(torch.load("saved_models/1_layer_rnn_16_linear_model_epoch_44.pth", map_location=torch.device('cpu')))
    model.eval()
    csv_writers = {}
    fieldnames = ['UID', 'Predicted', 'True']
    # No gradient calculation is needed
    progress_bar = tqdm(test_loader, desc=f"test progress", leave=False)
    with torch.no_grad():
        for images, labels, metadata, uid in progress_bar:
            # images, labels, metadata = images.cuda(), labels.cuda(), metadata.cuda()
            outputs = model(images, metadata)

            # Assuming labels is a tuple in the same order as output_labels
            for i, label_name in enumerate(output_labels):
                output = outputs[i]
                label = labels[:, i]
                # print(output)
                # print(label)

                if label_name not in csv_writers:
                    if label_name == "score":
                        category_file = "10000_relaxed_score.csv"
                        file = open(category_file, mode='w', newline='')
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        writer.writeheader()
                        csv_writers["relaxed_score"] = (writer, file)
                    category_file = f'10000_{label_name}.csv'
                    file = open(category_file, mode='w', newline='')
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    csv_writers[label_name] = (writer, file)
                if label_name in categorical_labels:
                    _, predicted = torch.max(
                        output.data, 1
                    )  # Get the index of the max log-probability
                    predictions[label_name].extend(predicted.cpu().numpy())
                    true_values[label_name].extend(label.cpu().numpy())
                    if predicted.cpu().numpy() != label.cpu().numpy():
                        mismatch[label_name].extend(uid)
                        writer, _ = csv_writers[label_name]
                        writer.writerow({'UID':uid, 'Predicted':predicted.cpu().numpy(), 'True': label.cpu().numpy()})
                    if label_name == "score":
                        # we also edit the relaxed score
                        predict = predicted.cpu().numpy()
                        true = label.cpu().numpy()
                        relaxed_accurate = (predict == true) or (predict == 2 and true == 3) or (predict == 3 and true == 2)
                        if not relaxed_accurate:
                            mismatch["relaxed_score"].extend(uid)
                            writer, _ = csv_writers["relaxed_score"]
                            writer.writerow({'UID':uid, 'Predicted':predict, 'True': true})

                elif label_name in binary_labels:
                    predicted = (
                        torch.sigmoid(output.data) > 0.5
                    ).int()  # Binary classification
                    predictions[label_name].extend(predicted.view(-1).cpu().numpy())
                    true_values[label_name].extend(label.view(-1).cpu().numpy())
                    if predicted.cpu().numpy() != label.cpu().numpy():
                        mismatch[label_name].extend(uid)
                        writer, _ = csv_writers[label_name]
                        writer.writerow({'UID':uid, 'Predicted':predicted.cpu().numpy(), 'True': label.cpu().numpy()})
            # Optionally update the progress bar description to show the running loss
            progress_bar.set_description(f"")
    # Calculate and print accuracy for each type of label
    for label_name in output_labels:
        accuracy = accuracy_score(true_values[label_name], predictions[label_name])
        print(f"Accuracy for {label_name}: {accuracy}")
       # print(mismatch[label_name])
        # print(f"Accuracy of the model on the test images: {100 * correct / total}%")
    # now special handle score
    # if the score is 2 or 3, we still call it correct
    relaxed_correct_score = 0
    for (true_score, predict_score) in zip(true_values["score"], predictions["score"]):
        if true_score == predict_score:
            relaxed_correct_score += 1
        elif true_score == 2 and predict_score == 3:
            relaxed_correct_score += 1
        elif true_score == 3 and predict_score == 2:
            relaxed_correct_score += 1
    relaxed_accuracy = relaxed_correct_score * 1.0 / len(true_values["score"])
    print(f"Relaxed Accuracy for score: {relaxed_accuracy}")



    # Specify your bucket name and file path
    for writer, file in csv_writers.values():
        file.close()

    # bucket_name = "10000-training-data-west"
    # # Initialize a session using your credentials
    # s3 = boto3.client("s3")
    # for name in csv_writers.keys():
    #      s3.upload_file(f"{name}.csv", bucket_name, f"{name}.csv")