from model import MultiView3DModelClassifier
from data_loader import get_loaders
import torch

# Assuming the get_loaders function can also be used here to get test_loader
_, test_loader = get_loaders()

model = MultiView3DModelClassifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()

correct = 0
total = 0
# No gradient calculation is needed
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # Assuming the output is the predicted class for simplicity; adjust as needed
        _, predicted = torch.max(outputs[0].data, 1)  # Example for getting predictions from one of the outputs
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total}%')