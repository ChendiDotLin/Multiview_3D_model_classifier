import torch
import torch.nn as nn
from torchvision import models, transforms

class MultiView3DModelClassifier(nn.Module):
    def __init__(self):
        super(MultiView3DModelClassifier, self).__init__()
        # Use ResNet50 pretrained on ImageNet for feature extraction
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # Disable training for the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # RNN (LSTM) for processing the sequential features
        self.rnn = nn.LSTM(input_size=2048, hidden_size=512, num_layers=1, batch_first=True)
        
        # Classification heads
        self.style_head = nn.Linear(512, 5)  # Style (1 to 5)
        self.score_head = nn.Linear(512, 5)  # Score (1 to 5)
        self.multi_object_head = nn.Linear(512, 1)  # Is_multi_object (bool)
        self.transparent_head = nn.Linear(512, 1)  # Is_transparent (bool)
        self.figure_head = nn.Linear(512, 1)  # Is_figure (bool)

    def forward(self, x):
        # x is expected to be of shape (batch_size, num_views, C, H, W)
        batch_size, num_views, C, H, W = x.size()
        x = x.view(batch_size * num_views, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_views, -1)
        
        # RNN
        _, (hn, _) = self.rnn(features)
        
        # Predictions
        style = self.style_head(hn.squeeze(0))
        score = self.score_head(hn.squeeze(0))
        is_multi_object = torch.sigmoid(self.multi_object_head(hn.squeeze(0)))
        is_transparent = torch.sigmoid(self.transparent_head(hn.squeeze(0)))
        is_figure = torch.sigmoid(self.figure_head(hn.squeeze(0)))
        
        return style, score, is_multi_object, is_transparent, is_figure

# Instantiate the model
model = MultiView3DModelClassifier()

# Example tensor for 40 views (assuming batch size of 1 and 3x224x224 images)
# x = torch.randn(1, 40, 3, 224, 224)
# style, score, is_multi_object, is_transparent, is_figure = model(x)
