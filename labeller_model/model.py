import torch
import torch.nn as nn
from torchvision import models, transforms


class MultiView3DModelClassifier(nn.Module):
    def __init__(self, num_layers=1):
        super(MultiView3DModelClassifier, self).__init__()
        # Use ResNet50 pretrained on ImageNet for feature extraction
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # Disable training for the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # RNN (LSTM) for processing the sequential features
        self.rnn = nn.LSTM(
            input_size=2048, hidden_size=512, num_layers=num_layers, batch_first=True
        )
        # four metadata: vertexCount,faceCount,viewCount,likeCount
        self.meta_fc = nn.Linear(4, 32)  # Process metadata

        # Classification heads
        # Assuming labels are stored as a tuple:
        # (style, score, is_multi_object, is_weird, is_scene, is_figure, is_transparent, density)
        self.style_head = nn.Linear(512 + 32, 8)  # Style (0 to 7)
        self.score_head = nn.Linear(512 + 32, 4)  # Score (0 to 3)
        self.multi_object_head = nn.Linear(512 + 32, 1)  # Is_multi_object (bool)
        self.weird_head = nn.Linear(512 + 32, 1)  # Is_weird (bool)
        self.scene_head = nn.Linear(512 + 32, 1)  # Is_scene (bool)
        self.figure_head = nn.Linear(512 + 32, 1)  # Is_figure (bool)
        self.transparent_head = nn.Linear(512 + 32, 1)  # Is_transparent (bool)
        self.density_head = nn.Linear(512 + 32, 3)  # Density (0 to 2)

    def forward(self, x, metadata):
        # x is expected to be of shape (batch_size, num_views, C, H, W)
        batch_size, num_views, C, H, W = x.size()
        x = x.view(batch_size * num_views, C, H, W).float()
        metadata = metadata.float()
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_views, -1)
        meta_out = nn.functional.relu(self.meta_fc(metadata))  # Process metadata

        # RNN
        rnn_out, (hn, _) = self.rnn(combined)
        # Combine features with metadata
        combined = torch.cat((rnn_out[:, -1, :], meta_out), dim=1)

        # Predictions
        style = self.style_head()
        score = self.score_head(rnn_out[:, -1, :])
        is_multi_object = self.multi_object_head(rnn_out[:, -1, :])
        is_weird = self.weird_head(rnn_out[:, -1, :])
        is_scene = self.scene_head(rnn_out[:, -1, :])
        is_figure = self.figure_head(rnn_out[:, -1, :])
        is_transparent = self.transparent_head(rnn_out[:, -1, :])
        density = self.density_head(rnn_out[:, -1, :])

        return (
            style,
            score,
            is_multi_object,
            is_weird,
            is_scene,
            is_figure,
            is_transparent,
            density,
        )


# Instantiate the model
model = MultiView3DModelClassifier()

# Example tensor for 40 views (assuming batch size of 1 and 3x224x224 images)
# x = torch.randn(1, 40, 3, 224, 224)
# style, score, is_multi_object, is_transparent, is_figure = model(x)
