import torch
import torch.nn as nn
from torchvision.models import densenet169

# Define the custom DenseNet-169 model with added dropout and custom classifier
class DenseNet169Custom(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet169Custom, self).__init__()
        # Load pre-trained DenseNet-169
        self.densenet = densenet169(pretrained=True)
        num_features = self.densenet.classifier.in_features
        
        # Replace the default classifier with a custom one
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)
