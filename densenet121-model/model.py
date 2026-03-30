import torch.nn as nn
import torchvision.models as models


class PneumoniaDenseNet(nn.Module):
    def __init__(self):
        super(PneumoniaDenseNet, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)  # Binary classification

    def forward(self, x):
        return self.model(x)
