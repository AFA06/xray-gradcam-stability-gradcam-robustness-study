import torch.nn as nn
from torchvision.models import resnet18


class CheXpertResNet18(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.model = resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
