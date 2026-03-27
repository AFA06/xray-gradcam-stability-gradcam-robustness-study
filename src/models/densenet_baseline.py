import torch
import torch.nn as nn
import torchvision.models as models


class DenseNetBaseline(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super(DenseNetBaseline, self).__init__()

        # Load model
        if pretrained:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.model = models.densenet121(weights=None)

        # 🔥 CRITICAL FIX: disable ALL inplace ReLU (GradCAM crash fix)
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Replace classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ✅ This now matches your script call
def get_densenet121(num_classes=5, pretrained=False):
    return DenseNetBaseline(num_classes=num_classes, pretrained=pretrained)