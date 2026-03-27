import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):

        self.model.zero_grad()

        output = self.model(input_tensor)

        #  used raw logit
        score = output[:, class_idx]

        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2,3), keepdim=True)

        cam = (weights * activations).sum(dim=1)

        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam