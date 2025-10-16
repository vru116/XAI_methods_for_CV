import torch
import torch.nn.functional as F
import numpy as np

class AbsCAMInit:
    def __init__(self, model, target_layer='layer4', device='cuda'):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
        else:
            raise ValueError(f"Layer {self.target_layer} not found in model")

    def __call__(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        # B, C, H, W = gradients.shape
        # print(gradients.shape)
        abs_gradients = gradients.abs()
        weights = abs_gradients.mean(dim=(2, 3), keepdim=True)  # shape: (B, C, 1, 1)
        # print(weights.shape)
        weighted_activations = weights * activations  # shape: (B, C, H, W)
        # print(weighted_activations.shape)
        cam = weighted_activations.sum(dim=1, keepdim=True)  # shape: (B, 1, H, W)

        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().squeeze().cpu().numpy()


class AbsCAMFinal:
    def __init__(self, model, target_layer='layer4', device='cuda'):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
        else:
            raise ValueError(f"Layer {self.target_layer} not found in model")

    def __call__(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients
        # print(gradients.shape)
        activations = self.activations

        B, C, H1, W1 = gradients.shape
        # print(gradients.shape)
        abs_gradients = gradients.abs()
        weights = abs_gradients.mean(dim=(2, 3), keepdim=True)  # shape: (B, C, 1, 1)
        # print(weights.shape)
        weighted_activations = weights * activations  # shape: (B, C, H, W)
        # print(weighted_activations.shape)
        M0_k = weighted_activations

        M0_k = F.interpolate(M0_k, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)

        M0_k = M0_k - M0_k.view(B, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        M0_k = M0_k / (M0_k.view(B, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1) + 1e-8)
        # (B, C, H, W)
        # print(f"M0k {M0_k.shape}")
        X0 = input_tensor  # (B, 3, H, W)
        M1_k = X0.unsqueeze(1) * M0_k.unsqueeze(2)  # (B, C, 3, H, W)
        # print(M1_k.shape)
        M1_k = M1_k.view(B * C, 3, input_tensor.shape[2], input_tensor.shape[3])  # (B*C, 3, H, W)

        with torch.no_grad():
            y_M1 = self.model(M1_k)  # (B*C, num_classes)
            y_M1 = torch.softmax(y_M1, dim=1)
            y_c = y_M1[:, class_idx]  # (B*C,)

        y_c = y_c.view(B, C, 1, 1)
        # print(y_c.shape)
        # y_c = y_c.abs()

        # print("M0_k:", M0_k.min().item(), M0_k.max().item(), M0_k.mean().item())
        # print("y_c:", y_c.min().item(), y_c.max().item(), y_c.mean().item())
        # print("(y_c * M0_k).sum:", ((y_c * M0_k).sum(dim=1)).min().item(), ((y_c * M0_k).sum(dim=1)).max().item())

        Lc = F.relu((y_c * M0_k).sum(dim=1))
        # Lc = (y_c * M0_k).sum(dim=1)
        return Lc.squeeze().cpu().detach().numpy()
