import torch
import cv2
import numpy as np
import os

def generate_cam(model, input_tensor, class_idx, img_name):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_conv = model.features[-1]
    forward = final_conv.register_forward_hook(forward_hook)
    backward = final_conv.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    forward.remove()
    backward.remove()

    grad = gradients[0].squeeze()
    act = activations[0].squeeze()
    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros(act.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = cam.numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = np.uint8(cam * 255)
    cam = cv2.resize(cam, (224, 224))
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    os.makedirs("static", exist_ok=True)
    cam_path = f"static/cam_{img_name}"
    cv2.imwrite(cam_path, cam)

    return cam_path
