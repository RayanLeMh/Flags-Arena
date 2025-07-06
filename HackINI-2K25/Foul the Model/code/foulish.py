import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as F_trans
from PIL import Image
import json
import numpy as np

# Load models
resnet = models.resnet50(weights='IMAGENET1K_V1')
squeezenet = models.squeezenet1_1(weights='IMAGENET1K_V1')
resnet.eval()
squeezenet.eval()

# Load ImageNet class indices
with open('imagenet_class_index.json', 'r') as f:
    imagenet_classes = json.load(f)

# Find class indices
lion_idx = next(int(idx) for idx, (code, _) in imagenet_classes.items() if code == "n02129165")
cat_idx = next(int(idx) for idx, (code, _) in imagenet_classes.items() if code == "n02123045")  # Tabby cat

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)
squeezenet.to(device)

# Load and preprocess base image
base_image = Image.open('./assets/lion.png').convert('RGB')
base_tensor = transforms.ToTensor()(base_image).to(device)  # [0,1] range

# Server preprocessing simulation (differentiable)
def server_preprocess(x):
    h, w = x.shape[1], x.shape[2]
    aspect_ratio = w / h
    if h < w:
        new_h, new_w = 256, int(256 * aspect_ratio)
    else:
        new_h, new_w = int(256 / aspect_ratio), 256
    x_resized = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
    top = (new_h - 224) // 2
    left = (new_w - 224) // 2
    x_cropped = x_resized[:, :, top:top+224, left:left+224]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x_normalized = (x_cropped - mean) / std
    return x_normalized.squeeze(0)

# Initialize adversarial image
x_adv = base_tensor.clone().detach().requires_grad_(True)

# PGD parameters
epsilon = 0.05
step_size = 0.01
num_steps = 200
target_resnet = torch.tensor([lion_idx], device=device)
target_squeezenet = torch.tensor([cat_idx], device=device)

# Adversarial attack loop
for i in range(num_steps):
    # Apply server preprocessing
    x_preprocessed = server_preprocess(x_adv).unsqueeze(0)
    
    # Model predictions
    resnet_out = resnet(x_preprocessed)
    squeezenet_out = squeezenet(x_preprocessed)
    
    # Losses
    loss_resnet = F.cross_entropy(resnet_out, target_resnet)
    loss_squeezenet = F.cross_entropy(squeezenet_out, target_squeezenet)
    total_loss = loss_resnet + loss_squeezenet
    
    # Backpropagation
    total_loss.backward()
    
    # PGD step
    with torch.no_grad():
        grad = x_adv.grad.sign()
        x_adv = x_adv - step_size * grad
        x_adv = torch.max(torch.min(x_adv, base_tensor + epsilon), base_tensor - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
    
    # Reset gradients
    x_adv = x_adv.detach().requires_grad_(True)
    
    # Log progress
    if i % 10 == 0:
        resnet_pred = resnet_out.argmax().item()
        squeezenet_pred = squeezenet_out.argmax().item()
        print(f'Step {i}: ResNet={resnet_pred}, SqueezeNet={squeezenet_pred}, Loss={total_loss.item():.4f}')

# Save adversarial image
adv_image = transforms.ToPILImage()(x_adv.cpu())
adv_image.save('adversarial_image.jpg')
print("Adversarial image saved.")