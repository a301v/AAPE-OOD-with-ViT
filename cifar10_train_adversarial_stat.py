import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
from typing import List
from torchvision.datasets import CIFAR10
from sklearn.covariance import EmpiricalCovariance


os.chdir('/Users/unicorn/Desktop/OOD-with-ViT-main/experiment')

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR-10 Dataset Class
test_dataset = CIFAR10(root='/Users/unicorn/Desktop/OOD-with-ViT-main/data', train=True, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


processor = AutoImageProcessor.from_pretrained("nielsr/vit-base-patch16-224-in21k-finetuned-cifar10")
model = AutoModelForImageClassification.from_pretrained("nielsr/vit-base-patch16-224-in21k-finetuned-cifar10")
model.eval()

os.makedirs('cifar10_train_adversarial/penultimate', exist_ok=True)

group_lasso = EmpiricalCovariance(assume_centered=False)

def get_penultimate(module, input, output):
    penultimate.append(output[0])


# Process the first 100 images
class_to_features = [[] for _ in range(10)]

for i, (image_tensor, target) in enumerate(test_loader):

    if i >= 500:
        break

    if i % 100 == 0:
        print(f'working on {i}-th image...')

    penultimate = []

    hook = model.vit.layernorm.register_forward_hook(get_penultimate)

    with torch.no_grad():
        outputs = model(image_tensor)

    hook.remove()

    penultimate = penultimate[0]

    penultimate_path = f'cifar10_train_adversarial/penultimate/penultimate_{i}'
    np.save(penultimate_path, penultimate.numpy())


    for feature, label in zip(penultimate, target):
        class_to_features[label.item()].append(feature.view(1, -1))


for i in range(len(class_to_features)):
    class_to_features[i] = torch.cat(class_to_features[i], dim = 0)



sample_means = [None] * 10

for i, features in enumerate(class_to_features):
    sample_means[i] = torch.mean(features, dim = 0).numpy()


X = []

for list_feature, cls_mean in zip(class_to_features, sample_means):
    X.append(list_feature - cls_mean)

X = torch.cat(X, dim = 0).numpy()
group_lasso.fit(X)
precision = torch.from_numpy(group_lasso.precision_).float()


penultimate_mean_path = f'cifar10_train_adversarial/penultimate/penultimate_mean'
np.save(penultimate_mean_path, sample_means)

penultimate_precision_path = f'cifar10_train_adversarial/penultimate/penultimate_precision'
np.save(penultimate_precision_path, precision)