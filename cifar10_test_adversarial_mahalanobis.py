import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os


os.chdir('/Users/unicorn/Desktop/OOD-with-ViT-main/experiment')


processor = AutoImageProcessor.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10")
model = AutoModelForImageClassification.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10")
model.eval()


os.makedirs('cifar10_test_adversarial/penultimate', exist_ok=True)


sample_means = torch.from_numpy(np.load('cifar10_train_original/penultimate/penultimate_mean.npy'))
precision = torch.from_numpy(np.load('cifar10_train_original/penultimate/penultimate_precision.npy'))

def get_penultimate(module, input, output):
    penultimate.append(output[0])


file_list = os.listdir('cifar10_test_adversarial/attention_adversarial_image_numpy')


mahalanobis_distance_list = []

for i in range(1, len(file_list)):
    attention_adversarial_image = np.load(
        f"cifar10_test_adversarial/attention_adversarial_image_numpy/attention_adversarial_image_numpy_{i}.npy")
    attention_adversarial_image_tensor = torch.from_numpy(attention_adversarial_image).unsqueeze(0)

    penultimate = []

    hook = model.vit.layernorm.register_forward_hook(get_penultimate)

    with torch.no_grad():
        outputs = model(attention_adversarial_image_tensor)

    hook.remove()

    penultimate = penultimate[0]

    penultimate_path = f'cifar10_test_adversarial/penultimate/penultimate_{i}'
    np.save(penultimate_path, penultimate.numpy())


    gaussian_scores = []

    feature = penultimate[0]

    for sample_mean in sample_means:
        centered_feature = (feature - sample_mean).unsqueeze(0)
        gau_term = torch.mm(torch.mm(centered_feature, precision), centered_feature.t()).diag()
        gaussian_scores.append(gau_term.view(-1, 1))

    gaussian_scores = torch.cat(gaussian_scores, dim = 1)
    mahalanobis_distance, _ = gaussian_scores.min(dim = 1)

    mahalanobis_distance_list.append(mahalanobis_distance)


mahalanobis_distance_list = torch.cat(mahalanobis_distance_list, dim = 0)

torch.mean(mahalanobis_distance_list)

os.makedirs('cifar10_test_adversarial/mahalanobis', exist_ok=True)

mahalanobis_distance_list.numpy()

mahalanobis_distance_path = 'cifar10_test_adversarial/mahalanobis/mahalanobis_distance'
np.save(mahalanobis_distance_path, mahalanobis_distance_list.numpy())

