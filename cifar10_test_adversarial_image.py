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

os.chdir('/Users/unicorn/Desktop/OOD-with-ViT-main/experiment')

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR-10 Dataset Class
test_dataset = CIFAR10(root='/Users/unicorn/Desktop/OOD-with-ViT-main/data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Load the Vision Transformer model
processor = AutoImageProcessor.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10")
model = AutoModelForImageClassification.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10")
model.eval()


class ViTAttentionRollout:
    def __init__(self,
                 head_fusion: str = 'max',
                 discard_ratio: float = 0.9):
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def spatial_rollout(self, attentions: List[torch.Tensor]):
        return self.rollout(
            attentions=attentions,
            reshape_mask=True,
        )

    def temporal_rollout(self, attentions: List[torch.Tensor]):
        return self.rollout(
            attentions=attentions,
            reshape_mask=False,
        )

    def rollout(self, attentions: List[torch.Tensor], reshape_mask=True):
        result = torch.eye(attentions[0].size(-1)).unsqueeze(0)
        result = result.repeat((attentions[0].size(0), 1, 1))

        with torch.no_grad():
            for attention in attentions:
                if self.head_fusion == 'mean':
                    fused_attention_heads = attention.mean(axis=1)
                elif self.head_fusion == 'max':
                    fused_attention_heads = attention.max(axis=1)[0]
                elif self.head_fusion == 'min':
                    fused_attention_heads = attention.min(axis=1)[0]
                else:
                    raise ValueError('Invalid attention head fusion type.')

                flat = fused_attention_heads.view(fused_attention_heads.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * self.discard_ratio), dim=-1, largest=False)
                indices = [idx[idx != 0] for idx in indices]
                for i in range(flat.size(0)):
                    flat[i, indices[i]] = 0

                I = torch.eye(fused_attention_heads.size(-1))
                a = (fused_attention_heads + 1.0 * I) / 2
                a = a / torch.sum(a, dim=-1, keepdim=True)

                result = torch.bmm(a, result)

        mask = result[:, 0, 1:]
        if reshape_mask:
            width = int(mask.size(-1) ** 0.5)
            mask = mask.reshape(-1, width, width)
        mask = mask.numpy()
        mask = mask / np.max(mask)

        return mask

    def get_visualized_masks(self, img, mask):
        img_h, img_w, _ = img.shape
        mask = cv2.resize(mask, (img_h, img_w), interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)


# Create output directory for attention maps
os.makedirs('cifar10_test_adversarial/attention_map', exist_ok=True)

# Create output directory for attention rollouts
os.makedirs('cifar10_test_adversarial/attention_rollout', exist_ok=True)



# Process the first 100 images
for i, (input_image, target) in enumerate(test_loader):

    if i % 100 == 0:
        print(f'Working on {i}-th image...')

    if i >= 1000:
        break

    attn_map_output_path = f"cifar10_test_adversarial/attention_map/attention_map_{i}.png"
    attn_rollout_output_path = f"cifar10_test_adversarial/attention_rollout/attention_rollout_{i}.npy"

    # Check if files already exist
    if os.path.exists(attn_map_output_path) and os.path.exists(attn_rollout_output_path):
        print(f"Files for image {i} already exist, skipping...")
        continue

    attentions = []


    # Register hooks to capture attention weights
    def get_attention(module, input, output):
        attentions.append(output[1].detach().cpu())


    hooks = []
    for name, module in model.vit.encoder.layer.named_children():
        hooks.append(module.attention.attention.register_forward_hook(get_attention))

    with torch.no_grad():
        outputs = model(input_image, output_attentions=True)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Check if attentions were captured
    if len(attentions) == 0:
        print(f"No attentions captured for image {i}.")
        continue

    rollout = ViTAttentionRollout(head_fusion='max', discard_ratio=0.5)
    attn_rollout = rollout.spatial_rollout(attentions)

    input_image_pil = transforms.ToPILImage()(input_image.squeeze(0))
    input_image_np = np.array(input_image_pil)

    # Get the visualized attention map
    visualized_map = rollout.get_visualized_masks(input_image_np, attn_rollout[0])

    # Save the attention map
    attn_map_output_path = f"cifar10_test_adversarial/attention_map/attention_map_{i}.png"
    cv2.imwrite(attn_map_output_path, visualized_map)

    # Save the attention rollout numpy data
    attn_rollout_output_path = f"cifar10_test_adversarial/attention_rollout/attention_rollout_{i}"
    np.save(attn_rollout_output_path, attn_rollout)

print("Attention maps saved in the 'attention_map' folder.")
print("Attention rollouts saved in the 'attention_rollout' folder.")



########################################################

import torch.nn.functional as F


class MakeAttentionAdversarialImage:
    def __init__(self, model):
        self.model = model

    # Get attention mask from attention rollout
    def get_attention_mask(self, image_tensor, attn_rollout, perturb_ratio = 1):
        image_tensor_pil = transforms.ToPILImage()(image_tensor.squeeze(0))
        image_tensor_np = np.array(image_tensor_pil)
        img_h, img_w, _ = image_tensor_np.shape


        threshold = np.percentile(attn_rollout, perturb_ratio * 100)
        original_attention_mask = (attn_rollout >= threshold).astype(int)
        original_attention_mask_tensor = torch.tensor(original_attention_mask).unsqueeze(0).float()

        resized_attention_mask_tensor = F.interpolate(original_attention_mask_tensor, size = (img_h, img_w), mode = 'nearest')
        resized_attention_mask = resized_attention_mask_tensor.squeeze(0).numpy()

        return resized_attention_mask


    # get gradient for fgsm attack
    def get_grad(self, model, image_tensor, target, temperature):
        image_tensor.requires_grad = True

        CEloss = nn.CrossEntropyLoss()

        model_output = model(image_tensor)

        logits = model_output.logits / temperature
        model_loss = CEloss(logits, target)

        model.zero_grad()
        model_loss.backward()
        data_grad = image_tensor.grad.data

        return data_grad


    # apply reversed fgsm attack to the original image tensor
    def reversed_fgsm_attack(self, image_tensor, target, epsilon, temperature):

        data_grad = self.get_grad(model, image_tensor, target, temperature)

        sign_data_grad = data_grad.sign()
        perturbed_image_tensor = image_tensor - epsilon * sign_data_grad

        return perturbed_image_tensor


    # mix original and perturbed image w.r.t attention mask
    def get_mixed_image(self, image_tensor, perturbed_image_tensor, resized_attention_mask):
        masked_original_image = image_tensor.squeeze(0).detach() * resized_attention_mask
        masked_perturbed_image = perturbed_image_tensor.squeeze(0).detach().numpy() * (1 - resized_attention_mask)

        mixed_image = masked_original_image + masked_perturbed_image

        return mixed_image





# denormalize the image tensor to visualize
def denormalize(image_tensor, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    denormalized_image_tensor = image_tensor * std + mean
    return torch.clamp(denormalized_image_tensor, 0, 1)




# Create output directory for attention masks
os.makedirs('cifar10_test_adversarial/attention_mask', exist_ok=True)

# Create output directory for perturbed image
os.makedirs('cifar10_test_adversarial/perturbed_image_numpy', exist_ok=True)

# Create output directory for attention adversarial image
os.makedirs('cifar10_test_adversarial/attention_adversarial_image_numpy', exist_ok=True)


os.makedirs('cifar10_test_adversarial/perturbed_image', exist_ok=True)
os.makedirs('cifar10_test_adversarial/attention_adversarial_image', exist_ok=True)



model.eval()
image_maker = MakeAttentionAdversarialImage(model)

for i, (image_tensor, target) in enumerate(test_loader):

    if i % 100 == 0:
        print(f'Working on {i}-th image...')


    if i >= 100:
        break

    output = model(image_tensor)
    pseudo_target = torch.argmax(output.logits, dim = 1)


    attn_rollout = np.load(f"cifar10_test_adversarial/attention_rollout/attention_rollout_{i}.npy")
    attention_mask = image_maker.get_attention_mask(image_tensor, attn_rollout, 1)
    perturbed_image_tensor = image_maker.reversed_fgsm_attack(image_tensor, pseudo_target, epsilon = 0.1, temperature = 1)
    attention_adversarial_image = image_maker.get_mixed_image(image_tensor, perturbed_image_tensor, attention_mask)

    perturbed_image = perturbed_image_tensor.detach().numpy()


    # Save the attention mask numpy data
    attn_mask_path = f"cifar10_test_adversarial/attention_mask/attention_mask_{i}"
    np.save(attn_mask_path, attention_mask)

    # Save the perturbed image numpy data
    perturbed_numpy_path = f"cifar10_test_adversarial/perturbed_image_numpy/perturbed_image_numpy_{i}"
    np.save(perturbed_numpy_path, perturbed_image)

    # Save the attention adversarial image numpy data
    attn_adv_numpy_path = f"cifar10_test_adversarial/attention_adversarial_image_numpy/attention_adversarial_image_numpy_{i}"
    np.save(attn_adv_numpy_path, attention_adversarial_image)



    denormalized_perturbed_image_tensor = denormalize(perturbed_image_tensor)

    denormalized_perturbed_image = denormalized_perturbed_image_tensor.squeeze(0).detach().numpy()
    denormalized_perturbed_image = np.transpose(denormalized_perturbed_image, (1, 2, 0))
    denormalized_perturbed_image_rgb = cv2.cvtColor((denormalized_perturbed_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


    denormalized_attention_adversarial_image_tensor = denormalize(attention_adversarial_image)

    denormalized_attention_adversarial_image = denormalized_attention_adversarial_image_tensor.squeeze(0).detach().numpy()
    denormalized_attention_adversarial_image = np.transpose(denormalized_attention_adversarial_image, (1, 2, 0))
    denormalized_attention_adversarial_image_rgb = cv2.cvtColor((denormalized_attention_adversarial_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)



    # Save the perturbed image data
    perturbed_path = f"cifar10_test_adversarial/perturbed_image/perturbed_image_{i}.jpg"
    cv2.imwrite(perturbed_path, denormalized_perturbed_image_rgb)


    # Save the attention adversarial image data
    attn_adv_path = f"cifar10_test_adversarial/attention_adversarial_image/attention_adversarial_image_{i}.jpg"
    cv2.imwrite(attn_adv_path, denormalized_attention_adversarial_image_rgb)