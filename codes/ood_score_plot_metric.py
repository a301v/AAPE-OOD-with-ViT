import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve

os.chdir('/Users/unicorn/Desktop/OOD-with-ViT-main/experiment')

id_score = np.load('cifar10_test_adversarial/mahalanobis/mahalanobis_distance.npy')
ood_score = np.load('cifar100_test_adversarial/mahalanobis/mahalanobis_distance.npy')



# Plot the histograms
plt.figure(figsize=(10, 6))
bins = np.linspace(min(min(id_score), min(ood_score)), max(max(id_score), max(ood_score)), 100)
plt.hist(id_score, bins=bins, alpha=0.5, label='ID Score (CIFAR-10)')
plt.hist(ood_score, bins=bins, alpha=0.5, label='OOD Score (CIFAR-100)')
plt.xlabel('Mahalanobis Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Mahalanobis Distances for ID and OOD Scores')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


max_score = max(max(id_score), max(ood_score))
min_score = min(min(id_score), min(ood_score))

id_score_scaled = (id_score - min_score) / (max_score - min_score)
ood_score_scaled = (ood_score - min_score) / (max_score - min_score)

all_score_scaled = np.concatenate([id_score_scaled, ood_score_scaled])


id_label = [0] * len(id_score)
ood_label = [1] * len(ood_score)

all_label = np.concatenate([id_label, ood_label])


from sklearn.metrics import auc, roc_curve, precision_recall_curve

def auroc(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    return fpr, tpr, auc(fpr, tpr)

def aupr(labels, preds):
    precision, recall, _ = precision_recall_curve(labels, preds)
    return precision, recall, auc(recall, precision)

fpr, tpr, roc_auc = auroc(all_label, all_score_scaled)
precision, recall, pr_auc = aupr(all_label, all_score_scaled)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.4f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()
