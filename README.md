# Adv-OOD-with-ViT
Advanced OOD detection, leveraging ViT with attention based adversarial attack

This experiance is greatly influenced by https://github.com/Simcs/OOD-with-ViT



1. cifar10_train_adversarial_image.py
   1) this program will make cifar-10 based train dataset.
   2) get attention map (made by attention rollout mechanism), using cifar-10 pre-trained ViT.
   3) get attention mask, based on attention score strategy (top 50% or bottom 50%).
   4) apply reversed-fgsm attack on original cifar-10 image tensor, save it as perturbed image.
   5) mix original image & perturbed image, using attention mask.
    
2. cifar10_test_adversarial_image.py
   1) this program will make cifar-10 based test dataset.
   2) use pseudo-target to get the attention maps (there's no true label for unknown data).
   3) same as above, except dataset.
  
3. cifar100_test_adversarial_image.py
   1) this program will make cifar-100 based test dataset.
   2) use pseudo-target to get the attention maps (there's no true label for unknown data).
   3) same as above, except dataset.

4. cifar10_train_adversarial_stat.py
   1) get penultimate feature vectors of cifar-10 attention-adversarial train datset, created by 1.
   2) get mean and precision (inverse Cov matrix) of cifar-10 train dataset.
  
5. cifar10_test_adversarial_mahalanobis.py
   1) get penultimate feature vectors of cifar-10 attention-adversarial test datset, created by 2.
   2) get ood-score (mahalanobis distance) of each cifar-10 test dataset, using mean and precision of cifar-10 train dataset.
  
6. cifar100_test_adversarial_mahalanobis.py
   1) get penultimate feature vectors of cifar-100 attention-adversarial test datset, created by 3.
   2) get ood-score (mahalanobis distance) of each cifar-100 test dataset, using mean and precision of cifar-10 train dataset.
