# 3.3 Comparative Experiment

This folder compares DRPR-SAR with representative defense methods on CNN and Transformer backbones across multiple SAR ATR datasets.

![3.3 CNN robust accuracy comparison on MSTAR](./3.3_MSTAR_CNN_Robust_Accuracy_Comparison.png)

Fig. 3.3 shows the CNN-based robust accuracy comparison on MSTAR. DRPR-SAR achieves strong average robustness across ResNet50, WideResNet-34, VGG13, DenseNet121, and MobileNet, and performs favorably under white-box and black-box attacks such as FGSM, PGD, C&W, Square Attack, OnePixel, and AutoAttack. This indicates that perturbation routing does not rely on a particular backbone or a single attack setting, but consistently improves resistance to adversarial perturbations in standard SAR ground-target recognition.

![3.4 CNN robust accuracy comparison on FUSAR-Ship](./3.4_FUSAR_Ship_CNN_Robust_Accuracy_Comparison.png)

Fig. 3.4 shows the CNN-based robust accuracy comparison on FUSAR-Ship. Compared with MSTAR, FUSAR-Ship contains larger variations in ship appearance and is affected by sea clutter, non-ship regions, and complex backgrounds, making robust recognition more challenging. DRPR-SAR still achieves competitive average robust accuracy, showing that the redundant stream does not simply preserve background information but extracts stable SAR representations that remain discriminative in complex maritime scenes.

![3.5 CNN robust accuracy comparison on ATRNet-STAR](./3.5_ATRNet_STAR_CNN_Robust_Accuracy_Comparison.png)

Fig. 3.5 shows the CNN-based robust accuracy comparison on ATRNet-STAR. ATRNet-STAR contains more categories, more complex scenes, and richer imaging conditions, making it useful for evaluating large-scale SAR ATR robustness. DRPR-SAR maintains stable improvements across multiple CNN backbones, suggesting that its benefit comes from representation-level information decoupling and perturbation routing rather than overfitting to a small dataset or a specific scenario.

![3.6 Robust accuracy comparison on Transformer architectures](./3.6_Transformer_Robust_Accuracy_Comparison.png)

Fig. 3.6 shows the robust accuracy comparison of DRPR-SAR on Transformer architectures, including ViT-B, HiViT-B, and Swin-T. Consistent with the CNN results, DRPR-SAR improves robustness under different attacks on Transformer backbones. This indicates that the core mechanism is not tied to convolutional local receptive fields, but can serve as a more general representation-level defense strategy for different feature extractors.

