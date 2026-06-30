# 3.5 Ablation Study

This folder analyzes the contribution of individual modules in DRPR-SAR.

![3.7 Ablation study](./3.7_Ablation_Study_Clean_PGD_Settings.png)

Fig. 3.7 shows the ablation results of TGDM, the target classifier, the perturbation routing branch, knowledge distillation, and rotation enhancement. When only the main classifier is used, robustness under PGD attack is weak. As TGDM, perturbation routing, and knowledge distillation are progressively introduced, robust accuracy improves noticeably. This demonstrates that DRPR-SAR is not driven by a single trick, but by the cooperation of three components: TGDM provides a separable representation space, perturbation routing guides attack-induced variations into the discriminative stream, and knowledge distillation helps the redundant stream preserve class semantics from the clean teacher.

![3.8 Effect of rotation-enhanced VQ-VAE](./3.8_Rotation_Enhanced_VQVAE_Effect.png)

Fig. 3.8 shows the effect of the rotation-enhanced VQ-VAE on dual-stream perturbation response and quantization stability. With rotation enhancement, the redundant stream has lower FID and higher correlation, indicating better stability before and after attack. The discriminative stream shows stronger perturbation response, which matches the goal of concentrating perturbation-sensitive variations in the auxiliary branch. Meanwhile, lower quantization error and higher codebook usage indicate that rotation enhancement also improves discrete representation learning in VQ-VAE.

