# 3.2 Visualizations and Feature Analysis

This folder explains how DRPR-SAR organizes feature representations and classification behavior under attack.

![Fig. 4 t-SNE feature distribution on MSTAR](./3.9_MSTAR_TSNE_Feature_Distribution_PGD.png)

Fig. 4 shows the t-SNE feature distribution on MSTAR under PGD attack. Clean features usually form clear class clusters, while adversarial examples disturb inter-class boundaries and make the distribution more entangled. After DRPR-SAR decoupling, the redundant stream \(R(x^a)\) better preserves the clean clustering structure, indicating that stable class semantics are retained. In contrast, the discriminative stream \(D(x^a)\) is more scattered, suggesting that perturbation-sensitive information is routed into the auxiliary branch.

![Fig. 5 Confusion matrices on FUSAR-Ship](./3.10_FUSAR_Ship_Confusion_Matrices_PGD.png)

Fig. 5 shows the confusion matrices on FUSAR-Ship under PGD attack. Comparing clean inputs, adversarial inputs, the redundant stream, and the discriminative stream shows that attacks significantly increase class confusion, while the redundant stream recovers more correct predictions and preserves more stable class discrimination in complex maritime scenes. The discriminative stream exhibits stronger confusion, further indicating that it carries more attack-induced sensitive variations.


