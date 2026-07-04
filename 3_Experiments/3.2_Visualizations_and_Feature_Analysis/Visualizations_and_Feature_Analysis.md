# 3.2 Visualizations and Feature Analysis

This folder explains how DRPR-SAR organizes feature representations and classification behavior under attack.

![Fig. 4 t-SNE feature distribution on MSTAR](./3.9_MSTAR_TSNE_Feature_Distribution_PGD.png)

Fig. 4 shows the t-SNE feature distribution on MSTAR under PGD attack. It compares clean features, adversarial features, the redundant stream \(R(x^a)\), and the discriminative stream \(D(x^a)\). Clean samples form compact and well-separated clusters, while adversarial examples blur class boundaries. After TGDM, \(R(x^a)\) restores a more compact class-wise structure, which means the redundant stream preserves stable target semantics. By contrast, \(D(x^a)\) is more diffuse and entangled, showing that perturbation-sensitive variation has been pushed toward the auxiliary discriminative stream.

![Fig. 5 Confusion matrices on FUSAR-Ship](./3.10_FUSAR_Ship_Confusion_Matrices_PGD.png)

Fig. 5 shows confusion matrices on FUSAR-Ship under PGD attack. It helps readers see the class-level effect of perturbation routing rather than only aggregate accuracy. The clean input has a clearer diagonal pattern, while the adversarial input introduces obvious confusion among ship and background-related categories. Prediction from \(R(x^a)\) recovers more diagonal structure, indicating that stable class semantics are preserved in the redundant stream. Prediction from \(D(x^a)\) is much more confused, which supports the interpretation that the discriminative stream mainly carries perturbation-sensitive variations.


