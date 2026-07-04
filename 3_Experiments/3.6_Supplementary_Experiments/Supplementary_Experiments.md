# 3.6 Supplementary Experiments

This folder contains additional results that further support the generality and deployability of DRPR-SAR.

![Table 1 SAMPLE synthetic-to-measured robust accuracy](./3.13_SAMPLE_Synthetic_to_Measured_Robust_Accuracy.png)

Table 1 reports robust accuracy in the SAMPLE synthetic-to-measured setting. This is a cross-domain scenario: models are trained only on synthetic SAR images and evaluated on measured SAR images, where scattering patterns, background clutter, and speckle noise differ from the training domain. The table compares DRPR-SAR with AT and TRADES on ConvNeXt and VGG16 under white-box attacks and transfer-based attacks. Its role is to show that the redundant representation learned by DRPR-SAR remains useful even when the data distribution shifts from synthetic to measured SAR imagery.

![Table 2 Clean accuracy of the TGDM inference pipeline](./3.14_TGDM_Inference_Clean_Accuracy.png)

Table 2 compares clean accuracy between the original backbone and the deployed `TGDM + C_r` inference pipeline on MSTAR and FUSAR. This table addresses a practical deployment question: whether adding TGDM before the classifier damages clean-sample recognition. The results show that clean accuracy remains competitive across ResNet50 and DenseNet121, meaning the decoupling module does not introduce severe degradation while enabling robust inference.

![Fig. 1 Clean confusion matrices on MSTAR](./3.15_MSTAR_Clean_Confusion_Matrices_TGDM_Pipeline.png)

Fig. 1 shows normalized confusion matrices on clean MSTAR samples for original ResNet50/DenseNet121 and the corresponding `TGDM + C_r` pipelines. The purpose is to inspect class-wise behavior, not just average accuracy. The strong diagonal patterns show that TGDM largely preserves the original recognition structure on clean ground-target samples. Any remaining off-diagonal responses are limited and mainly occur among visually or structurally similar vehicle classes, rather than showing a systematic class bias introduced by TGDM.

![Fig. 2 Clean confusion matrices on FUSAR](./3.16_FUSAR_Clean_Confusion_Matrices_TGDM_Pipeline.png)

Fig. 2 shows normalized confusion matrices on clean FUSAR samples for original backbones and their `TGDM + C_r` pipelines. FUSAR is more difficult than MSTAR because maritime categories such as Cargo, Fishing, Tanker, and Other Ship can have similar scattering characteristics. The figure shows that most confusion remains concentrated among these inherently similar categories, while the deployed pipeline still maintains meaningful diagonal dominance. This supports the claim that TGDM does not create arbitrary recognition bias and preserves clean-sample behavior while adding the structure needed for robust defense.


