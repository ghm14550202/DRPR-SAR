# 3.4 Parameter Sensitivity Analysis

This folder studies how key loss weights affect robust accuracy.

![Fig. 6 Hyperparameter sensitivity on MSTAR](./3.11_MSTAR_Hyperparameter_Sensitivity_PGD.png)

Fig. 6 shows the sensitivity of robust accuracy on MSTAR to \(\lambda_{C_r}\) and \(\lambda_{PR}\). These two parameters control the redundant-stream classification constraint and the perturbation routing constraint, respectively. If the former is too weak, the final classifier loses semantic discrimination; if the latter is too weak, perturbations cannot be sufficiently guided into the discriminative stream. However, overly strong weights may also disturb the balance between the two streams. The performance surface shows that DRPR-SAR remains robust within a reasonable parameter range while requiring a proper balance between classification and routing objectives.

![Fig. 7 Distillation weight sensitivity on FUSAR-Ship](./3.12_FUSAR_Ship_Distillation_Weight_Sensitivity.png)

Fig. 7 shows the effect of the knowledge distillation weight \(\lambda_{KD}\) on FUSAR-Ship under different attacks. Knowledge distillation allows the redundant-stream classifier to inherit class semantics from a clean teacher, compensating for information loss caused by robust training and decoupling. An appropriate distillation weight achieves a better balance between clean accuracy and robustness, while too small a weight provides insufficient semantic constraint and too large a weight may overemphasize the teacher distribution and affect robust optimization.


