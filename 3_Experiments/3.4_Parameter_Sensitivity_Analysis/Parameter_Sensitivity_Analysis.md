# 3.4 Parameter Sensitivity Analysis

This folder studies how key loss weights affect robust accuracy.

<img src="./3.11_MSTAR_Hyperparameter_Sensitivity_PGD.png" alt="Fig. 6 Hyperparameter sensitivity on MSTAR" width="60%">

Fig. 6 shows how robust accuracy on MSTAR changes with \(\lambda_{C_r}\) and \(\lambda_{PR}\), which weight the redundant-stream classification objective and the perturbation routing objective. The broad high-accuracy region indicates that the method is not extremely sensitive to a single exact setting. At the same time, the figure explains the trade-off: too little classification weight weakens semantic discrimination in the redundant stream, while too little routing weight fails to move perturbation-sensitive changes away from the prediction path. The figure therefore serves as evidence that DRPR-SAR needs a balance between stable classification and active routing, but has a reasonably tolerant operating range.

<img src="./3.12_FUSAR_Ship_Distillation_Weight_Sensitivity.png" alt="Fig. 7 Distillation weight sensitivity on FUSAR-Ship" width="60%">

Fig. 7 shows how the knowledge distillation weight \(\lambda_{KD}\) affects clean and robust accuracy on FUSAR-Ship under multiple attacks. This figure is important because distillation is used to preserve class semantics in the redundant-stream classifier after decoupling. A moderate \(\lambda_{KD}\) helps transfer clean teacher knowledge without overwhelming the robust training objective. When the weight is too small, the redundant stream receives insufficient semantic guidance; when it is too large, the model may overfit the teacher distribution and lose robustness. The figure therefore clarifies why distillation is useful but must be balanced.

