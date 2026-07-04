# 2. Method

This section introduces DRPR-SAR from two perspectives. Section 2.1 presents the observation analysis that motivates representation decoupling, and Section 2.2 describes the overall method, including the DRPR-SAR framework and TGDM.

## 2.1 Observation Analysis

![Fig. 2 Visualization of adversarial perturbations in the decoupled space](./2.1_Decoupled_Adversarial_Perturbation_Visualization.png)

Fig. 2 visualizes how an adversarial perturbation behaves after the SAR image is decomposed into redundant and discriminative components. The rows compare the clean input, the redundant stream \(R(x)\), the discriminative stream \(D(x)\), and the perturbation responses in each space. The key observation is that the perturbation is not uniformly distributed: the discriminative stream carries much stronger perturbation-sensitive variation, while the redundant stream remains visually and structurally more stable. This figure provides the main empirical motivation for DRPR-SAR: instead of trying to erase perturbations everywhere, the method aims to route them away from the representation used for final prediction.

![Table I Classification accuracy of decoupled components](./2.1_Decoupled_Component_Classification_Accuracy.png)

Table I reports how classifiers trained or tested on the original image, the redundant component, and the discriminative component behave on MSTAR and FUSAR-Ship. Its role is to show that the two streams are not arbitrary visual decompositions. The redundant stream still preserves non-negligible class information, while the discriminative stream is more strongly tied to decision-specific cues and is therefore more vulnerable to attack-induced changes. This supports the central assumption of the method: robust recognition can be built on a stable redundant representation, provided that perturbation-sensitive variations are separated and controlled elsewhere.

## 2.2 Overall Method

![Fig. 3 Overall framework of DRPR-SAR](./2.2_DRPR_SAR_Framework_Architecture.png)

Fig. 3 shows the complete DRPR-SAR framework and how the different modules interact. Stage 1 pre-trains TGDM to produce a redundant stream \(R(x)\) and a discriminative stream \(D(x)\). Stage 2 performs adversarial fine-tuning: the deployed target classifier \(C_r\) predicts from the redundant stream, the perturbation-routing branch encourages attack-induced changes to be expressed in the discriminative stream, and knowledge distillation transfers clean semantic guidance from a teacher model to \(C_r\). The figure is the main architectural guide for the method because it clarifies which branch is used for final inference and which branch is used to absorb perturbation-sensitive behavior during training.

During inference, the model mainly relies on the redundant stream for classification. The discriminative stream absorbs and represents perturbation-sensitive variations, while the redundant stream provides a more stable representation for target recognition. This design allows adversarial effects to be controlled in the representation space rather than simply suppressed at the input level.

![Fig. 3 TGDM encoder architecture](./2.2_TGDM_Encoder_Architecture.png)

Fig. 3 further illustrates the encoder architecture used in the rotation-enhanced VQ-VAE based TGDM. This figure zooms in on the representation extractor inside TGDM: convolutional layers capture local SAR scattering patterns, normalization and nonlinear activation stabilize the feature transformation, and residual blocks refine the latent representation before quantization and reconstruction. Its purpose is to show how TGDM obtains a structured latent space in which stable redundant information can be reconstructed while perturbation-sensitive variations remain separable for later routing.

