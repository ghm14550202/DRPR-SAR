# 1. Introduction

DRPR-SAR addresses adversarial robustness in synthetic aperture radar (SAR) automatic target recognition (ATR). SAR ATR is often used in security-sensitive scenarios, where models are expected to maintain high recognition accuracy on clean samples while remaining reliable under adversarial perturbations, cross-scene variations, and complex backgrounds.

Existing defenses generally follow two paradigms. Model-centric defenses, such as adversarial training and architectural modification, aim to strengthen the classifier itself. Data-centric defenses, such as adversarial detection and input purification, attempt to identify or suppress perturbations before classification. Although these methods can reduce the effect of attacks, they usually lack explicit control over how adversarial effects are encoded inside the representation space.

The key difference of DRPR-SAR is that it does not treat robustness simply as perturbation suppression. Instead, it reformulates adversarial defense as perturbation routing. The basic observation is that adversarial perturbations mainly distort discriminative features close to the classification boundary, whereas comparatively redundant SAR information remains more stable and still preserves class semantics. Based on this observation, DRPR-SAR separates stable task-relevant information from perturbation-sensitive variations and routes attack-induced changes toward an auxiliary branch, allowing the final classifier to rely more on stable representations.

This forms a clear problem-to-method logic: identify the limitations of conventional defense paradigms, exploit the difference between stable and sensitive information in SAR representations, and build a robust recognition framework through representation decoupling, perturbation routing, and knowledge distillation.

![Fig. 1 Defense paradigm comparison](./1.1_Defense_Paradigm_Comparison.png)

Fig. 1 shows the conceptual position of DRPR-SAR relative to existing SAR ATR defense paradigms. Data-centric defenses act before recognition, either detecting adversarial inputs or purifying the whole input image; model-centric defenses strengthen the classifier through training or architectural changes, but still operate on a unified representation. DRPR-SAR introduces a third route: it explicitly controls where adversarial effects are encoded inside the representation space. The figure is useful as a high-level map of the paper because it explains why the proposed method focuses on decoupling stable task-relevant information from perturbation-sensitive variations instead of only suppressing perturbations at the input or classifier level.



