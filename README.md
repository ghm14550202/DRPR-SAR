# DRPR-SAR

DRPR-SAR is a defense framework for adversarially robust SAR automatic target recognition. Instead of treating adversarial defense as direct perturbation suppression, it reformulates the problem as representation-level perturbation routing: stable task-relevant information is preserved for final recognition, while perturbation-sensitive variations are guided toward an auxiliary branch.

This repository is organized according to the logic of the paper:

1. [Introduction](./1_Introduction/README.md): explains the motivation, the limitation of data-centric and model-centric defenses, and the shift from suppression to routing.
2. [Method](./2_Method/README.md): introduces task-guided decoupling, rotation-enhanced VQ-VAE, dynamic perturbation routing, and knowledge distillation.
3. [Experiments](./3_Experiments/README.md): follows Sections B-F of the experimental part, excluding the setup section, and includes supplementary experiments.
4. [Code](./4_Code/README.md): records the code release status.

The main idea is that adversarial perturbations primarily distort discriminative features, while redundant SAR information remains comparatively stable and still predictive. DRPR-SAR uses this difference to build a two-stream representation, allowing the deployed classifier to operate on a more stable redundant stream while routing attack-induced variations away from the final decision path.



