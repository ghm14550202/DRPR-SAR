# 3. Experiments

This section follows the organization of the experimental part in the paper, excluding Section A, which describes the dataset and experimental setup. The results here correspond to Sections B-F of the paper, with an additional folder for supplementary experiments.

## Structure

1. [3.1 Quantitative Results](./3.1_Quantitative_Results/Quantitative_Results.md): corresponds to Section B and reports direct quantitative robustness under PGD attacks and analyzes how perturbations are distributed across the decoupled streams.
2. [3.2 Visualizations and Feature Analysis](./3.2_Visualizations_and_Feature_Analysis/Visualizations_and_Feature_Analysis.md): corresponds to Section C and explains feature organization and classification behavior under attack using t-SNE plots and confusion matrices.
3. [3.3 Comparative Experiment](./3.3_Comparative_Experiment/Comparative_Experiment.md): corresponds to Section D and compares DRPR-SAR with representative defenses across CNN and Transformer backbones on multiple SAR ATR datasets.
4. [3.4 Parameter Sensitivity Analysis](./3.4_Parameter_Sensitivity_Analysis/Parameter_Sensitivity_Analysis.md): corresponds to Section E and studies how key loss weights affect robust accuracy.
5. [3.5 Ablation Study](./3.5_Ablation_Study/Ablation_Study.md): corresponds to Section F and evaluates the contribution of individual modules in DRPR-SAR.
6. [3.6 Supplementary Experiments](./3.6_Supplementary_Experiments/Supplementary_Experiments.md): provides additional evidence for cross-domain robustness, clean-sample behavior, and deployability.

Together, these experiments support the main claim of DRPR-SAR: adversarial robustness in SAR ATR can be improved by decoupling stable task-relevant information from perturbation-sensitive variations and actively routing the latter away from the final decision path.




