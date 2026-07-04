# 3.1 Quantitative Results

This folder summarizes the direct quantitative evidence for DRPR-SAR under PGD attacks and analyzes how perturbations are distributed across the decoupled streams.

![Table III PGD attacks under different Lp norms](./3.1_PGD_Attacks_Different_Lp_Norms.png)

Table III reports clean and adversarial accuracy under PGD attacks with different \(L_p\) constraints on MSTAR and FUSAR-Ship. It is used to test whether the defense remains effective when the attack norm and perturbation budget change. DRPR-SAR keeps strong accuracy under \(L_\infty\), \(L_2\), and \(L_1\) attacks, rather than only under the training-style \(L_\infty\) setting. This matters because it suggests that the gain comes from the representation-level routing mechanism, not from overfitting to a single perturbation geometry. The FUSAR-Ship results are especially important because this dataset contains sea clutter and more complex target/background interactions.

![Table IV Perturbation magnitudes in decoupled streams](./3.2_Perturbation_Magnitudes_Decoupled_Streams.png)

Table IV measures the magnitude of the original input perturbation \(\delta\) and the induced changes in the redundant stream \(\delta_R\) and discriminative stream \(\delta_D\). This table is the quantitative counterpart of the visual observation in Fig. 2. Across different norms and datasets, \(\delta_D\) is much larger than \(\delta_R\), showing that attack-induced changes are concentrated in the discriminative branch while the redundant branch remains comparatively stable. This directly verifies the intended behavior of perturbation routing: adversarial effects are not eliminated everywhere, but are redirected away from the stream used for final prediction.


