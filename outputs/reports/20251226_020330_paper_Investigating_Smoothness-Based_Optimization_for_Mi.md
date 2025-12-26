---
title: Investigating Smoothness-Based Optimization for Mitigating Catastrophic Forgetting in Experience Replay
type: paper
generated: 2025-12-26T02:03:30.378928
total_cycles: 3
---

# Title: Investigating Smoothness-Based Optimization for Mitigating Catastrophic Forgetting in Experience Replay

## Abstract
This study investigates the efficacy of smoothness-inducing optimization techniques—specifically Sharpness-Aware Minimization (SAM) and Stochastic Weight Averaging (SWA)—in mitigating catastrophic forgetting within an Experience Replay (ER) framework. Through three iterative research cycles on the SplitCIFAR-10 benchmark, we evaluated whether seeking flatter minima in the loss landscape enhances knowledge retention. Our results indicate that while SAM is difficult to stabilize when mixing current and replayed gradients, SWA combined with Batch Normalization calibration provides a consistent, albeit marginal, performance gain (+0.52%). These findings suggest that weight-space smoothing is a more compatible approach for continual learning than adversarial gradient perturbations, though neither method fully resolves the challenge of weight drift.

## Introduction
Catastrophic forgetting remains a primary obstacle in continual learning, where neural networks lose previously acquired knowledge when trained on new tasks. Experience Replay (ER) mitigates this by interleaving past samples with new data. Recent work suggests that the geometry of the loss landscape, particularly the "flatness" of minima, plays a crucial role in generalization and robustness to parameter shifts. This research explores whether explicitly optimizing for flatter minima using SAM and SWA can reduce the drift caused by sequential updates.

## Methods
We employed a "hypothesis-critique-experiment" cycle. All experiments used the SplitCIFAR-10 benchmark (5 tasks, 2 classes per task) with a small CNN and a replay buffer of 500 samples.
- **Cycle 1:** SAM applied exclusively to replayed samples.
- **Cycle 2:** SAM applied to the entire combined batch of current and replayed samples.
- **Cycle 3:** SWA applied during the final 20% of training for each task, followed by post-task Batch Normalization calibration using a balanced mix of current and buffer data.

## Results
- **Cycle 1:** Direct application of SAM to replayed samples was found to be detrimental to knowledge retention compared to standard SGD-based ER. The hypothesis that SAM on replay samples alone would improve stability was refuted.
- **Cycle 2:** Applying SAM to the full combined batch yielded a negligible average accuracy increase of 0.18% on previous tasks, failing to show statistical or practical significance.
- **Cycle 3:** The SWA + BN calibration approach achieved a final average accuracy of 65.82% compared to a baseline of 65.30%. The resulting delta of +0.52% was positive but failed the pre-defined 1.0% success threshold.

## Discussion
The failure of SAM in Cycles 1 and 2 suggests that adversarial weight perturbations are highly sensitive to the non-i.i.d. nature of continual learning. Mixing gradients from current tasks and memory buffers appears to create unstable optimization trajectories when SAM is used. In contrast, Cycle 3's SWA approach provided the first consistent improvement. The use of weight averaging acts as a smoothing regularizer, finding a more central point in the flat region of the current task's landscape. However, the modest gains suggest "average drift": the model still shifts towards the minima of the most recent task, even if those minima are flatter. The requirement for BN calibration was also highlighted as a critical factor in preventing biased statistics from the current task dominating the network.

## Conclusions
Smoothing the loss landscape through weight-space averaging (SWA) is more effective and stable than gradient-space perturbations (SAM) for Experience Replay. While SWA provides a measurable benefit in stability, it is not a standalone solution for catastrophic forgetting. Future research should investigate whether combining SWA with stronger constraints on weight drift (e.g., functional regularization or synaptic consolidation) can amplify these stability gains.

## References
- Izmailov, P., Podoprikhin, A., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. *arXiv preprint arXiv:1803.05407*.
- Mirzadeh, S. I., Farajtabar, M., Pascanu, R., & Ghasemzadeh, H. (2020). Understanding the Role of Training Regimes in Continual Learning. *Advances in Neural Information Processing Systems*.
- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2020). Sharpness-Aware Minimization for Efficiently Improving Generalization. *International Conference on Learning Representations*.

