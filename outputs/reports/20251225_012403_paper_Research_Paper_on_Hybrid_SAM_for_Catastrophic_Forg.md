---
title: Research Paper on Hybrid SAM for Catastrophic Forgetting in Continual Learning
type: paper
generated: 2025-12-25T01:24:03.232753
total_cycles: 1
---

# Title: Research on Hybrid Sharpness-Aware Minimization for Catastrophic Forgetting in Continual Learning

## Abstract
This paper investigates the effectiveness of applying Sharpness-Aware Minimization (SAM) exclusively to replayed samples in an episodic memory replay setting for mitigating catastrophic forgetting in continual learning. We hypothesized that this targeted application would enhance the robustness of consolidated past knowledge, leading to improved performance on previous tasks. Our experimental findings, conducted on SplitMNIST, demonstrated a refutation of this hypothesis, with the proposed method yielding a 3.35% lower average accuracy on previous tasks compared to a standard SGD baseline. This suggests that the chosen hybrid optimization strategy was detrimental to knowledge retention.

## Introduction
Catastrophic forgetting, the abrupt decline in performance on previously learned tasks when a neural network is sequentially trained on new tasks, remains a significant challenge in continual learning. Addressing this issue is crucial for developing intelligent systems capable of continuous adaptation without compromising accumulated knowledge. Episodic memory replay methods are a popular approach to combat forgetting, where a subset of past data is stored and replayed alongside new task data to reinforce prior learning. Sharpness-Aware Minimization (SAM) has emerged as a technique to improve model generalization and robustness by seeking flatter minima in the loss landscape. This research explored a novel integration of SAM within a memory replay framework, specifically targeting its application to replayed samples, with the aim of achieving a more robust consolidation of past knowledge.

## Methods
The research was conducted using the Discovery Engine, an autonomous AI scientist system employing a hypothesis-critique-experiment cycle. For Cycle 1, the following methodology was implemented:

**Hypothesis:** When using episodic memory replay in a continual learning setting, applying Sharpness-Aware Minimization (SAM) *exclusively* to the replayed samples (while using standard SGD for the current task's batch) will lead to a neural network that exhibits reduced catastrophic forgetting compared to a baseline using standard SGD for all samples.

**Falsification Condition:** If a baseline memory replay method, which applies standard SGD to both current and replayed samples, achieves an average accuracy on previous tasks that is equal to or higher than the proposed method across 5 tasks.

**Expected Effect:** A 3-5% improvement in average accuracy on previous tasks.

**Benchmark:** SplitMNIST, a dataset where digits are divided into sequential tasks (e.g., Task 0: digits 0-1, Task 1: digits 2-3, etc.).

**Experimental Setup:**
*   **Model:** A `SimpleMLP` (Multi-Layer Perceptron) with two hidden layers (256, 128 neurons) and ReLU activations, suitable for MNIST-like tasks.
*   **Dataset:** SplitMNIST with 5 tasks, each containing 2 digits. Standard MNIST normalization was applied.
*   **Memory Buffer:** An `Episodic MemoryBuffer` was used, storing 20 samples per class. Replay batch size was half of the current task batch size (32).
*   **Optimization:**
    *   **Baseline:** Standard SGD optimizer applied to a combined batch of current task data and replayed memory samples.
    *   **Proposed:** A custom `SAMReplayOptimizer` was implemented. This optimizer computed SGD gradients for the current task's batch and SAM-derived gradients for the replayed batch, subsequently combining these gradients for a single parameter update. SAM's perturbation strength (`rho`) was set to 0.05.
*   **Training:** Models were trained for 2 epochs per task with a learning rate of 0.01. Cross-entropy loss was used as the criterion.
*   **Evaluation:** After each task, the model was evaluated on all tasks learned so far. The key metric was the overall average accuracy on *previous* tasks (excluding the currently trained task).

## Results
The experiment involved training and evaluating both the baseline and the proposed methods across 5 sequential tasks on the SplitMNIST benchmark. The performance was measured by the overall average accuracy on previously learned tasks after each training increment.

*   **Overall Baseline Average Accuracy on Previous Tasks:** 89.28%
*   **Overall Proposed Average Accuracy on Previous Tasks:** 85.93%
*   **Delta Accuracy (Proposed - Baseline):** -3.35%

These results clearly indicate that the proposed method, which selectively applied SAM to replayed samples, did not yield the hypothesized improvement. Instead, it performed 3.35 percentage points *worse* than the baseline method.

## Discussion
The hypothesis, positing that targeted SAM application to replayed samples would reduce catastrophic forgetting, was unequivocally refuted. The observed negative delta of -3.35% suggests that this specific hybrid optimization strategy was detrimental to the model's ability to retain knowledge from past tasks. This outcome aligns with concerns raised during the critique phase regarding "Potential for Conflicting Optimization Goals." It appears that the attempt to encourage flatter minima for past knowledge via SAM, while simultaneously optimizing for potentially sharper minima for current tasks via SGD, may have led to unstable or conflicting gradient updates.

The expectation was that SAM would robustly consolidate old knowledge, but the implementation likely introduced interference rather than synergy. The computational overhead of SAM, even when partially applied, combined with the added complexity of mixing optimization strategies, did not translate into performance gains. This experiment, while on a simpler benchmark like SplitMNIST, provided a clear falsification, indicating that this particular approach to integrating SAM in continual learning is not effective.

## Conclusions
This research conclusively refutes the hypothesis that applying Sharpness-Aware Minimization exclusively to replayed samples, while using standard SGD for current task samples, mitigates catastrophic forgetting in continual learning. The proposed method resulted in a 3.35% *decrease* in average accuracy on previous tasks compared to a standard SGD replay baseline on SplitMNIST. This finding highlights the complexities of integrating advanced optimization techniques in continual learning and suggests that combining distinct optimization objectives for different parts of the training batch can lead to detrimental effects on knowledge retention. Future research should explore alternative ways to leverage SAM in continual learning, perhaps by applying it consistently across all samples or by investigating how to reconcile potentially conflicting optimization goals more effectively.

## References
None
