# DL Project 2: Fine-Tuning with LoRA for AGNEWS Classification

## Project Overview

This project applies **Low-Rank Adaptation (LoRA)** to fine-tune a **RoBERTa-base model** for the **AGNEWS text classification task**, while enforcing a strict upper bound of **<1 million trainable parameters**.

We systematically evaluated LoRA configurations, applied early stopping and regularization, and achieved **94.09% accuracy** and **0.9409 macro F1 score**, tuning just **888,580 parameters** (~0.71% of total model).

---

## Objective

To design a parameter-efficient pipeline for text classification under constraints:

- Use a transformer model without full fine-tuning
- Keep trainable parameters < 1M
- Maintain high accuracy and confidence
- Provide reproducible training, evaluation, and visualizations

---

##  Model Architecture

- **Base Model**: `roberta-base` (124M parameters)
- **Approach**: Inject **LoRA adapters** into transformer **query** and **value projection layers**
- **Total Trainable Parameters**: **888,580**
- **LoRA Injection Target**: Attention layers in each of the 12 transformer blocks

---

## LoRA Configuration

| Parameter       | Value       |
|----------------|-------------|
| Rank (r)       | 16          |
| Scaling Factor | 32          |
| Dropout        | 0.1         |
| Bias           | None        |
| Target Modules | Query + Value |

We found this configuration offers a strong tradeoff between generalization and parameter budget.

---

## Training Strategy

- **Framework**: HuggingFace Transformers + PEFT
- **Dataset Split**: 90% training / 10% validation
- **Hyperparameters**:
  - Optimizer: `AdamW`
  - Learning rate: `3e-4`
  - Batch size: `32`
  - Epochs: `5`
  - Weight Decay: `0.01`
  - Warmup Ratio: `0.1`
- **Early Stopping**: Patience = 2 (monitored on validation accuracy)

---

## Implementation Details

- **Libraries Used**:
  - HuggingFace `transformers`, `datasets`, `peft`
  - `wandb` for experiment tracking
  - `matplotlib` + `seaborn` for visualizations

- **Tokenization**: RoBERTa tokenizer with `max_length = 128`

- **Hardware**: NVIDIA GPU with 16GB+ VRAM

---

## Results & Analysis

### Final Performance

| Metric                  | Value    |
|-------------------------|----------|
| **Test Accuracy**       | 94.09%   |
| **Macro F1 Score**      | 0.9409   |
| **Trainable Parameters**| 888,580  |
| **Mean Confidence**     | 0.9162   |

### Class-wise F1 Scores

| Class     | F1 Score |
|-----------|----------|
| World     | 94.5%    |
| Sports    | 95.4%    |
| Business  | 93.6%    |
| Sci/Tech  | 93.2%    |

### Visual Insights

- **Training Curve** – Stable convergence within 4 epochs.
- ![image](https://github.com/user-attachments/assets/359c94b7-9892-463c-bc6b-894622ecebd4)

- **Validation Accuracy** – Peaked early due to effective early stopping.
- ![image](https://github.com/user-attachments/assets/355be91a-24d9-424a-8cb4-9ae42f2b1910)

- **Confusion Matrix** – Misclassifications mainly between Business & Sci/Tech.
- ![image](https://github.com/user-attachments/assets/15b335e7-9877-45d7-8f29-fc6be37bce00)

- **Prediction Confidence**:
  - >90% confidence for 64.7% of predictions
  - Min confidence: 0.526, Max: 0.996

---

## Ethical Considerations

- AGNEWS is generally clean, but domain bias is possible (e.g., overfitting to news tone).
- LoRA improves accessibility of fine-tuning but must be monitored to prevent malicious use.
- Misclassification analysis found no clear societal harm but highlighted potential improvements through bias audits.

---

## Ablation Study Highlights

- **LoRA Query-only vs. Query+Value**: Dual injection improved early training stability.
- **Dropout Tuning**: Lower dropout increased speed but reduced generalization.
- **Data Splits**: 90/10 offered better validation stability than 95/5.
- **Parameter Budgeting**: Staying under 1M preserved performance with minimal variance.

---

## Conclusion

Our project validates the effectiveness of **LoRA-based fine-tuning** under extreme resource constraints. We:

- Trained a high-performing RoBERTa model with **only 0.71%** of parameters
- Achieved **94%+ accuracy and F1 score**
- Maintained high confidence and balanced predictions across all classes
- Ensured reproducibility and low-overhead deployment

---

## Repository

🔗 [GitHub Repo](https://github.com/Abhi8602/Finetuning-with-Lora)  
📬 For queries: ak11553@nyu.edu | hv2201@nyu.edu
