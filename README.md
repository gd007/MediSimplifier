# MediSimplifier

**LoRA Fine-Tuning for Medical Discharge Summary Simplification**

[![Hugging Face Models](https://img.shields.io/badge/🤗%20Models-MediSimplifier-yellow)](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-medisimplifier-blue)](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Overview

MediSimplifier fine-tunes open-source LLMs using LoRA to simplify medical discharge summaries to a 6th-grade reading level, improving patient comprehension of their medical documents.

**Course Project:** Technion DS25 Deep Learning  
**Authors:** Guy Dor, Shmulik Avraham

## Key Results

| Model | ROUGE-L | SARI | BERTScore | FK-Grade | Improvement |
|-------|---------|------|-----------|----------|-------------|
| **OpenBioLLM-8B** 🏆 | **0.6749** | **74.64** | **0.9498** | 7.16 | +157.3% |
| Mistral-7B | 0.6491 | 73.79 | 0.9464 | **6.91** | +65.9% |
| BioMistral-7B-DARE | 0.6318 | 73.01 | 0.9439 | 6.95 | +53.3% |

**Achievement:** ~50% readability reduction (FK 14.5 → ~7.0)

## Project Structure

```
├── notebooks/
│   ├── MediSimplifier_Part1.ipynb  # Data prep & ground truth generation
│   ├── MediSimplifier_Part2.ipynb  # Baseline evaluation
│   ├── MediSimplifier_Part3.ipynb  # LoRA fine-tuning & ablation
│   └── MediSimplifier_Part4.ipynb  # Evaluation & analysis
├── results/
│   ├── ablation/                   # Ablation study results
│   ├── baseline/                   # Zero-shot baseline metrics
│   ├── evaluation/                 # Final evaluation metrics
│   ├── training/                   # Training logs
│   └── figures/                    # All visualizations
├── report/
│   └── MediSimplifier_IEEE_Paper_with_Figures.pdf
├── MediSimplifier_Master_Document.md
└── MediSimplifier_Final_Presentation.pdf
```

## Quick Start

### Installation

```bash
git clone https://github.com/GuyDor007/MediSimplifier.git
cd MediSimplifier
pip install -r requirements.txt
```

### Using the Fine-Tuned Models

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load best model (OpenBioLLM-8B)
base_model = AutoModelForCausalLM.from_pretrained(
    "aaditya/Llama3-OpenBioLLM-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model, 
    "GuyDor007/MediSimplifier-LoRA-Adapters",
    subfolder="openbiollm_8b_lora"
)
tokenizer = AutoTokenizer.from_pretrained("aaditya/Llama3-OpenBioLLM-8B")

# Simplify medical text
SYSTEM_MESSAGE = "You are a helpful medical assistant that simplifies complex medical text for patients."
TASK_INSTRUCTION = """Simplify the following medical discharge summary in plain language for patients with no medical background.
Guidelines:
- Replace medical jargon with everyday words
- Keep all important information
- Use short, clear sentences
- Aim for a 6th-grade reading level"""

prompt = f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
{TASK_INSTRUCTION}

{your_medical_text}<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Methodology

### Dataset
- **Source:** [Asclepius-Synthetic-Clinical-Notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) (10K samples)
- **Ground Truth:** Generated using Claude Opus 4.5
- **Splits:** Train (7,999) / Val (999) / Test (1,001)

### Models Compared
| Model | Type | Architecture |
|-------|------|--------------|
| OpenBioLLM-8B | Medical | Llama3 |
| BioMistral-7B-DARE | Medical | Mistral |
| Mistral-7B-Instruct-v0.2 | General | Mistral |

### LoRA Configuration (Optimal)
| Parameter | Value |
|-----------|-------|
| Rank (r) | 32 |
| Alpha (α) | 64 |
| Target Modules | q, k, v, o projections |
| rsLoRA | True |
| Trainable Params | 27.3M (0.38%) |

## Ablation Study Findings

| Phase | Finding |
|-------|---------|
| **Rank** | r=32 optimal (contradicts Hu et al. 2021) |
| **Modules** | all_attn best despite 2x parameters |
| **Data Size** | More data = better (+5.5-6.6% ROUGE-L) |
| **rsLoRA** | Adopted based on literature |

## Key Research Findings

1. **Ranking Reversal:** Worst zero-shot model (OpenBioLLM) achieved best fine-tuned performance (+157%)
2. **Medical Pretraining:** Advantage disappears after task-specific fine-tuning
3. **Consistent Success:** All models achieve ~50% readability reduction
4. **Statistical Significance:** All pairwise ROUGE-L differences significant (p < 0.001)

## Resources

| Resource | Link |
|----------|------|
| 🤗 Models | [MediSimplifier-LoRA-Adapters](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters) |
| 🤗 Dataset | [medisimplifier-dataset](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset) |
| 📄 Paper | [IEEE Format Report](report/MediSimplifier_IEEE_Paper_with_Figures.pdf) |
| 📊 Presentation | [Final Presentation](MediSimplifier_Final_Presentation.pdf) |

## Citation

```bibtex
@misc{medisimplifier2026,
  author = {Dor, Guy and Avraham, Shmulik},
  title = {MediSimplifier: LoRA Fine-Tuning for Medical Discharge Summary Simplification},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/GuyDor007/MediSimplifier}}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Technion DS25 Deep Learning Course
- Base model teams: OpenBioLLM, Mistral, BioMistral
- Asclepius dataset creators (starmpcc)
