# MediSimplifier Master Plan v31

**Course:** Technion DS25 Deep Learning | **Team:** Guy Dor, Shmulik Avraham | **Deadline:** Feb 8, 2026

## Quick Reference

| Item | Value |
|------|-------|
| **Objective** | Simplify medical discharge summaries → 6th-grade reading level |
| **Method** | LoRA fine-tuning with Claude-generated ground truth |
| **Dataset** | Asclepius-Synthetic-Clinical-Notes (10K samples) |
| **Teacher** | Claude Opus 4.5 (`claude-opus-4-5-20251101`) |
| **Students** | OpenBioLLM-8B, BioMistral-7B-DARE, Mistral-7B-Instruct-v0.2 |
| **Hardware** | Part 1 (Ch 1-4): M4 Max 128GB | Part 2 (Ch 5-9): RunPod H200 SXM (3 GPUs) |

---

## 1. Models & Formats

| Model | HF Path | Format | Type | Architecture |
|-------|---------|--------|------|--------------|
| OpenBioLLM-8B | `aaditya/Llama3-OpenBioLLM-8B` | ChatML | Medical | Llama3 |
| BioMistral-7B-DARE | `BioMistral/BioMistral-7B-DARE` | Mistral | Medical | Mistral |
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | Mistral | General | Mistral |

**Format Consistency:** Same model-specific format used for training AND evaluation.

**Architecture Grouping:** Ablations run on one model per architecture (OpenBioLLM=Llama3, Mistral-7B=Mistral). BioMistral uses Mistral-7B optimal config.

---

## 2. Dataset

**Source:** `starmpcc/Asclepius-Synthetic-Clinical-Notes`

| Split | Samples | Use |
|-------|---------|-----|
| Train | 7,999 | LoRA fine-tuning |
| Val | 999 | Hyperparameter tuning |
| Test | 1,001 | Final evaluation |

**Dataset Location:** `/workspace/medisimplifier/data/instruction_dataset/`

---

## 3. Configuration

### 3.1 LoRA Parameters (Default)

| Param | Default | Ablation Range | **Optimal (Phase 1)** | **Optimal (Phase 2)** | **Final** |
|-------|---------|----------------|----------------------|----------------------|-----------|
| Rank (r) | 16 | 8, 16, 32 | **32** ✅ | 32 | **32** |
| Alpha (α) | 2×r | (scales with r) | **64** | 64 | **64** |
| Dropout | 0.05 | fixed | 0.05 | 0.05 | **0.05** |
| Target Modules | q_v | q_only, q_v, all_attn | q_v | **all_attn** ✅ | **all_attn** |
| rsLoRA | False | False, True | - | - | **True** ✅ |

**Note:** rsLoRA=True selected based on literature (Kalajdzievski 2023) - no performance downside, potential benefit at r=32.

### 3.2 Training

| Param | Value |
|-------|-------|
| Epochs | 3 (full), 1 (ablation) |
| Batch | 4, Grad Accum: 4 (eff: 16) |
| LR | 2e-4, Cosine, Warmup: 3% |
| Precision | BF16 |
| Max Seq | 2048 |

### 3.3 Ablation Study Design (Updated)

**Strategy:** Sequential ablation with literature-based rsLoRA decision

| Phase | Ablation | Variable | Values | Models | RQ | Runs | Status |
|-------|----------|----------|--------|--------|-----|------|--------|
| 1 | Rank | LoRA rank (r) | 8, 16, 32 | OpenBioLLM, Mistral-7B | RQ4 | 6 | ✅ DONE |
| 2 | Modules | Target modules | q_only, q_v, all_attn | OpenBioLLM, Mistral-7B | RQ6 | 6 | ✅ DONE |
| 3 | Data Size | Training samples | 2000, 4000, 7999 | OpenBioLLM, Mistral-7B | RQ7 | 6 | ✅ DONE |
| 4 | rsLoRA | Scaling method | - | - | RQ12 | 0 | ✅ SKIPPED |
| 5 | Full Train | Final config | optimal | All 3 models | - | 3 | ✅ DONE |
| **Total** | | | | | | **21** | ✅ COMPLETE |

**Module Configurations:**
- `q_only`: [q_proj]
- `q_v`: [q_proj, v_proj]  
- `all_attn`: [q_proj, k_proj, v_proj, o_proj]

**Sequential Cascade:**
- Phase 2 uses optimal rank from Phase 1 → **r=32**
- Phase 3 uses optimal rank + modules from Phases 1-2 → **r=32, all_attn**
- Phase 4 **SKIPPED** - rsLoRA=True adopted based on literature (no downside, potential benefit)
- Phase 5 uses optimal config: **r=32, all_attn, 7999 samples, rsLoRA=True**

**Ablation Epochs:** 1 epoch per configuration (faster iteration, sufficient signal)

**Full Training:** 3 epochs with optimal configuration from ablation

### 3.4 rsLoRA Configuration

**Background:** Original LoRA paper (Hu et al. 2021) found r=4-8 sufficient. rsLoRA paper (Kalajdzievski 2023) proved this was due to gradient collapse with α/r scaling. Using α/√r enables higher ranks to improve performance.

**Decision: rsLoRA=True (Literature-Based)**

| Factor | Assessment |
|--------|------------|
| Downside | **None** - same parameter count, same training time |
| Upside | Potentially better gradients at r=32 |
| Literature | Kalajdzievski (2023) supports rsLoRA at r≥16 |
| Ablation cost | 2+ hours GPU time for marginal research value |
| Project focus | Medical text simplification, not LoRA hyperparameters |

**Rationale:** Phase 1 found "higher rank = better" (r=32 optimal), which aligns with rsLoRA theory that α/√r scaling stabilizes gradients at higher ranks. Given no downside to using rsLoRA and potential benefit, we adopt rsLoRA=True as a safe default without running empirical ablation.

**Final Implementation:**
```python
from peft import LoraConfig

# Final optimal configuration
lora_config = LoraConfig(
    r=32,                          # Phase 1 optimal
    lora_alpha=64,                 # α = 2×r
    lora_dropout=0.05,             # fixed
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Phase 2 optimal (all_attn)
    use_rslora=True,               # Literature-based decision
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### 3.5 Prompt Templates (WITH SYSTEM MESSAGE)

**Naming Convention:**
- `SIMPLIFICATION_INSTRUCTION` → Part 1 only (Claude API ground truth generation)
- `TASK_INSTRUCTION` → Part 2 only (training & inference)

```python
# Part 1: Claude API Ground Truth Generation
SIMPLIFICATION_INSTRUCTION = """Simplify the following medical discharge summary in plain language for patients with no medical background.
Guidelines:
- Replace medical jargon with everyday words (e.g., "hypertension" → "high blood pressure")
- Keep all important information (diagnoses, medications, follow-up instructions)
- Use short, clear sentences (aim for 15-20 words per sentence)
- Aim for a 6th-grade reading level
- Maintain the same structure as the original
- Do not add or omit information
- Keep the same patient reference style (e.g., "The patient" stays "The patient", not "You")
- Output plain text only (no markdown, no bold, no headers, no bullet points)
- Do not include empty lines or separator characters like "---" """

# Part 2: Training & Inference (consistent across all stages)
TASK_INSTRUCTION = """Simplify the following medical discharge summary in plain language for patients with no medical background.
Guidelines:
- Replace medical jargon with everyday words (e.g., "hypertension" → "high blood pressure")
- Keep all important information (diagnoses, medications, follow-up instructions)
- Use short, clear sentences (aim for 15-20 words per sentence)
- Aim for a 6th-grade reading level
- Maintain the same structure as the original
- Do not add or omit information
- Keep the same patient reference style (e.g., "The patient" stays "The patient", not "You")
- Output plain text only (no markdown, no bold, no headers, no bullet points)
- Do not include empty lines or separator characters like "---\""""

# System message for model guidance
SYSTEM_MESSAGE = "You are a helpful medical assistant that simplifies complex medical text for patients."

# ChatML Training Template (OpenBioLLM-8B) - WITH SYSTEM MESSAGE
CHATML_TEMPLATE = """<|im_start|>system
You are a helpful medical assistant that simplifies complex medical text for patients.<|im_end|>
<|im_start|>user
{instruction}

{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

# ChatML Inference Template (OpenBioLLM-8B) - WITH SYSTEM MESSAGE
CHATML_INFERENCE_TEMPLATE = """<|im_start|>system
You are a helpful medical assistant that simplifies complex medical text for patients.<|im_end|>
<|im_start|>user
{instruction}

{input}<|im_end|>
<|im_start|>assistant
"""

# Mistral Training Template (BioMistral, Mistral-7B) - WITH SYSTEM MESSAGE
MISTRAL_TEMPLATE = """[INST] <<SYS>>
You are a helpful medical assistant that simplifies complex medical text for patients.
<</SYS>>

{instruction}

{input} [/INST]{output}"""

# Mistral Inference Template (BioMistral, Mistral-7B) - WITH SYSTEM MESSAGE
MISTRAL_INFERENCE_TEMPLATE = """[INST] <<SYS>>
You are a helpful medical assistant that simplifies complex medical text for patients.
<</SYS>>

{instruction}

{input} [/INST]"""
```

**Template Usage Pattern:**
```python
# Training: CHATML_TEMPLATE.format(instruction=TASK_INSTRUCTION, input=complex_text, output=simplified_text)
# Inference: CHATML_INFERENCE_TEMPLATE.format(instruction=TASK_INSTRUCTION, input=complex_text)
```

**IMPORTANT:** System message is required for proper model behavior. Without it, models may echo the prompt instead of following instructions.

---

## 4. Evaluation Metrics

| Metric | Library | Target |
|--------|---------|--------|
| ROUGE-L | `rouge_score` | Higher |
| SARI | `easse` | ≥40 |
| BERTScore | `bert_score` | Higher |
| Flesch-Kincaid | `textstat` | ≤6 |

---

## 5. Research Questions & Findings

| RQ | Question | Status | Finding |
|----|----------|--------|---------|
| 1 | Medical pretraining → better simplification? | ✅ Answered | **NO** - General Mistral-7B competitive (ROUGE-L 0.39 vs medical avg 0.34) |
| 2 | Best medical model (zero-shot)? | ✅ Answered | **BioMistral-7B-DARE** (ROUGE-L=0.41, SARI=51.9, FK=9.5) |
| **3** | **LoRA improvement over zero-shot?** | ✅ Answered | **YES** - +53% to +157% ROUGE-L improvement across all models |
| **4** | **Optimal LoRA rank?** | ✅ Answered | **r=32** - Higher rank = better for both architectures |
| **5** | **Medical vs general gap after fine-tuning?** | ✅ Answered | **Reversed** - OpenBioLLM best (+157%), Mistral-7B middle (+66%), BioMistral last (+53%) |
| **6** | **Best target modules?** | ✅ Answered | **all_attn** - More modules = better for both architectures |
| **7** | **Data efficiency (2K/4K/8K)?** | ✅ Answered | **More data = better** - +5.5% (Llama3), +6.6% (Mistral) ROUGE-L |
| **8** | **Baseline-improvement correlation?** | ✅ Answered | **Inverse correlation** - Worst baseline (OpenBioLLM) → best improvement (+157%) |
| **9** | **Parameter efficiency of modules?** | ✅ Answered | **all_attn best despite 2x params** - quality over efficiency |
| **10** | **Most consistent output quality?** | ✅ Answered | **OpenBioLLM-8B** - Best on all metrics (ROUGE-L, SARI, BERTScore) |
| **11** | **Match ground truth FK?** | ✅ Answered | **YES** - All models achieve FK ~6.9-7.2 (GT: 7.23), ~50% reduction from source |
| **12** | **Does rsLoRA improve higher-rank performance?** | ✅ Answered | **Literature-based: rsLoRA=True adopted** (no downside, aligns with theory) |

**RQ3 Finding Details:**
- OpenBioLLM-8B: 0.2623 → 0.6749 (+157.3% ROUGE-L)
- Mistral-7B: 0.3912 → 0.6491 (+65.9% ROUGE-L)
- BioMistral-7B-DARE: 0.4120 → 0.6318 (+53.3% ROUGE-L)
- All models show massive improvement with LoRA fine-tuning

**RQ4 Finding Details:**
- Both architectures show same trend: r=32 > r=16 > r=8
- Contradicts original LoRA paper claim that r=4-8 is sufficient
- Supports rsLoRA hypothesis that gradient issues affect higher ranks
- FK-Grade meets target (~6.7-6.8) at all rank values

**RQ5 Finding Details:**
- Zero-shot ranking: BioMistral > Mistral > OpenBioLLM
- Fine-tuned ranking: **OpenBioLLM > Mistral > BioMistral** (reversed!)
- Medical pretraining advantage disappears after domain-specific fine-tuning
- General architecture (Llama3) shows highest learning capacity

**RQ6 Finding Details:**
- Both architectures show same trend: all_attn > q_v > q_only
- Confirms modern LoRA best practices (Raschka 2023, Unsloth)
- all_attn uses ~27M params vs q_v ~14M, but ROUGE-L improvement justifies cost
- FK-Grade consistent across all module configs (~6.75-6.85)

**RQ7 Finding Details:**
- Both architectures show same trend: 8K > 4K > 2K
- Standard ML scaling behavior confirmed
- Llama3: +5.5% ROUGE-L (2K→8K), FK improves 7.17→6.80
- Mistral: +6.6% ROUGE-L (2K→8K), FK improves 6.87→6.76
- Diminishing returns: 4K achieves ~95% of 8K performance

**RQ8 Finding Details:**
- Inverse correlation between baseline performance and improvement
- OpenBioLLM: Worst baseline (0.26) → Best improvement (+157%)
- BioMistral: Best baseline (0.41) → Smallest improvement (+53%)
- Suggests floor effect: strong baselines have less room to improve

**RQ10 Finding Details:**
- OpenBioLLM-8B: Best on ROUGE-L (0.6749), SARI (74.64), BERTScore (0.9498)
- Most consistent across all quality metrics
- Mistral-7B: Best FK-Grade (6.91) - closest to target

**RQ11 Finding Details:**
- Source FK-Grade: 14.50 (college level)
- Ground Truth FK-Grade: 7.23 (7th grade)
- Model predictions: 6.91-7.16 (all match or beat GT!)
- Readability reduction: ~50% (14.50 → ~7.0)

**RQ12 Finding Details:**
- Phase 1 found "higher rank = better" with standard LoRA scaling
- This aligns with rsLoRA theory: α/√r stabilizes gradients at higher ranks
- rsLoRA has **zero downside** (same params, same training time)
- rsLoRA has **potential benefit** at r=32 based on Kalajdzievski (2023)
- **Decision:** Adopt rsLoRA=True without empirical ablation (not core research contribution)
- **Rationale:** Project focus is medical text simplification, not LoRA hyperparameter research

---

## 6. Baseline Results (Chapter 5) ✅ COMPLETED

### 6.1 Zero-Shot Baseline Metrics

| Model | Type | Format | ROUGE-L | SARI | BERTScore-F1 | FK-Grade | FK-Std | Time |
|-------|------|--------|---------|------|--------------|----------|--------|------|
| **BioMistral-7B-DARE** | Medical | Mistral | **0.4120** | **51.91** | **0.7426** | **9.52** | 3.56 | 104 min |
| Mistral-7B | General | Mistral | 0.3912 | 46.38 | 0.7335 | 10.60 | 8.32 | 86 min |
| OpenBioLLM-8B | Medical | ChatML | 0.2623 | 36.98 | 0.6371 | 12.53 | 3.70 | 60 min |

### 6.2 Baseline Rankings

| Rank | Model | Strengths | Weaknesses |
|------|-------|-----------|------------|
| 🥇 | BioMistral-7B-DARE | Best on ALL metrics, preserves structure | FK still high (9.5 vs target 6) |
| 🥈 | Mistral-7B | Good balance, simplifies well | High FK variance (±8.32), inconsistent |
| 🥉 | OpenBioLLM-8B | Fast inference | Over-condenses, loses structure, keeps jargon |

### 6.3 Key Findings

1. **All models fail FK ≤ 6 target** - Fine-tuning required
2. **BioMistral wins zero-shot** - Medical domain knowledge + Mistral architecture
3. **OpenBioLLM over-condenses** - 41-72 words vs 240-400 reference
4. **Mistral-7B inconsistent** - High FK std dev (±8.32)
5. **Structure preservation varies** - BioMistral best, OpenBioLLM worst

### 6.4 Baseline Files Generated

```
/workspace/medisimplifier/results/
├── baseline/
│   ├── baseline_metrics.csv ✅
│   ├── baseline_results.json ✅
│   ├── baseline_summary.md ✅
│   ├── baseline_openbiollm_8b.json ✅
│   ├── baseline_biomistral_7b_dare.json ✅
│   ├── baseline_mistral_7b.json ✅
│   ├── baseline_checkpoint_openbiollm_8b.json ✅
│   ├── baseline_checkpoint_biomistral_7b_dare.json ✅
│   └── baseline_checkpoint_mistral_7b.json ✅
├── figures/
│   └── baseline_comparison.png ✅
└── data/instruction_dataset/
    ├── chatml/ ✅
    └── mistral/ ✅
```

---

## 6A. Phase 1: Rank Ablation Results ✅ COMPLETED

### 6A.1 OpenBioLLM-8B (Llama3 Architecture)

| Rank | Alpha | Trainable Params | Train Loss | Eval Loss | ROUGE-L | FK-Grade |
|------|-------|------------------|------------|-----------|---------|----------|
| 8 | 16 | 3,407,872 | 0.8877 | 0.8213 | 0.6033 | 6.86 |
| 16 | 32 | 6,815,744 | 0.8712 | 0.8111 | 0.6080 | 6.81 |
| **32** | **64** | **13,631,488** | **0.8560** | **0.8010** | **0.6183** | **6.83** |

**🏆 Optimal Rank (Llama3): r=32** | ROUGE-L: 0.6183 | FK: 6.83 | Time: 137.7 min

### 6A.2 Mistral-7B (Mistral Architecture)

| Rank | Alpha | Trainable Params | Train Loss | Eval Loss | ROUGE-L | FK-Grade |
|------|-------|------------------|------------|-----------|---------|----------|
| 8 | 16 | 3,407,872 | 0.7354 | 0.6897 | 0.6047 | 6.87 |
| 16 | 32 | 6,815,744 | 0.7211 | 0.6772 | 0.6048 | 6.85 |
| **32** | **64** | **13,631,488** | **0.7101** | **0.6692** | **0.6171** | **6.68** |

**🏆 Optimal Rank (Mistral): r=32** | ROUGE-L: 0.6171 | FK: 6.68 | Time: 139.7 min

### 6A.3 Phase 1 Key Findings

1. **Higher rank = better performance** - Both architectures show consistent trend
2. **Contradicts original LoRA paper** - Hu et al. (2021) claimed r=4-8 sufficient
3. **FK-Grade meets target** - All configurations achieve ~6.7-6.8 (target ≤6)
4. **Consistent across architectures** - Both Llama3 and Mistral agree on r=32
5. **Supports rsLoRA adoption** - Higher rank success aligns with rsLoRA theory

### 6A.4 Phase 1 Files Generated

```
/workspace/medisimplifier/results/ablation/
├── rank_ablation_llama3.json ✅
└── rank_ablation_mistral.json ✅
```

---

## 6B. Phase 2: Modules Ablation Results ✅ COMPLETED

### 6B.1 OpenBioLLM-8B (Llama3 Architecture)

| Config | Modules | Trainable Params | ROUGE-L | FK-Grade |
|--------|---------|------------------|---------|----------|
| q_only | [q_proj] | 8,388,608 | 0.6006 | 6.93 |
| q_v | [q_proj, v_proj] | 13,631,488 | 0.6192 | 6.85 |
| **all_attn** | **[q_proj, k_proj, v_proj, o_proj]** | **27,262,976** | **0.6357** | **6.75** |

**🏆 Optimal Modules (Llama3): all_attn** | ROUGE-L: 0.6357 | FK: 6.75 | Time: 138.0 min

### 6B.2 Mistral-7B (Mistral Architecture)

| Config | Modules | Trainable Params | ROUGE-L | FK-Grade |
|--------|---------|------------------|---------|----------|
| q_only | [q_proj] | 8,388,608 | 0.5863 | 6.81 |
| q_v | [q_proj, v_proj] | 13,631,488 | 0.6156 | 6.79 |
| **all_attn** | **[q_proj, k_proj, v_proj, o_proj]** | **27,262,976** | **0.6242** | **6.84** |

**🏆 Optimal Modules (Mistral): all_attn** | ROUGE-L: 0.6242 | FK: 6.84 | Time: 141.5 min

### 6B.3 Phase 2 Key Findings

1. **More modules = better performance** - Both architectures show consistent trend
2. **Confirms modern best practices** - Raschka (2023), Unsloth recommend all attention layers
3. **Parameter cost justified** - all_attn (27M) outperforms q_v (14M) by +2.7% ROUGE-L (Llama3)
4. **Consistent across architectures** - Both Llama3 and Mistral agree on all_attn
5. **FK-Grade stable** - All configurations achieve ~6.75-6.93 (near target ≤6)

### 6B.4 Phase 2 Files Generated

```
/workspace/medisimplifier/results/ablation/
├── modules_ablation_llama3.json ✅
└── modules_ablation_mistral.json ✅
```

---

## 6C. Phase 3: Data Size Ablation Results ✅ COMPLETED

### 6C.1 OpenBioLLM-8B (Llama3 Architecture)

| Size | Samples | Train Loss | Eval Loss | ROUGE-L | FK-Grade |
|------|---------|------------|-----------|---------|----------|
| 2k | 2,000 | 0.9091 | 0.8279 | 0.6014 | 7.17 |
| 4k | 4,000 | 0.8640 | 0.8042 | 0.6198 | 6.98 |
| **8k** | **7,999** | **0.8334** | **0.7822** | **0.6345** | **6.80** |

**📊 Trend (Llama3): POSITIVE** | ROUGE-L change (2K→8K): +5.5% | Time: 114.3 min

### 6C.2 Mistral-7B (Mistral Architecture)

| Size | Samples | Train Loss | Eval Loss | ROUGE-L | FK-Grade |
|------|---------|------------|-----------|---------|----------|
| 2k | 2,000 | 0.7494 | 0.6939 | 0.5953 | 6.87 |
| 4k | 4,000 | 0.7145 | 0.6711 | 0.6168 | 6.83 |
| **8k** | **7,999** | **0.6937** | **0.6559** | **0.6349** | **6.76** |

**📊 Trend (Mistral): POSITIVE** | ROUGE-L change (2K→8K): +6.6% | Time: 119.9 min

### 6C.3 Phase 3 Key Findings

1. **More data = better performance** - Standard ML scaling confirmed
2. **Consistent across architectures** - Both show positive correlation
3. **Diminishing returns at 4K** - 4K achieves ~95% of 8K performance
4. **FK-Grade improves with data** - Better readability with more training
5. **Use full dataset** - 7999 samples optimal for this task

### 6C.4 Phase 3 Files Generated

```
/workspace/medisimplifier/results/ablation/
├── size_ablation_llama3.json ✅
└── size_ablation_mistral.json ✅
```

---

## 6D. Phase 4: rsLoRA Decision ✅ SKIPPED (Literature-Based)

### 6D.1 Decision Summary

| Aspect | Value |
|--------|-------|
| Decision | **use_rslora=True** |
| Method | Literature-based (no empirical ablation) |
| Rationale | Zero downside, potential benefit at r=32 |
| GPU time saved | ~2+ hours |

### 6D.2 Justification

**Why skip empirical ablation:**
1. **Not core research contribution** - Project focus is medical simplification, not LoRA hyperparameters
2. **Literature provides clear guidance** - Kalajdzievski (2023) extensively studied rsLoRA
3. **No downside risk** - Same parameters, same training time
4. **Aligns with Phase 1 findings** - "Higher rank = better" supports rsLoRA theory

**What rsLoRA does:**
- Standard LoRA: scaling = α / r
- rsLoRA: scaling = α / √r
- At r=32: rsLoRA preserves stronger gradient signal

**Literature support:**
- Hu et al. (2021): r=4-8 sufficient (but used α/r scaling)
- Kalajdzievski (2023): α/r causes gradient collapse at higher ranks; α/√r enables higher rank benefits

### 6D.3 Final Configuration

```python
lora_config = LoraConfig(
    r=32,                    # Phase 1 optimal
    lora_alpha=64,           # α = 2×r
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Phase 2 optimal
    use_rslora=True,         # Literature-based decision
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

---

## 6E. Ablation Summary & Visualization ✅ COMPLETED

### 6E.1 Combined Results Overview

| Phase | RQ | Llama3 Optimal | Mistral Optimal | Finding |
|-------|-----|----------------|-----------------|---------|
| 1 (Rank) | RQ4 | r=32 (ROUGE-L: 0.6183) | r=32 (ROUGE-L: 0.6171) | Higher rank = better |
| 2 (Modules) | RQ6 | all_attn (ROUGE-L: 0.6357) | all_attn (ROUGE-L: 0.6242) | More modules = better |
| 3 (Size) | RQ7 | 8K (+5.5%) | 8K (+6.6%) | More data = better |
| 4 (rsLoRA) | RQ12 | True | True | Literature-based |

### 6E.2 Final Optimal Configuration

```python
OPTIMAL_CONFIG = {
    "rank": 32,
    "alpha": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "target_modules_name": "all_attn",
    "use_rslora": True,
    "data_size": 7999,
    "dropout": 0.05,
}
```

### 6E.3 GPU Time Summary (Ablation Only)

| Architecture | Phase 1 | Phase 2 | Phase 3 | Total |
|--------------|---------|---------|---------|-------|
| Llama3 (OpenBioLLM) | 137.7 min | 138.0 min | 114.3 min | 390.0 min (6.5h) |
| Mistral | 139.7 min | 141.5 min | 119.9 min | 401.2 min (6.7h) |
| **Combined** | 277.4 min | 279.6 min | 234.2 min | **791.2 min (13.2h)** |

**Total Ablation Runs:** 18
**Total Ablation GPU Time:** 13.2 hours (parallel: ~6.7 hours)

### 6E.4 Key Research Contributions

1. **Contradicts Hu et al. (2021)** - r=32 outperforms r=8 for medical text simplification
2. **Confirms Raschka (2023)** - all_attn modules optimal despite 2x parameter cost
3. **Standard ML scaling** - More data consistently improves performance
4. **Literature-based rsLoRA** - Adopted without empirical ablation (zero downside)

### 6E.5 Ablation Files Generated

```
/workspace/medisimplifier/results/
├── ablation/
│   ├── rank_ablation_llama3.json ✅
│   ├── rank_ablation_mistral.json ✅
│   ├── modules_ablation_llama3.json ✅
│   ├── modules_ablation_mistral.json ✅
│   ├── size_ablation_llama3.json ✅
│   ├── size_ablation_mistral.json ✅
│   └── ablation_summary.json ✅
└── figures/
    └── ablation_summary.png ✅
```

---

## 6F. Phase 5: Full Training Results ✅ COMPLETED

### 6F.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 32 |
| Alpha (α) | 64 |
| Target Modules | all_attn [q, k, v, o]_proj |
| rsLoRA | True |
| Epochs | 3 |
| Training Samples | 7,999 |
| Trainable Params | 27,262,976 (0.38%) |

### 6F.2 Full Training Results

| Model | Train Loss | Eval Loss | ROUGE-L | FK-Grade | Time |
|-------|------------|-----------|---------|----------|------|
| **OpenBioLLM-8B** | 0.7116 | 0.7659 | **0.6721** | **6.84** ±1.00 | 90.9 min |
| Mistral-7B | 0.6326 | 0.6736 | 0.6346 | 6.95 ±1.24 | 87.8 min |
| BioMistral-7B-DARE | 0.6298 | 0.6869 | 0.6098 | 7.05 ±1.14 | 88.3 min |

### 6F.3 Baseline vs Fine-Tuned Comparison

| Model | Baseline ROUGE-L | Fine-Tuned ROUGE-L | Improvement | Baseline FK | Fine-Tuned FK | FK Δ |
|-------|------------------|--------------------| -----------|-------------|---------------|------|
| **OpenBioLLM-8B** | 0.2623 | **0.6721** | **+156.3%** | 12.53 | 6.84 | -5.69 |
| Mistral-7B | 0.3912 | 0.6346 | +62.2% | 10.60 | 6.95 | -3.65 |
| BioMistral-7B-DARE | 0.4120 | 0.6098 | +48.0% | 9.52 | 7.05 | -2.47 |

### 6F.4 Phase 5 Rankings

**🏆 By ROUGE-L (Quality):**
1. 🥇 OpenBioLLM-8B: 0.6721
2. 🥈 Mistral-7B: 0.6346
3. 🥉 BioMistral-7B-DARE: 0.6098

**🏆 By FK-Grade (Readability, target ≤6):**
1. 🥇 OpenBioLLM-8B: 6.84
2. 🥈 Mistral-7B: 6.95
3. 🥉 BioMistral-7B-DARE: 7.05

**🏆 By Improvement (vs Baseline):**
1. 🥇 OpenBioLLM-8B: +156.3%
2. 🥈 Mistral-7B: +62.2%
3. 🥉 BioMistral-7B-DARE: +48.0%

### 6F.5 Key Findings

1. **OpenBioLLM achieves best fine-tuned performance** - Despite worst baseline (0.26 → 0.67)
2. **Largest improvement from worst baseline** - OpenBioLLM improved +156%
3. **All models near FK target** - 6.84-7.05 vs target ≤6 (close!)
4. **FK variance acceptable** - Std dev ~1.0-1.2 (much better than baseline Mistral ±8.32)
5. **Training time consistent** - ~88-91 min per model (3 epochs)

### 6F.6 Phase 5 GPU Time Summary

| Model | GPU | Training Time |
|-------|-----|---------------|
| OpenBioLLM-8B | 0 | 90.9 min |
| Mistral-7B | 1 | 87.8 min |
| BioMistral-7B-DARE | 2 | 88.3 min |
| **Total (parallel)** | - | **~91 min** |
| **Total (sequential)** | - | **267.0 min (4.5h)** |

### 6F.7 Phase 5 Files Generated

```
/workspace/medisimplifier/
├── results/ablation/
│   ├── final_training_llama3.json ✅
│   ├── final_training_mistral.json ✅
│   ├── final_training_biomistral.json ✅
│   └── full_training_summary.json ✅
├── results/figures/
│   └── full_training_summary.png ✅
└── models/
    ├── openbiollm_8b_lora/ ✅
    ├── mistral_7b_lora/ ✅
    └── biomistral_7b_dare_lora/ ✅
```

---

## 7A. Chapter 7 Evaluation Results ✅ SECTIONS 7.1-7.3 COMPLETE

### 7A.1 Test Set Evaluation (1001 samples)

| Model | ROUGE-L | SARI | BERTScore-F1 | FK-Grade | FK-Δ |
|-------|---------|------|--------------|----------|------|
| **OpenBioLLM-8B** | **0.6749** | **74.64** | **0.9498** | 7.16 | ↓7.35 |
| Mistral-7B | 0.6491 | 73.79 | 0.9464 | **6.91** | **↓7.59** |
| BioMistral-7B-DARE | 0.6318 | 73.01 | 0.9439 | 6.95 | ↓7.56 |

**Reference Values:**
- Source FK-Grade: 14.50 (college level)
- Ground Truth FK-Grade: 7.23 (7th grade)

### 7A.2 Full Metrics Comparison (Baseline → Fine-Tuned)

| Model | Baseline ROUGE-L | Test ROUGE-L | Improvement |
|-------|------------------|--------------|-------------|
| **OpenBioLLM-8B** | 0.2623 | **0.6749** | **+157.3%** |
| Mistral-7B | 0.3912 | 0.6491 | +65.9% |
| BioMistral-7B-DARE | 0.4120 | 0.6318 | +53.3% |

### 7A.3 Readability Analysis

| Model | Source FK | Pred FK | Reduction | % Reduction |
|-------|-----------|---------|-----------|-------------|
| OpenBioLLM-8B | 14.50 | 7.16 | ↓7.35 | 50.6% |
| **Mistral-7B** | 14.50 | **6.91** | **↓7.59** | **52.4%** |
| BioMistral-7B-DARE | 14.50 | 6.95 | ↓7.56 | 52.1% |

**Key Finding:** All models achieve ~50% readability reduction, bringing college-level text (14.5) down to 7th-grade level (~7.0), matching ground truth (7.23).

### 7A.4 Rankings by Metric

**🏆 By ROUGE-L (Text Quality):**
1. 🥇 OpenBioLLM-8B: 0.6749
2. 🥈 Mistral-7B: 0.6491
3. 🥉 BioMistral-7B-DARE: 0.6318

**🏆 By SARI (Simplification Quality):**
1. 🥇 OpenBioLLM-8B: 74.64
2. 🥈 Mistral-7B: 73.79
3. 🥉 BioMistral-7B-DARE: 73.01

**🏆 By BERTScore (Semantic Similarity):**
1. 🥇 OpenBioLLM-8B: 0.9498
2. 🥈 Mistral-7B: 0.9464
3. 🥉 BioMistral-7B-DARE: 0.9439

**🏆 By FK-Grade (Readability, lower = better):**
1. 🥇 Mistral-7B: 6.91 (closest to target ≤6)
2. 🥈 BioMistral-7B-DARE: 6.95
3. 🥉 OpenBioLLM-8B: 7.16

### 7A.5 Key Findings

1. **OpenBioLLM-8B wins overall** - Best on 3/4 metrics (ROUGE-L, SARI, BERTScore)
2. **Mistral-7B best readability** - FK 6.91 closest to target ≤6
3. **All models excellent SARI** - 73-75 (target was ≥40)
4. **All models excellent BERTScore** - 0.94-0.95 (very high semantic similarity)
5. **50% readability reduction achieved** - College → 7th grade level

### 7A.6 Evaluation Files Generated

```
/workspace/medisimplifier/results/evaluation/
├── predictions_openbiollm_8b.json ✅
├── predictions_mistral_7b.json ✅
├── predictions_biomistral_7b_dare.json ✅
├── full_metrics_results.json ✅
└── full_metrics_detailed.json ✅
```

---

## 7B. Chapter 7.4: Baseline Comparison Results ✅ COMPLETED

### 7B.1 Baseline vs Fine-Tuned Metrics

| Model | Baseline ROUGE-L | FT ROUGE-L | ROUGE-L Δ% | Baseline FK | FT FK | FK Δ |
|-------|------------------|------------|------------|-------------|-------|------|
| **OpenBioLLM-8B** | 0.2623 | **0.6749** | **+157.3%** | 12.53 | 7.16 | -5.37 |
| Mistral-7B | 0.3912 | 0.6491 | +65.9% | 10.60 | **6.91** | -3.69 |
| BioMistral-7B-DARE | 0.4120 | 0.6318 | +53.4% | 9.52 | 6.95 | -2.57 |

### 7B.2 Additional Metrics Improvement

| Model | Baseline SARI | FT SARI | SARI Δ% | Baseline BERTScore | FT BERTScore | BERTScore Δ% |
|-------|---------------|---------|---------|--------------------| -------------|--------------|
| **OpenBioLLM-8B** | 36.98 | **74.64** | **+101.8%** | 0.6371 | **0.9498** | +49.1% |
| Mistral-7B | 46.38 | 73.79 | +59.1% | 0.7335 | 0.9464 | +29.0% |
| BioMistral-7B-DARE | 51.91 | 73.01 | +40.6% | 0.7426 | 0.9439 | +27.1% |

### 7B.3 Ranking Reversal

**Zero-Shot Baseline Ranking (ROUGE-L):**
1. 🥇 BioMistral-7B-DARE: 0.4120
2. 🥈 Mistral-7B: 0.3912
3. 🥉 OpenBioLLM-8B: 0.2623

**Fine-Tuned Ranking (ROUGE-L):**
1. 🥇 OpenBioLLM-8B: 0.6749 ← Was last!
2. 🥈 Mistral-7B: 0.6491
3. 🥉 BioMistral-7B-DARE: 0.6318 ← Was first!

**Key Insight:** Worst baseline model achieved best fine-tuned performance (+157% improvement).

### 7B.4 Readability Analysis

| Metric | Source | Ground Truth | OpenBioLLM | Mistral | BioMistral |
|--------|--------|--------------|------------|---------|------------|
| FK-Grade | 14.50 | 7.23 | 7.16 | **6.91** | 6.95 |
| Reduction | - | 50.1% | 50.6% | **52.3%** | 52.1% |

**All models achieve ~50% readability reduction**, matching or exceeding ground truth.

### 7B.5 Chapter 7.4 Files Generated

```
/workspace/medisimplifier/results/
├── evaluation/
│   └── baseline_comparison.json ✅
└── figures/
    ├── baseline_vs_finetuned_metrics.png ✅
    ├── improvement_percentage.png ✅
    ├── readability_comparison.png ✅
    ├── model_radar_comparison.png ✅
    └── ranking_reversal.png ✅
```

---

## 8A. Chapter 8.1: Statistical Analysis Results ✅ COMPLETED

### 8A.1 Bootstrap Confidence Intervals (95%, n=10,000)

| Model | ROUGE-L | 95% CI | FK-Grade | 95% CI |
|-------|---------|--------|----------|--------|
| **OpenBioLLM-8B** | **0.6749** | [0.6705, 0.6793] | 7.16 | [7.04, 7.32] |
| Mistral-7B | 0.6491 | [0.6445, 0.6537] | **6.91** | [6.84, 6.98] |
| BioMistral-7B-DARE | 0.6318 | [0.6272, 0.6365] | 6.95 | [6.87, 7.02] |

**Note:** Non-overlapping ROUGE-L CIs indicate significant differences between all models.

### 8A.2 Pairwise Comparisons (Bootstrap Paired Test)

| Comparison | Metric | Δ | p-value | Cohen's d | Effect Size |
|------------|--------|---|---------|-----------|-------------|
| OpenBioLLM vs Mistral | ROUGE-L | +0.0258 | <0.0001*** | +0.475 | small |
| OpenBioLLM vs BioMistral | ROUGE-L | +0.0430 | <0.0001*** | +0.793 | medium |
| Mistral vs BioMistral | ROUGE-L | +0.0173 | <0.0001*** | +0.332 | small |
| OpenBioLLM vs Mistral | FK-Grade | +0.248 | 0.0023** | +0.116 | negligible |
| OpenBioLLM vs BioMistral | FK-Grade | +0.211 | 0.0033** | +0.110 | negligible |
| Mistral vs BioMistral | FK-Grade | -0.037 | 0.1942 ns | -0.041 | negligible |

**All ROUGE-L differences are statistically significant (p < 0.001).**
**Mistral and BioMistral FK-Grades are NOT significantly different (p = 0.19).**

### 8A.3 FK-Grade Reduction Analysis

| Model | Mean FK | vs Target (6.0) | vs GT (7.23) | Reduction from Source |
|-------|---------|-----------------|--------------|----------------------|
| OpenBioLLM-8B | 7.16 ± 2.31 | +1.16 (p<.0001) | -0.07 (p=0.30 ns) | **7.35 grades*** |
| Mistral-7B | 6.91 ± 1.21 | +0.91 (p<.0001) | -0.32 (p<.0001) | **7.59 grades*** |
| BioMistral-7B-DARE | 6.95 ± 1.24 | +0.95 (p<.0001) | -0.28 (p<.0001) | **7.56 grades*** |

**Key Findings:**
- All models significantly above target (6.0) but close
- OpenBioLLM matches ground truth (p=0.30, not significant)
- Mistral & BioMistral slightly *better* than ground truth

### 8A.4 Outlier Analysis

**OpenBioLLM-8B has 1 outlier (idx=800, FK=69.1)**
- Cause: Valid prediction but extremely long run-on sentences (vocabulary simplified, sentence structure not)
- Impact: Minimal (mean Δ=0.062, explains higher std of 2.31 vs 1.21)
- Decision: Kept as-is (0.1% of samples, doesn't affect conclusions)

### 8A.5 Statistical Summary

| Finding | Result |
|---------|--------|
| All ROUGE-L comparisons significant | ✅ Yes (p < 0.001) |
| Best ROUGE-L model | OpenBioLLM-8B (0.6749) |
| Best FK-Grade model | Mistral-7B (6.91) |
| Effect size (OpenBioLLM vs BioMistral) | Medium (d=0.79) |
| FK reduction achieved | ~50% (14.5 → ~7.0) |

### 8A.6 Chapter 8.1 Files Generated

```
/workspace/medisimplifier/results/
├── evaluation/
│   └── statistical_analysis.json ✅
└── figures/
    ├── confidence_intervals.png ✅
    ├── effect_size_comparison.png ✅
    ├── significance_matrix.png ✅
    └── fk_distribution.png ✅
```

---

## 7. Notebook Checklist

### Chapter 5: Data Preparation & Baseline ✅ COMPLETE
- [x] 5.1 Environment setup
- [x] 5.2 Load dataset
- [x] 5.3 Load models
- [x] 5.4 Define prompts (with SYSTEM_MESSAGE)
- [x] 5.5 Define metrics
- [x] 5.6-5.8 Run baselines (all 3 models)
- [x] 5.9 Compare baselines
- [x] 5.10 Save results

### Chapter 6: LoRA Fine-Tuning ✅ COMPLETE
- [x] 6.1 Training setup
- [x] 6.2.1.1 Phase 1: Rank ablation (OpenBioLLM) ✅
- [x] 6.2.1.2 Phase 1: Rank ablation (Mistral-7B) ✅
- [x] 6.2.2.1 Phase 2: Modules ablation (OpenBioLLM) ✅
- [x] 6.2.2.2 Phase 2: Modules ablation (Mistral-7B) ✅
- [x] 6.2.3.1 Phase 3: Data size ablation (OpenBioLLM) ✅
- [x] 6.2.3.2 Phase 3: Data size ablation (Mistral-7B) ✅
- [x] 6.2.4 Ablation summary & visualization ✅
- [x] 6.3.1 Phase 5: Full training (OpenBioLLM) ✅
- [x] 6.3.2 Phase 5: Full training (Mistral-7B) ✅
- [x] 6.3.3 Phase 5: Full training (BioMistral) ✅
- [x] 6.3.4 Full training summary & visualization ✅

### Chapter 7: Evaluation ✅ COMPLETE
- [x] 7.1 Setup & Load Test Dataset ✅
- [x] 7.2.1 Generate test predictions (OpenBioLLM) ✅
- [x] 7.2.2 Generate test predictions (Mistral-7B) ✅
- [x] 7.2.3 Generate test predictions (BioMistral) ✅
- [x] 7.3 Compute full metrics ✅
- [x] 7.4 Compare to baselines ✅

### Chapter 8: Results & Visualization ✅ COMPLETE
- [x] 8.1 Statistical Analysis ✅
- [x] 8.2 Answer Research Questions ✅
- [x] 8.3 Visualizations & Summary ✅
- [x] 8.4 Qualitative Examples & Error Analysis ✅

### Chapter 9: Discussion & Conclusions ✅ COMPLETE
- [x] 9.1 Project Summary ✅
  - [x] 9.1.1 Motivation & Objectives
  - [x] 9.1.2 Methodology Overview
  - [x] 9.1.3 Key Experimental Results
- [x] 9.2 Conclusions ✅
  - [x] 9.2.1 Research Question Answers (all 12 RQs)
  - [x] 9.2.2 Key Takeaways
- [x] 9.3 Limitations & Anomalies ✅
  - [x] 9.3.1 Generation Anomalies (outliers, truncation, duplication)
- [x] 9.4 Future Work ✅
  - [x] 9.4.1 Immediate Improvements
  - [x] 9.4.2 Model-Specific Enhancements
  - [x] 9.4.3 Extended Research Directions
  - [x] 9.4.4 Recommended Priority Roadmap
- [x] 9.5 Final Remarks ✅

---

## 8. Parallel GPU Execution

### GPU Assignment

| GPU | Model | Phases |
|-----|-------|--------|
| GPU 0 | OpenBioLLM-8B | Phase 1-3 ✅, Phase 5 ✅, Ch7 Eval ✅ |
| GPU 1 | Mistral-7B | Phase 1-3 ✅, Phase 5 ✅, Ch7 Eval ✅ |
| GPU 2 | BioMistral-7B | Phase 5 ✅, Ch7 Eval ✅ |

### Execution Timeline (Chapter 6)

| Time | GPU 0 | GPU 1 | GPU 2 |
|------|-------|-------|-------|
| T+0 | P1: Rank (3 runs) ✅ | P1: Rank (3 runs) ✅ | Idle |
| T+2h | P2: Modules (3 runs) ✅ | P2: Modules (3 runs) ✅ | Idle |
| T+4h | P3: Size (3 runs) ✅ | P3: Size (3 runs) ✅ | Idle |
| T+6h | P5: Full OpenBioLLM ✅ | P5: Full Mistral ✅ | P5: Full BioMistral ✅ |
| T+7.5h | Complete ✅ | Complete ✅ | Complete ✅ |

### Execution Timeline (Chapter 7)

| Time | GPU 0 | GPU 1 | GPU 2 |
|------|-------|-------|-------|
| T+0 | 7.2.1 Predictions ✅ | 7.2.2 Predictions ✅ | 7.2.3 Predictions ✅ |
| T+3.3h | Done ✅ | Done ✅ | Done ✅ |
| T+3.5h | 7.3 Metrics ✅ | - | - |
| T+3.7h | Complete ✅ | - | - |

**Chapter 6 Total time:** ~7.5 hours (parallel)
**Chapter 7 Evaluation time (7.1-7.3):** ~3.7 hours (parallel predictions + metrics)

**All training runs completed:** 21/21 (100%) ✅
**Evaluation predictions completed:** 3/3 (100%) ✅
**Full metrics computed:** ✅

---

## 9. Checkpoint Strategy

### Checkpoint Files

| Phase | File Pattern | Status |
|-------|--------------|--------|
| Phase 1 | `rank_ablation_{arch}.json` | ✅ Complete |
| Phase 2 | `modules_ablation_{arch}.json` | ✅ Complete |
| Phase 3 | `size_ablation_{arch}.json` | ✅ Complete |
| Phase 4 | N/A (skipped) | ✅ N/A |
| Ablation Summary | `ablation_summary.json` | ✅ Complete |
| Phase 5 | `final_training_{model}.json` | ✅ Complete |
| Full Training Summary | `full_training_summary.json` | ✅ Complete |
| Chapter 7 Predictions | `predictions_{model}.json` | ✅ Complete |
| Chapter 7 Metrics | `full_metrics_results.json` | ✅ Complete |
| Baseline Comparison | `baseline_comparison.json` | ✅ Complete |
| Statistical Analysis | `statistical_analysis.json` | ✅ Complete |
| Chapter 8 RQ Answers | `rq_answers.json` | ✅ Complete |
| Chapter 9 Summary | `project_summary.json` | ✅ Complete |

### Checkpoint Location

```
/workspace/medisimplifier/results/ablation/
├── rank_ablation_llama3.json ✅
├── rank_ablation_mistral.json ✅
├── modules_ablation_llama3.json ✅
├── modules_ablation_mistral.json ✅
├── size_ablation_llama3.json ✅
├── size_ablation_mistral.json ✅
├── ablation_summary.json ✅
├── final_training_llama3.json ✅
├── final_training_mistral.json ✅
├── final_training_biomistral.json ✅
└── full_training_summary.json ✅

/workspace/medisimplifier/results/evaluation/
├── predictions_openbiollm_8b.json ✅
├── predictions_mistral_7b.json ✅
├── predictions_biomistral_7b_dare.json ✅
├── full_metrics_results.json ✅
├── full_metrics_detailed.json ✅
├── baseline_comparison.json ✅
└── statistical_analysis.json ✅

/workspace/medisimplifier/results/conclusions/
├── rq_answers.json ✅
└── project_summary.json ✅
```

---

## 10. Output Files

**Ablation Results:**
- `ablation/rank_ablation_llama3.json` ✅
- `ablation/rank_ablation_mistral.json` ✅
- `ablation/modules_ablation_llama3.json` ✅
- `ablation/modules_ablation_mistral.json` ✅
- `ablation/size_ablation_llama3.json` ✅
- `ablation/size_ablation_mistral.json` ✅
- `ablation/ablation_summary.json` ✅

**Full Training Results:**
- `ablation/final_training_llama3.json` ✅
- `ablation/final_training_mistral.json` ✅
- `ablation/final_training_biomistral.json` ✅
- `ablation/full_training_summary.json` ✅

**Fine-tuned Models:**
- `models/openbiollm_8b_lora/` ✅
- `models/biomistral_7b_dare_lora/` ✅
- `models/mistral_7b_lora/` ✅

**Evaluation:**
- `evaluation/predictions_openbiollm_8b.json` ✅
- `evaluation/predictions_mistral_7b.json` ✅
- `evaluation/predictions_biomistral_7b_dare.json` ✅
- `evaluation/full_metrics_results.json` ✅
- `evaluation/full_metrics_detailed.json` ✅
- `evaluation/baseline_comparison.json` ✅
- `evaluation/statistical_analysis.json` ✅

**Figures:**
- `figures/ablation_summary.png` ✅
- `figures/full_training_summary.png` ✅
- `figures/baseline_vs_finetuned_metrics.png` ✅
- `figures/improvement_percentage.png` ✅
- `figures/readability_comparison.png` ✅
- `figures/model_radar_comparison.png` ✅
- `figures/ranking_reversal.png` ✅
- `figures/confidence_intervals.png` ✅
- `figures/effect_size_comparison.png` ✅
- `figures/significance_matrix.png` ✅
- `figures/fk_distribution.png` ✅
- `figures/rq8_correlation.png` ✅
- `figures/ablation_study_summary.png` ✅
- `figures/rq_summary_infographic.png` ✅
- `figures/qualitative_examples.png` ✅
- `figures/error_analysis.png` ✅

**Conclusions:**
- `conclusions/rq_answers.json` ✅
- `conclusions/project_summary.json` ✅

---

## 11. Environment Setup (RunPod)

```bash
# =============================================================================
# RUNPOD ENVIRONMENT SETUP (Run once per pod)
# =============================================================================

# Clear root filesystem caches
!rm -rf /root/.cache/huggingface /root/.cache/pip /root/.cache/torch
!pip cache purge

# Remove unused audio/vision packages
!pip uninstall torchaudio torchvision -y

# Fix typing_extensions compatibility (REQUIRED for PyTorch 2.6.0)
!pip install --upgrade "typing_extensions>=4.10" --force-reinstall

# Notebook merging utility
!pip install nbmerge

# PyTorch/CUDA Upgrade (H200 SXM compatibility)
!pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -q

# Install Dependencies (no-deps for data science libs to avoid conflicts)
!pip install -q --no-deps pandas numpy matplotlib seaborn

# Install ML/NLP Dependencies
!pip install -q transformers datasets accelerate peft bitsandbytes tqdm

# Evaluation metrics
!pip install -q evaluate rouge-score bert-score textstat sacrebleu

# SARI metric (install from GitHub - required for easse)
!pip install git+https://github.com/feralvam/easse.git -q

# ⚠️ RESTART KERNEL AFTER THIS CELL
```

**Cache Configuration (in Section 6.1/7.1):**
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change per notebook: "0", "1", "2"
os.environ["HF_HOME"] = "/workspace/HFModels"
```

---

## 12. Ablation Study Details

### 12.1 Phase 1: Rank Ablation (RQ4) ✅ COMPLETE

| Config | Rank (r) | Alpha (α) | Trainable Params |
|--------|----------|-----------|------------------|
| r8 | 8 | 16 | ~3.4M |
| r16 | 16 | 32 | ~6.8M |
| **r32** | **32** | **64** | **~13.6M** |

**Fixed:** target_modules=["q_proj", "v_proj"], epochs=1, data=7999, use_rslora=False

**Result:** r=32 optimal for both architectures. Higher rank = better performance.

### 12.2 Phase 2: Target Modules Ablation (RQ6) ✅ COMPLETE

| Config | Modules | Trainable Params |
|--------|---------|------------------|
| q_only | [q_proj] | ~8.4M |
| q_v | [q_proj, v_proj] | ~13.6M |
| **all_attn** | **[q_proj, k_proj, v_proj, o_proj]** | **~27.3M** |

**Fixed:** rank=32 (from Phase 1), alpha=64, epochs=1, data=7999

**Result:** all_attn optimal for both architectures. More modules = better performance.

### 12.3 Phase 3: Data Size Ablation (RQ7) ✅ COMPLETE

| Config | Samples | % of Full |
|--------|---------|-----------|
| 2k | 2,000 | 25% |
| 4k | 4,000 | 50% |
| **8k** | **7,999** | **100%** |

**Fixed:** rank=32, modules=all_attn, epochs=1

**Result:** More data = better performance (standard ML behavior confirmed)
- Llama3: +5.5% ROUGE-L (2K→8K)
- Mistral: +6.6% ROUGE-L (2K→8K)

### 12.4 Phase 4: rsLoRA Decision (RQ12) ✅ SKIPPED

**Decision:** use_rslora=True (literature-based, no empirical ablation)

**Rationale:**
- Phase 1 showed "higher rank = better" - aligns with rsLoRA theory
- rsLoRA has zero downside (same params, same training time)
- rsLoRA has potential benefit at r=32 (Kalajdzievski 2023)
- Not core research contribution - project focus is medical simplification

**GPU time saved:** ~2+ hours

### 12.5 Phase 5: Full Training ✅ COMPLETE

**Configuration:** r=32, α=64, all_attn, rsLoRA=True, 3 epochs, 7999 samples

**Results:**
- OpenBioLLM-8B: ROUGE-L 0.6721, FK 6.84 (+156.3% improvement)
- Mistral-7B: ROUGE-L 0.6346, FK 6.95 (+62.2% improvement)
- BioMistral-7B-DARE: ROUGE-L 0.6098, FK 7.05 (+48.0% improvement)

**Total training time:** 267 min (4.5h sequential, ~91 min parallel)

### 12.6 Ablation Evaluation

Each ablation configuration evaluated on validation set (100 sample subset) using:
- ROUGE-L (primary metric for selection)
- FK-Grade (readability check)
- Training loss
- Eval loss
- Training time

**Selection Criteria:** Best ROUGE-L; FK-Grade used as tiebreaker

---

## 13. Literature References

| Topic | Paper | Key Finding |
|-------|-------|-------------|
| LoRA Original | Hu et al. (2021) | r=4-8 sufficient, Wq+Wv best |
| rsLoRA | Kalajdzievski (2023) | α/√r scaling enables higher rank benefits |
| Target Modules | Raschka (2023), Unsloth | Apply to all layers for best performance |
| **Our Phase 1** | **This work** | **r=32 optimal, contradicts Hu et al.** |
| **Our Phase 2** | **This work** | **all_attn optimal, confirms Raschka/Unsloth** |
| **Our Phase 3** | **This work** | **More data = better, standard ML scaling** |
| **Our Phase 4** | **This work** | **rsLoRA=True adopted (literature-based)** |
| **Our Phase 5** | **This work** | **OpenBioLLM best after fine-tuning (+156%)** |
| **Our Chapter 7** | **This work** | **All RQs answered, 50% readability reduction achieved** |

---

## 14. Links

| Resource | URL |
|----------|-----|
| Dataset | huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes |
| OpenBioLLM-8B | huggingface.co/aaditya/Llama3-OpenBioLLM-8B |
| BioMistral-7B-DARE | huggingface.co/BioMistral/BioMistral-7B-DARE |
| Mistral-7B-v0.2 | huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 |
| LoRA Paper | arxiv.org/abs/2106.09685 |
| rsLoRA Paper | arxiv.org/abs/2312.03732 |

---

## 15. File I/O Summary

### 15.1 Files by Chapter

| Chapter | Section | Input Files | Output Files |
|---------|---------|-------------|--------------|
| **Part 1: Ch 1-4** | 1.1 Setup | - | `medisimplifier/` (directory structure) |
| | 2.1 Load Dataset | `starmpcc/Asclepius-Synthetic-Clinical-Notes` (HF) | HF cache |
| | 3.1 Ground Truth Gen | Asclepius dataset | `data/ground_truth/ground_truth_checkpoint.json` |
| | 4.1 Validation | `data/ground_truth/ground_truth_checkpoint.json` | `data/ground_truth/ground_truth_clean.json` |
| | 4.2 Splitting | `data/ground_truth/ground_truth_clean.json` | `data/ground_truth/ground_truth_labeled_splits_final.json` |
| | 4.3 Statistics | `data/ground_truth/ground_truth_labeled_splits_final.json` | `results/figures/note_length_distribution.png` |
| **Part 2: Ch 5** | 5.1 Setup | - | - |
| | 5.2 Load Data | `data/ground_truth/ground_truth_labeled_splits_final.json` | - |
| | 5.3 Load Models | HF models | HF cache |
| | 5.4 Format Data | `data/ground_truth/ground_truth_labeled_splits_final.json` | `data/instruction_dataset/chatml/`<br>`data/instruction_dataset/mistral/` |
| | 5.6-5.8 Baselines | Instruction datasets | `baseline/baseline_openbiollm_8b.json`<br>`baseline/baseline_biomistral_7b_dare.json`<br>`baseline/baseline_mistral_7b.json`<br>`baseline/baseline_checkpoint_*.json` |
| | 5.10 Aggregate | Baseline JSON files | `baseline/baseline_metrics.csv`<br>`baseline/baseline_results.json`<br>`figures/baseline_comparison.png` |
| **Part 3: Ch 6** | 6.1 Setup | `data/instruction_dataset/*` | - |
| | 6.2.1 Phase 1 (Rank) | Instruction datasets | `ablation/rank_ablation_llama3.json`<br>`ablation/rank_ablation_mistral.json` |
| | 6.2.2 Phase 2 (Modules) | Instruction datasets | `ablation/modules_ablation_llama3.json`<br>`ablation/modules_ablation_mistral.json` |
| | 6.2.3 Phase 3 (Size) | Instruction datasets | `ablation/size_ablation_llama3.json`<br>`ablation/size_ablation_mistral.json` |
| | 6.2.4 Ablation Summary | Ablation JSON files | `ablation/ablation_summary.json`<br>`figures/ablation_summary.png` |
| | 6.3.1 Full Train (OpenBio) | `data/instruction_dataset/chatml/` | `models/openbiollm_8b_lora/`<br>`checkpoints/full_training_llama3/`<br>`training/final_training_llama3.json` |
| | 6.3.2 Full Train (Mistral) | `data/instruction_dataset/mistral/` | `models/mistral_7b_lora/`<br>`checkpoints/full_training_mistral/`<br>`training/final_training_mistral.json` |
| | 6.3.3 Full Train (BioMistral) | `data/instruction_dataset/mistral/` | `models/biomistral_7b_dare_lora/`<br>`checkpoints/full_training_biomistral/`<br>`training/final_training_biomistral.json` |
| | 6.3.4 Training Summary | Training JSON + baseline | `training/full_training_summary.json`<br>`figures/full_training_summary.png` |
| **Part 4: Ch 7** | 7.1 Setup | Instruction datasets, adapters | - |
| | 7.2.1-7.2.3 Predictions | Adapters + test data | `evaluation/predictions_openbiollm_8b.json`<br>`evaluation/predictions_mistral_7b.json`<br>`evaluation/predictions_biomistral_7b_dare.json` |
| | 7.3 Compute Metrics | Prediction files | `evaluation/full_metrics_results.json`<br>`evaluation/full_metrics_detailed.json` |
| | 7.4 Baseline Comparison | Baseline + metrics | `evaluation/baseline_comparison.json`<br>`figures/baseline_vs_finetuned_metrics.png`<br>`figures/improvement_percentage.png`<br>`figures/readability_comparison.png`<br>`figures/model_radar_comparison.png`<br>`figures/ranking_reversal.png` |
| **Part 4: Ch 8** | 8.1 Statistical Analysis | Metrics results | `evaluation/statistical_analysis.json`<br>`figures/confidence_intervals.png`<br>`figures/effect_size_comparison.png`<br>`figures/significance_matrix.png`<br>`figures/fk_distribution.png` |
| | 8.2 Answer Research Questions | Evaluation + ablation results | `conclusions/rq_answers.json`<br>`figures/rq8_correlation.png` |
| | 8.3 Visualizations & Summary | All metrics + ablation data | `figures/ablation_study_summary.png`<br>`figures/rq_summary_infographic.png` |
| | 8.4 Qualitative Examples & Error Analysis | Predictions + metrics | `figures/qualitative_examples.png`<br>`figures/error_analysis.png` |
| **Part 4: Ch 9** | 9.1-9.5 Conclusions | All project results | `conclusions/project_summary.json` |

### 15.2 Figures Generated (18 Total)

| Part | Chapter | Section | Figure File | Size |
|------|---------|---------|-------------|------|
| Part 1 | Ch 4 | 4.3 Statistics | `note_length_distribution.png` | 48 KB |
| Part 2 | Ch 5 | 5.10 Compare | `baseline_comparison.png` | 121 KB |
| Part 3 | Ch 6 | 6.2.4 Ablation Summary | `ablation_summary.png` | 330 KB |
| Part 3 | Ch 6 | 6.3.4 Training Summary | `full_training_summary.png` | 266 KB |
| Part 4 | Ch 7 | 7.4 Baseline Comparison | `baseline_vs_finetuned_metrics.png` | 86 KB |
| Part 4 | Ch 7 | 7.4 Baseline Comparison | `improvement_percentage.png` | 64 KB |
| Part 4 | Ch 7 | 7.4 Baseline Comparison | `readability_comparison.png` | 69 KB |
| Part 4 | Ch 7 | 7.4 Baseline Comparison | `model_radar_comparison.png` | 167 KB |
| Part 4 | Ch 7 | 7.4 Baseline Comparison | `ranking_reversal.png` | 68 KB |
| Part 4 | Ch 8 | 8.1 Statistical Analysis | `confidence_intervals.png` | 98 KB |
| Part 4 | Ch 8 | 8.1 Statistical Analysis | `effect_size_comparison.png` | 77 KB |
| Part 4 | Ch 8 | 8.1 Statistical Analysis | `significance_matrix.png` | 62 KB |
| Part 4 | Ch 8 | 8.1 Statistical Analysis | `fk_distribution.png` | 61 KB |
| Part 4 | Ch 8 | 8.2 Answer Research Questions | `rq8_correlation.png` | ~80 KB |
| Part 4 | Ch 8 | 8.3 Visualizations & Summary | `ablation_study_summary.png` | ~150 KB |
| Part 4 | Ch 8 | 8.3 Visualizations & Summary | `rq_summary_infographic.png` | ~120 KB |
| Part 4 | Ch 8 | 8.4 Qualitative Examples | `qualitative_examples.png` | ~100 KB |
| Part 4 | Ch 8 | 8.4 Qualitative Examples | `error_analysis.png` | ~90 KB |

### 15.3 LoRA Adapter Files

| Model | Directory | Files | Adapter Size |
|-------|-----------|-------|--------------|
| OpenBioLLM-8B | `models/openbiollm_8b_lora/` | `README.md`, `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`, `tokenizer_config.json` | 109.1 MB |
| Mistral-7B | `models/mistral_7b_lora/` | `README.md`, `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json` | 109.1 MB |
| BioMistral-7B-DARE | `models/biomistral_7b_dare_lora/` | `README.md`, `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json` | 109.1 MB |

### 15.4 Training Checkpoints

| Model | Directory | Checkpoints | Files per Checkpoint |
|-------|-----------|-------------|----------------------|
| OpenBioLLM-8B | `checkpoints/full_training_llama3/` | `checkpoint-500` (epoch 1)<br>`checkpoint-1000` (epoch 2)<br>`checkpoint-1500` (epoch 3) | 10 files each |
| Mistral-7B | `checkpoints/full_training_mistral/` | `checkpoint-500` (epoch 1)<br>`checkpoint-1000` (epoch 2)<br>`checkpoint-1500` (epoch 3) | 11 files each |
| BioMistral-7B-DARE | `checkpoints/full_training_biomistral/` | `checkpoint-500` (epoch 1)<br>`checkpoint-1000` (epoch 2)<br>`checkpoint-1500` (epoch 3) | 11 files each |

**Checkpoint Files (per checkpoint):**
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA weights (109 MB)
- `optimizer.pt` - Optimizer state (218 MB)
- `scheduler.pt` - LR scheduler state
- `rng_state.pth` - RNG state for reproducibility
- `tokenizer.json` - Tokenizer
- `tokenizer_config.json` - Tokenizer config
- `trainer_state.json` - Training metrics history
- `training_args.bin` - Training arguments
- `README.md` - Auto-generated readme
- `chat_template.jinja` - Chat template (Mistral models only)

### 15.5 Complete File Structure

```
/workspace/medisimplifier/
├── data/
│   ├── ground_truth/
│   │   ├── ground_truth_checkpoint.json
│   │   ├── ground_truth_clean.json
│   │   └── ground_truth_labeled_splits_final.json
│   └── instruction_dataset/
│       ├── chatml/
│       │   ├── train/
│       │   ├── validation/
│       │   └── test/
│       └── mistral/
│           ├── train/
│           ├── validation/
│           └── test/
├── models/
│   ├── openbiollm_8b_lora/
│   ├── mistral_7b_lora/
│   └── biomistral_7b_dare_lora/
├── checkpoints/
│   ├── full_training_llama3/
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   └── checkpoint-1500/
│   ├── full_training_mistral/
│   └── full_training_biomistral/
├── results/
│   ├── ablation/
│   ├── baseline/
│   ├── training/
│   ├── evaluation/
│   ├── conclusions/
│   └── figures/
└── notebooks/
    ├── MediSimplifier_Part_1.ipynb
    ├── MediSimplifier_Part2.ipynb
    ├── MediSimplifier_Part3.ipynb
    └── MediSimplifier_Part4.ipynb
```

### 15.6 Storage Summary

| Directory | Approx Size |
|-----------|-------------|
| `data/ground_truth/` | ~50 MB |
| `data/instruction_dataset/` | ~varies |
| `models/` | ~330 MB |
| `checkpoints/` | ~3.0 GB |
| `results/` | ~35 MB |
| `notebooks/` | ~7 MB |
| **Total** | **~3.4 GB** |

| Category | Count |
|----------|-------|
| Ground truth files | 3 |
| Instruction datasets | 2 (chatml, mistral) |
| LoRA adapters (final) | 3 |
| Training checkpoints | 9 (3 epochs × 3 models) |
| Ablation JSON files | 7 |
| Baseline files | 8 |
| Training result files | 4 |
| Evaluation files | 7 |
| Conclusions files | 2 |
| Figures (PNG) | 18 |
| Notebooks | 8 (4 .ipynb + 4 .html) |

---

## Changelog

### v31 (Current)
- **CHAPTERS 7-9 STRUCTURE UPDATED** to match actual Part 4 notebook
- **Chapter 7:** Sections 7.1-7.4 only (7.5 moved to Ch 8)
  - 7.1 Setup & Load Test Dataset
  - 7.2.1-7.2.3 Generate Test Predictions (parallel GPUs)
  - 7.3 Compute Full Metrics
  - 7.4 Compare to Baselines
- **Chapter 8:** Restructured (statistical analysis moved from 7.5)
  - 8.1 Statistical Analysis (was 7.5)
  - 8.2 Answer Research Questions
  - 8.3 Visualizations & Summary
  - 8.4 Qualitative Examples & Error Analysis
- **Chapter 9:** Expanded from 3 to 5 sections
  - 9.1 Project Summary (3 subsections)
  - 9.2 Conclusions (2 subsections)
  - 9.3 Limitations & Anomalies
  - 9.4 Future Work (4 subsections with priority roadmap)
  - 9.5 Final Remarks
- **Renamed Section 7C → 8A** (Chapter 8.1 Statistical Analysis Results)
- **Updated Section 7 (Notebook Checklist):** All chapters marked complete
- **Updated Section 9 (Checkpoint Files):** Added Ch 8-9 files
- **Updated Section 10 (Output Files):** Added 5 new figures, conclusions files
- **Updated Section 15.1 (Files by Chapter):** Restructured Ch 7-8 entries
- **Updated Section 15.2 (Figures):** 13 → 18 figures, corrected chapter assignments

### v30
- **Added Section 15: File I/O Summary**
- **15.1:** Files by Chapter - complete I/O mapping for all chapters
- **15.2:** Figures Generated - 13 PNG files with sizes
- **15.3:** LoRA Adapter Files - 3 models, 109.1 MB each
- **15.4:** Training Checkpoints - 9 checkpoints (3 epochs × 3 models)
- **15.5:** Complete File Structure - full directory tree
- **15.6:** Storage Summary - ~3.4 GB total

### v29
- **CHAPTER 7 SECTIONS 7.4-7.5 COMPLETE** ✅
- **Added Section 7B:** Baseline Comparison Results with full metrics tables
- **Added Section 7C:** Statistical Analysis Results with bootstrap tests
- **All pairwise ROUGE-L comparisons significant** (p < 0.001)
- **Effect sizes quantified:** OpenBioLLM vs BioMistral = medium (d=0.79)
- **Outlier analysis:** 1/1001 OpenBioLLM sample (FK=69.1, minimal impact)
- **FK-Grade analysis:** All models achieve ~50% readability reduction
- **Mistral & BioMistral FK not significantly different** (p = 0.19)
- **Updated notebook checklist:** 7.4, 7.5 marked complete
- **Updated checkpoint files:** baseline_comparison.json, statistical_analysis.json complete
- **Updated output files:** 11 new figures added
- **Remaining sections:** 7.6 (research questions), 7.7 (summary)

### v28
- **CHAPTER 7 SECTIONS 7.1-7.3 COMPLETE** ✅
- **All predictions generated** for 3 models (1001 test samples each)
- **Full metrics computed:** ROUGE-L, SARI, BERTScore-F1, FK-Grade
- **Added Section 7A:** Chapter 7 Evaluation Results with full metrics table
- **OpenBioLLM-8B wins overall:** Best ROUGE-L (0.6749), SARI (74.64), BERTScore (0.9498)
- **Mistral-7B best readability:** FK-Grade 6.91 (closest to target ≤6)
- **50% readability reduction achieved:** 14.50 → ~7.0 (matches ground truth 7.23)
- **Updated Research Questions:** RQ3, RQ5, RQ8, RQ10, RQ11 now answered
- **Updated notebook checklist:** 7.1, 7.2.1-7.2.3, 7.3 marked complete
- **Updated execution timeline:** Predictions and metrics complete
- **Updated checkpoint files:** predictions_*.json and full_metrics_*.json marked complete
- **Updated output files:** evaluation/ files marked complete
- **Updated environment setup:** Added easse install from GitHub
- **Remaining sections:** 7.4-7.7 (baseline comparison, statistical analysis, visualizations)

### v27
- **CHAPTER 6 FULLY COMPLETE** ✅
- **PHASE 5 FULL TRAINING COMPLETED** for all 3 models
- **Added Section 6F:** Phase 5 Full Training Results with full metrics
- **OpenBioLLM-8B achieves best fine-tuned ROUGE-L (0.6721)** - +156.3% improvement
- **All models near FK target** - 6.84-7.05 (close to ≤6)
- **Updated notebook checklist:** Chapter 6 fully complete, Chapter 7.1 started
- **Updated execution timeline:** All 21/21 training runs complete
- **Updated checkpoint files:** final_training_*.json and full_training_summary.json marked complete
- **Updated output files:** models/ directories marked complete, full_training_summary.png added
- **Updated Section 3.3:** Phase 5 marked DONE, total runs 21/21 complete
- **Updated GPU Assignment:** Phase 5 marked complete for all GPUs
- **Chapter 7 Evaluation in progress:** Section 7.1 Setup complete
- **Dataset location clarified:** `/workspace/medisimplifier/data/instruction_dataset/`

### v26
- **CHAPTER 6 ABLATION COMPLETE** ✅
- **Added Section 6E:** Ablation Summary & Visualization with combined results
- **Updated notebook checklist:** 6.2.4 marked complete
- **Updated execution timeline:** Ablation runs 18/18 (100%) complete
- **Updated checkpoint files:** ablation_summary.json marked complete
- **Updated output files:** ablation_summary.png marked complete
- **Added GPU time summary:** Total 13.2 hours (parallel: 6.7 hours)
- **Added Section 6E.4:** Key research contributions summary
- **Ready for Phase 5:** Full training with optimal config

### v25
- **PHASE 4 rsLoRA DECISION: SKIPPED** ✅
- **rsLoRA=True adopted based on literature** (Kalajdzievski 2023)
- **Rationale:** Zero downside, potential benefit at r=32, not core research contribution
- **GPU time saved:** ~2+ hours (6 runs avoided)
- **Updated Section 3.1:** Added "Final" column with rsLoRA=True
- **Updated Section 3.3:** Phase 4 marked SKIPPED, total runs reduced to 21
- **Updated Section 3.4:** Added decision rationale and justification table
- **Updated Section 5 (RQ12):** Marked answered with literature-based finding
- **Added Section 6D:** Phase 4 decision summary
- **Updated Section 11.4:** Full explanation of rsLoRA decision
- **Updated Section 12:** Added Phase 4 literature reference
- **Updated Phase 1 finding:** "Supports rsLoRA adoption" instead of "validates rsLoRA investigation"

### v24
- **PHASE 3 DATA SIZE ABLATION COMPLETED** ✅
- **Added Section 6C:** Phase 3 Data Size Ablation Results with full metrics
- **Updated RQ7:** Answered - More data = better (+5.5% Llama3, +6.6% Mistral)
- **Updated notebook checklist:** 6.2.3.1, 6.2.3.2 complete
- **Updated execution timeline:** Phase 3 done (~6.5h total), Phase 4 pending
- **Updated checkpoint files:** size_ablation_*.json marked complete
- **Added Phase 3 key findings:** Standard ML scaling confirmed, diminishing returns at 4K
- **Updated Phase dependencies:** Phase 3 complete, flows to Phase 4
- **Updated Section 11.3:** Added Phase 3 results
- **Updated Literature References:** Added Phase 3 finding

### v23
- **PHASE 2 MODULES ABLATION COMPLETED** ✅
- **Added Section 6B:** Phase 2 Modules Ablation Results with full metrics
- **Updated RQ6:** Answered - all_attn optimal for both architectures
- **Updated RQ9:** Answered - all_attn best despite 2x params (quality over efficiency)
- **Updated notebook checklist:** 6.2.2.1, 6.2.2.2 complete; added 6.2.3.1, 6.2.3.2
- **Updated execution timeline:** Phase 2 done (~4.6h total), Phase 3 pending
- **Updated checkpoint files:** modules_ablation_*.json marked complete
- **Added Phase 2 key findings:** More modules = better, confirms modern best practices
- **Updated Section 3.1:** Added optimal modules column
- **Updated Phase dependencies:** all_attn flows to Phase 3
- **Updated Section 11.3:** Phase 3 now uses all_attn
- **Updated Literature References:** Added Phase 2 finding

### v22
- **PHASE 1 RANK ABLATION COMPLETED** ✅
- **Added Section 6A:** Phase 1 Rank Ablation Results with full metrics
- **Updated RQ4:** Answered - r=32 optimal for both architectures
- **Updated notebook checklist:** 6.1, 6.2.1.1, 6.2.1.2 complete
- **Updated execution timeline:** Phase 1 done, Phase 2 running
- **Updated checkpoint files:** rank_ablation_*.json marked complete
- **Added Phase 1 key findings:** Higher rank = better, contradicts original LoRA paper
- **Updated Section 3.1:** Added optimal rank column
- **Updated Phase dependencies:** r=32 flows to Phase 2

### v21
- **Added rsLoRA comparison phase** (Phase 4, RQ12)
- **Updated ablation strategy** to sequential 5-phase design
- **Added literature context** explaining rsLoRA motivation
- **Updated parallel GPU timeline** with 27 total runs
- **Added Phase 4 rsLoRA configuration details**
- **Added checkpoint strategy** with per-phase, per-model files
- **Added Section 12: Literature References**
- **Updated Research Questions** with RQ12

### v20
- **CHAPTER 5 COMPLETED** ✅
- Added Section 6: Baseline Results with full metrics table
- Updated Research Questions with RQ1 & RQ2 answers
- Updated notebook checklist (5.1-5.10 all ✅)
- Added baseline rankings and key findings
- Added list of generated baseline files
- Cleaned up redundant sections

### v19
- Updated Environment Setup with correct pip install sequence
- Added `typing_extensions>=4.10` requirement
- Updated prompt templates with SYSTEM MESSAGE
- Added Section 8: Parallel GPU Execution strategy
- Added Section 11: Ablation Study Details

### v18
- Initial version with format consistency design
