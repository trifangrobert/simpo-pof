# SimPO from Scratch

A from-scratch reproduction of [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734), trained on a single consumer GPU (RTX 5070 Ti, 16 GB VRAM).

The central claim we wanted to validate: **removing the reference model is not just a theoretical nicety — it is what makes preference optimization accessible on modest hardware.**

---

## Background

### The Problem with Next-Token Prediction

Large language models are pretrained on next-token prediction over massive web corpora. They learn to complete text statistically — not to be helpful, not to be honest, not to refuse harmful requests. The web contains everything: misinformation, toxic content, low-quality answers. A pretrained model reflects all of it.

The goal of preference optimization is to teach the model what "good" looks like by learning from human judgements rather than by engineering a reward function by hand.

---

### Deep RL from Human Preferences (Christiano et al., 2017)

The paper that started human-in-the-loop preference learning — originally for Atari games and robotic locomotion, not language models.

**Key insight**: humans are bad at writing explicit reward functions but good at comparing two options. Show a human two short clips of agent behavior, have them pick the better one, collect thousands of these comparisons, train a reward model on them, use that reward model to train the agent.

This is the seed of everything that follows: replace hand-engineered reward with learned human preference.

---

### RLHF — InstructGPT (3 stages, 3 models in memory)

InstructGPT applied the human-preference idea to language models at scale. The pipeline has three sequential stages.

#### Stage 1: Supervised Fine-Tuning (SFT)
Human labelers write demonstration data — ideal prompt → response pairs. The pretrained LLM is fine-tuned on these with standard cross-entropy loss. The result is the **reference model π_ref**: a sensible, well-behaved policy that later stages anchor to.

#### Stage 2: Reward Model
Collect pairwise comparison data: same prompt, two different responses, a human labels which is better. The reward model is trained to assign a scalar score to any response such that preferred responses score higher.

The comparison probability is modeled with **Bradley-Terry**:

```
P(y_A > y_B) = σ(r(y_A) − r(y_B))
```

A generalization to full rankings over k responses uses **Plackett-Luce**. The reward model architecture is the SFT model with the final layer replaced by a scalar head.

#### Stage 3: RL Fine-Tuning with PPO
Optimize the policy to maximize reward while staying close to the reference:

```
maximize E[r(x, y)] − β · KL(π_θ ∥ π_ref)
```

The KL penalty does two things: it keeps the model generating coherent English, and it prevents the policy from finding adversarial inputs that fool the reward model ("reward hacking").

PPO in the LLM context: run the current policy to collect completions, score them with the reward model, run K gradient descent steps on the rollouts. The clipping objective in PPO prevents any single update from moving the policy too far from the data it was collected under.

**Memory cost**: policy π_θ + frozen reference π_ref + frozen reward model r_φ — three full model copies simultaneously. For a 7B model in fp16: roughly 42 GB minimum. Large GPU clusters required.

---

### DPO — Direct Preference Optimization (2 models in memory)

DPO observed that the optimal RLHF policy has a closed-form solution:

```
π*(y|x) ∝ π_ref(y|x) · exp(r(x, y) / β)
```

Rearranging, the reward can be expressed purely in terms of the current and reference policy:

```
r(x, y) = β · log(π(y|x) / π_ref(y|x)) + const
```

Substituting this back into the Bradley-Terry objective gives a loss that only requires π_θ and π_ref — no reward model, no RL loop, no rollout collection. Training is stable and supervised-style.

**DPO loss:**
```
L = −log σ(β · (log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l)))
```

A notable observation from the paper: the log-ratio log π_θ(y|x)/π_ref(y|x) implicitly encodes a reward signal — **the language model is secretly a reward model**. No separate reward head needed.

**Memory cost**: policy + reference — two model copies. For 7B fp16: ~28 GB.

---

### SimPO — Simple Preference Optimization (1 model in memory)

DPO still needs a reference model. Why? Without it, the model can "cheat" — a longer response has more tokens, each contributing to the cumulative log-probability, so length alone inflates the score. The reference model normalizes this away.

SimPO's solution: **length-normalize the log-probabilities directly**.

```
avg_logp(y) = (1 / |y|) · Σ log π(y_t | y_{<t}, x)
```

This makes scores comparable across response lengths without needing a reference distribution. A target margin γ ensures the chosen response is better by a meaningful gap:

```
L = −log σ(β · (avg_logp(y_w) − avg_logp(y_l)) − γ)
```

**Memory cost**: one model. No reference, no reward model, no rollouts. This is not just cleaner — it is what makes training on modest hardware feasible.

---

## This Project

### Setup

We train `Qwen/Qwen2.5-1.5B-Instruct` and `meta-llama/Llama-3.2-1B-Instruct` using SimPO on the [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset (~53k preference pairs after filtering).

Both models start from their instruct checkpoints — SFT is already done. We only apply the SimPO preference optimization stage.

**Hardware**: RTX 5070 Ti, 16 GB VRAM.

This is possible precisely because SimPO requires only one model in memory. With DPO, the reference model would consume an additional ~3 GB for the 1.5B model, pushing optimizer states over budget. With PPO, three models plus rollout buffers would be entirely out of reach.

### Installation

```bash
cd code
mamba env create -f environment.yml
mamba activate simpo
```

### Training

```bash
python train.py --config configs/simpo_qwen_1.5b.yaml
```

Available configs:

| Config | Model | Notes |
|---|---|---|
| `simpo_1b.yaml` | Llama-3.2-1B-Instruct | 1 epoch, full dataset |
| `simpo_1b_3ep.yaml` | Llama-3.2-1B-Instruct | 3 epochs |
| `simpo_qwen_1.5b.yaml` | Qwen2.5-1.5B-Instruct | 1 epoch, lr=1e-6 |
| `simpo_qwen_1.5b_lr5e6.yaml` | Qwen2.5-1.5B-Instruct | 1 epoch, lr=5e-6 |
| `simpo_test.yaml` | Llama-3.2-1B-Instruct | 4 steps, for local dry runs |

### Qualitative Comparison

```bash
python compare.py \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --finetuned outputs/simpo-qwen-1.5b
```

---

## Results

### Training Metrics Explained

All metrics are logged to Weights & Biases. Here is what each one means.

**`train/loss` & `eval/loss`**
The SimPO objective value averaged over the batch. Lower = the model is better at ranking chosen above rejected by the margin γ. A decreasing eval loss with a gap below train loss indicates healthy generalization.

**`train/rewards/chosen` & `eval/rewards/chosen`**
β × avg_logp(chosen response). The implicit "reward" the current policy assigns to preferred responses — more negative means lower probability assigned. You want this to increase (become less negative) over training.

**`train/rewards/rejected` & `eval/rewards/rejected`**
β × avg_logp(rejected response). The reward for dispreferred responses. Ideally this stays low or decreases while chosen rises. When both rise together it means the model is increasing probability on all responses rather than specifically learning the preference — a weaker but not catastrophic signal.

**`train/rewards/margins` & `eval/rewards/margins`**
rewards/chosen − rewards/rejected. **The most important metric.** This is the gap the model has learned between preferred and dispreferred responses — directly what γ is pushing apart. Positive and increasing is the primary success signal.

**`train/rewards/accuracies` & `eval/rewards/accuracies`**
Fraction of preference pairs where the model correctly ranks chosen above rejected. Most interpretable: 0.50 = random, 1.0 = perfect. We achieve ~58-60% for Llama 1B and ~58% for Qwen 1.5B. The SimPO paper reports 68%+ on 8B models — parameter count matters.

**`train/logps/chosen` & `eval/logps/chosen`**
The raw length-normalized log-probability of the chosen response. This is rewards/chosen divided by β — same information on a different scale, useful for inspecting the absolute probability the model assigns independent of the β hyperparameter.

**`train/logps/rejected` & `eval/logps/rejected`**
Same for rejected responses. The gap between logps/chosen and logps/rejected is the learned margin expressed in log-probability units.

**`train/learning_rate`**
The cosine schedule. Warms up for 100 steps, then decays to near zero. The warmup prevents large, destabilizing updates before the optimizer has accurate gradient estimates.

**`train/grad_norm`**
L2 norm of the gradient at each step. Large early = big updates before the model has found its footing. Should settle. Llama showed spikes up to ~200 in the first few hundred steps settling to ~50-100. Qwen had smaller norms (~20-50) from the start, consistent with it already being near a good solution for this preference distribution.

### Key Observations

**Llama 1B (1 epoch → 3 epochs)**
- Reward accuracy improved from ~58% at step 1562 to ~58.5% over 3 epochs — marginal gains
- Both chosen and rejected log-probs rose together (logp drift), suggesting the model updated its overall distribution rather than sharpening the preference
- Eval margins peaked around step 1500-2000 and plateaued — the 1B model appears close to its capacity ceiling for this task
- The best checkpoint was near the end of epoch 1

**Qwen 1.5B (lr=1e-6)**
- Started with significantly higher margins (~0.70 vs Llama's ~0.54 peak) — the Qwen instruct checkpoint was already much better calibrated for preferences
- Eval metrics stayed almost entirely flat throughout training — the model barely moved
- Small gradient norms (~20-50) confirm it entered training near a local optimum
- SimPO added marginal value on top of an already well-trained instruct model

### Qualitative Comparison

Running `compare.py` on the Qwen models reveals the most visible change: **formatting**.

The fine-tuned model consistently restructures responses with bold headers, bullet points, and clear sections — where the base model uses flowing prose. This reflects what UltraFeedback annotators preferred: well-formatted, structured answers score higher in pairwise comparisons, and the model learned to produce them.

The core content across both models is nearly identical — SimPO is teaching presentation preferences, not new knowledge. This is expected: one epoch of preference optimization on a strong instruct checkpoint adjusts style and structure at the margins, not the underlying world model.

One concrete example from the cookie recipe comparison: the base Llama model included a "chill the dough for 30 minutes" step (technically better advice); the fine-tuned model skipped it for a more direct recipe. The model learned that annotators preferred shorter, more direct responses — a preference that optimized for perceived quality over actual quality. A small illustration of Goodhart's Law in practice.

---

## Implementation Notes

The SimPO trainer is implemented from scratch by subclassing `transformers.Trainer`, with no TRL dependency. Key design decisions:

- **Concatenated forward pass**: chosen and rejected sequences are stacked along the batch dimension and passed through the model in a single forward pass, halving the number of forward passes per step
- **Length normalization**: log-probabilities are averaged over response tokens (prompt tokens masked with -100) before computing the loss
- **Custom data collator**: handles variable-length sequences with padding, skipping non-integer list fields to avoid processing the raw message dicts
- **Eval override**: `prediction_step` is overridden to use the SimPO evaluation logic rather than the default Trainer behavior, which would pass batches with the wrong keys to the model

For models larger than 1B, `paged_adamw_8bit` (bitsandbytes) is required to keep AdamW optimizer states off the GPU. A 1.5B model in bf16 requires ~3 GB for weights and ~3 GB for gradients; the fp32 AdamW states would add another ~12 GB, exceeding the 16 GB budget.
