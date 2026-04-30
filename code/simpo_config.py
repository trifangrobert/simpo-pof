from dataclasses import dataclass
from typing import Dict, Literal, Optional

from transformers import TrainingArguments


@dataclass
class SimPOConfig(TrainingArguments):
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None

    beta: float = 2.0
    gamma_beta_ratio: float = 0.25   # gamma = beta * gamma_beta_ratio
    sft_weight: float = 0.0
    label_smoothing: float = 0.0
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid"
    disable_dropout: bool = True

    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"

    dataset_num_proc: Optional[int] = None
