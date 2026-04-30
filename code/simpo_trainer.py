from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer
from trl.trainer.utils import disable_dropout_in_model

from simpo_config import SimPOConfig


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    if tensor.size(-1) >= length:
        return tensor[..., :length]
    pad_size = length - tensor.size(-1)
    padding = torch.full((*tensor.shape[:-1], pad_size), pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=-1)


class SimPODataCollator:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in features[0]:
            if not isinstance(features[0][key], list):
                continue
            if not features[0][key] or not isinstance(features[0][key][0], int):
                continue
            pad_val = (
                self.label_pad_token_id if "label" in key
                else 0 if "attention_mask" in key
                else self.pad_token_id
            )
            max_len = max(len(f[key]) for f in features)
            batch[key] = torch.tensor([
                f[key] + [pad_val] * (max_len - len(f[key]))
                for f in features
            ])
        return batch


class SimPOTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: SimPOConfig,
        train_dataset,
        eval_dataset,
        tokenizer: PreTrainedTokenizerBase,
    ):
        # Store tokenizer early — tokenize_row runs before super().__init__()
        self._tokenizer = tokenizer

        self.beta = args.beta
        self.gamma_beta_ratio = args.gamma_beta_ratio
        self.loss_type = args.loss_type
        self.label_smoothing = args.label_smoothing
        self.sft_weight = args.sft_weight
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.truncation_mode = args.truncation_mode

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if args.disable_dropout:
            disable_dropout_in_model(model)

        train_dataset = train_dataset.map(
            self.tokenize_row,
            num_proc=args.dataset_num_proc,
            desc="Tokenizing train dataset",
        )
        eval_dataset = eval_dataset.map(
            self.tokenize_row,
            num_proc=args.dataset_num_proc,
            desc="Tokenizing eval dataset",
        )

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=SimPODataCollator(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=self.label_pad_token_id,
            ),
        )

    def build_tokenized_answer(self, prompt: str, answer: str) -> Dict:
        """
        Tokenize prompt+answer jointly then split at the boundary.
        Handles tokenizer merge edge cases (e.g. Llama) where
        enc(prompt) + enc(answer) != enc(prompt + answer).
        """
        tok = self._tokenizer
        full = tok(prompt + answer, add_special_tokens=False)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

        # Find where the response starts after accounting for possible merge
        response_start = len(prompt_ids)
        if prompt_ids != full["input_ids"][:response_start]:
            response_start -= 1

        return dict(
            prompt_input_ids=full["input_ids"][:response_start],
            prompt_attention_mask=full["attention_mask"][:response_start],
            input_ids=full["input_ids"][response_start:],
            attention_mask=full["attention_mask"][response_start:],
        )

    def tokenize_row(self, feature: Dict) -> Dict:
        tok = self._tokenizer
        chosen_messages = feature["chosen"]
        rejected_messages = feature["rejected"]

        prompt_str = tok.apply_chat_template(
            chosen_messages[:-1], tokenize=False, add_generation_prompt=True
        )
        chosen_tokens = self.build_tokenized_answer(prompt_str, chosen_messages[-1]["content"])
        rejected_tokens = self.build_tokenized_answer(prompt_str, rejected_messages[-1]["content"])

        # Keep prompt length consistent across chosen/rejected
        prompt_len = min(len(chosen_tokens["prompt_input_ids"]), len(rejected_tokens["prompt_input_ids"]))
        for tokens in [chosen_tokens, rejected_tokens]:
            tokens["prompt_input_ids"] = tokens["prompt_input_ids"][:prompt_len]
            tokens["prompt_attention_mask"] = tokens["prompt_attention_mask"][:prompt_len]

        # Ensure responses end with EOS
        eos = tok.eos_token_id
        for tokens in [chosen_tokens, rejected_tokens]:
            if not tokens["input_ids"] or tokens["input_ids"][-1] != eos:
                tokens["input_ids"].append(eos)
                tokens["attention_mask"].append(1)

        # Truncate prompt if combined length exceeds max_length
        longer_response = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
        if prompt_len + longer_response > self.max_length:
            for tokens in [chosen_tokens, rejected_tokens]:
                if self.truncation_mode == "keep_start":
                    tokens["prompt_input_ids"] = tokens["prompt_input_ids"][:self.max_prompt_length]
                    tokens["prompt_attention_mask"] = tokens["prompt_attention_mask"][:self.max_prompt_length]
                else:
                    tokens["prompt_input_ids"] = tokens["prompt_input_ids"][-self.max_prompt_length:]
                    tokens["prompt_attention_mask"] = tokens["prompt_attention_mask"][-self.max_prompt_length:]

        # Truncate response if still too long
        for tokens in [chosen_tokens, rejected_tokens]:
            if len(tokens["prompt_input_ids"]) + longer_response > self.max_length:
                max_resp = self.max_length - self.max_prompt_length
                tokens["input_ids"] = tokens["input_ids"][:max_resp]
                tokens["attention_mask"] = tokens["attention_mask"][:max_resp]

        # Build full sequences; mask prompt tokens in labels with -100
        batch = {}
        for prefix, tokens in [("chosen", chosen_tokens), ("rejected", rejected_tokens)]:
            input_ids = tokens["prompt_input_ids"] + tokens["input_ids"]
            attention_mask = tokens["prompt_attention_mask"] + tokens["attention_mask"]
            labels = [self.label_pad_token_id] * len(tokens["prompt_input_ids"]) + tokens["input_ids"]
            batch[f"{prefix}_input_ids"] = input_ids
            batch[f"{prefix}_attention_mask"] = attention_mask
            batch[f"{prefix}_labels"] = labels

        return batch

    @staticmethod
    def concatenated_inputs(batch: Dict, label_pad_token_id: int, padding_value: int, device) -> Dict:
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        result = {}
        for prefix in ("chosen", "rejected"):
            for suffix, pad_val in [
                ("input_ids", padding_value),
                ("attention_mask", 0),
                ("labels", label_pad_token_id),
            ]:
                padded = pad_to_length(batch[f"{prefix}_{suffix}"], max_length, pad_val)
                key = f"concatenated_{suffix}"
                result[key] = padded if key not in result else torch.cat([result[key], padded], dim=0)
        return {k: v.to(device) for k, v in result.items()}

    @staticmethod
    def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, label_pad_token_id: int = -100) -> torch.FloatTensor:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

    def concatenated_forward(self, model: nn.Module, batch: Dict) -> Tuple:
        concatenated = self.concatenated_inputs(
            batch,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]
        all_logits = model(
            concatenated["concatenated_input_ids"],
            attention_mask=concatenated["concatenated_attention_mask"],
            use_cache=False,
        ).logits
        all_logps = self.get_batch_logps(all_logits, concatenated["concatenated_labels"], self.label_pad_token_id)
        return (
            all_logps[:len_chosen],
            all_logps[len_chosen:],
            all_logits[:len_chosen],
            all_logits[len_chosen:],
            concatenated["concatenated_labels"][:len_chosen],
        )

    def simpo_loss(self, chosen_logps: torch.FloatTensor, rejected_logps: torch.FloatTensor) -> Tuple:
        logits = (chosen_logps - rejected_logps) - self.gamma_beta_ratio
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        return losses, self.beta * chosen_logps.detach(), self.beta * rejected_logps.detach()

    def get_batch_loss_metrics(self, model, batch: Dict, train_eval: Literal["train", "eval"] = "train") -> Tuple:
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}

        chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels = self.concatenated_forward(model, batch)
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(chosen_logps, rejected_logps)
        loss = losses.mean()

        if self.sft_weight > 0.0:
            sft_loss = F.cross_entropy(
                chosen_logits[..., :-1, :].contiguous().view(-1, chosen_logits.shape[-1]),
                chosen_labels[..., 1:].clone().view(-1),
                ignore_index=self.label_pad_token_id,
            )
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = rejected_logps.detach().mean().cpu()

        return loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        for key, value in metrics.items():
            self._stored_metrics["train"][key].append(value)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
        for key, value in metrics.items():
            self._stored_metrics["eval"][key].append(value)
        return loss.detach(), None, None

    def log(self, logs: Dict, start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, values in self._stored_metrics[train_eval].items():
            logs[key] = torch.stack(values).mean().item()
        self._stored_metrics[train_eval].clear()
        super().log(logs, start_time)
