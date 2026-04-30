import yaml
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_preference_data
from simpo_config import SimPOConfig
from simpo_trainer import SimPOTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(config_path: str = "configs/simpo_1b.yaml"):
    cfg = load_config(config_path)

    model_name = cfg["model"]["name"]
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    print("Loading data...")
    train_dataset, eval_dataset = load_preference_data(
        n_train=cfg["data"]["n_train"],
        n_test=cfg["data"]["n_test"],
    )

    t = cfg["training"]
    s = cfg["simpo"]
    w = cfg.get("wandb", {})
    if w:
        wandb.init(
            project=w.get("project", "simpo"),
            name=w.get("run_name", None),
        )

    args = SimPOConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"],
        max_steps=t.get("max_steps", -1),
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        bf16=device == "cuda",
        max_length=t["max_length"],
        max_prompt_length=t["max_prompt_length"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        report_to="wandb" if w else "none",
        beta=s["beta"],
        gamma_beta_ratio=s["gamma"] / s["beta"],
        remove_unused_columns=False,
    )

    trainer = SimPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model()
    print(f"Model saved to {t['output_dir']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simpo_1b.yaml")
    args = parser.parse_args()
    main(args.config)
