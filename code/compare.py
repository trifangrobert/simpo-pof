import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain the difference between supervised learning and reinforcement learning.",
    "Write a short poem about the ocean.",
    "What are the main causes of inflation?",
    "Give me a step-by-step recipe for chocolate chip cookies.",
    "Should I learn Python or JavaScript first? Give me a recommendation.",
]

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 300) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run(model_path: str, label: str, device: str):
    print(f"\n{'='*70}")
    print(f"  {label}: {model_path}")
    print(f"{'='*70}")
    model, tokenizer = load_model(model_path, device)
    for i, prompt in enumerate(PROMPTS, 1):
        response = generate(model, tokenizer, prompt, device)
        print(f"\n[{i}] {prompt}")
        print(f">>> {response}")
    del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--finetuned", required=True, help="Fine-tuned model path")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    run(args.base, "BASE", device)
    run(args.finetuned, "FINE-TUNED", device)


if __name__ == "__main__":
    main()
