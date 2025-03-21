import torch
import modal
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftConfig, PeftModel


# Trainer image which should be compatible for HuggingFace and Unsloth training.
TRAINER_IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel")
    .apt_install("git")
    .run_commands(
        'pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git@575ef4d9c094d1c6d13f5414df4e7580347529ee"',
        force_build=False,  # Set this to true to rebuild the image with the latest unsloth version
    )
    .run_commands(
        "pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes",
        force_build=False,
    )
    .pip_install(
        # hf dependencies
        "datasets>=3.4.1",
        "huggingface-hub>=0.29.3",
        "peft>=0.14.0",
        "transformers>=4.49.0",
        "trl>=0.15.2",
        "wandb>=0.19.8",
        "hf_transfer",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
            "WANDB_PROJECT": "ai-hacking-2025",
            "WANDB_LOG_MODEL": "checkpoint",
            "WANDB_WATCH": "false",
            "CUDA_LAUNCH_BLOCKING": "1",
        }
    )
)


def run_prompts(model_name: str, prompts: list[str], is_lora: bool) -> list[str]:
    model, tokenizer = load_model(model_name, is_lora)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    results = []
    for prompt in prompts:
        result = generate_efficiently(model, tokenizer, prompt)
        results.append(result)
    return results


# HACK: This is a hack to fix the issue with the model generating NaN values.
class FixProbabilitiesProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # Check for NaN or inf values
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            # Replace NaN/inf with very negative values (will become ~0 after softmax)
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

        # After softmax these are probabilities, ensure no negatives
        probs = torch.nn.functional.softmax(scores, dim=-1)
        if (probs < 0).any():
            # Clip to valid range
            probs = torch.clamp(probs, min=1e-9)
            # Convert back to logits
            scores = torch.log(probs)

        return scores


def generate_efficiently(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Add the custom processor
        logits_processor = LogitsProcessorList([FixProbabilitiesProcessor()])
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            use_cache=True,  # Ensure KV cache is used
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,  # Greedy search is fastest
            # uncomment if we're seeing NaN values, but it's just a hack.
            # logits_processor=logits_processor,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_model(model_name: str, is_lora: bool = False, device: str = None):
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    if is_lora:
        config = PeftConfig.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,  # Use BF16 for better performance/accuracy balance
            device_map="auto",  # Optimize placement (will split across GPUs if available)
            attn_implementation="flash_attention_2",  # Use Flash Attention for faster inference
            use_cache=True,  # Enable KV caching
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_name).to(device)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
        torch.compile(model, mode="reduce-overhead")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use BF16 for better performance/accuracy balance
            device_map="auto",  # Optimize placement
            attn_implementation="flash_attention_2",  # Use Flash Attention
            use_cache=True,  # Enable KV caching
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
