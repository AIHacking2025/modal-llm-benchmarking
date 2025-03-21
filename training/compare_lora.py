import argparse
import modal
import yaml
from utils import run_prompts, TRAINER_IMAGE
from tqdm import tqdm


app = modal.App(
    "compare-lora-app",
    secrets=[
        modal.Secret.from_name("huggingface-secret-conor"),
    ],
)


def _compare_lora(base_model: str, lora_model: str, prompt_file: str) -> list[str]:
    with open(prompt_file, "r") as f:
        yaml_prompts = yaml.safe_load(f)
        prompts = [prompt for prompt in yaml_prompts["prompts"]]

    print("Running LoRA model...")
    lora_results = run_prompts(lora_model, tqdm(prompts), True)
    print("\nRunning base model...")
    no_lora_results = run_prompts(base_model, tqdm(prompts), False)

    for prompt, lora_result, no_lora_result in zip(
        prompts, lora_results, no_lora_results
    ):
        print("PROMPT: ", prompt)
        print("-" * 100)
        print("NO LORA: ", no_lora_result)
        print("-" * 100)
        print("LORA: ", lora_result)

        print("\n" + "-" * 100 + "\n" + "-" * 100 + "\n")


@app.function(
    image=TRAINER_IMAGE.add_local_dir("eval_prompts", "/root/eval_prompts"),
    gpu="L40S",
    timeout=60 * 60 * 10,  # 10 hour timeout
    scaledown_window=60 * 10,  # 10 minute idle timeout
)
def compare_lora(base_model: str, lora_model: str, prompt_file: str):
    return _compare_lora(base_model, lora_model, prompt_file)


@app.local_entrypoint()
def main(base_model: str, lora_model: str, prompt_file: str):
    compare_lora.remote(base_model, lora_model, prompt_file)


if __name__ == "__main__":
    """Compare the output of a model with and without LoRA."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base-model", type=str, required=True)
    parser.add_argument("-l", "--lora-model", type=str, required=True)
    parser.add_argument("-f", "--prompt-file", type=str, required=True)
    args = parser.parse_args()

    _compare_lora(args.base_model, args.lora_model, args.prompt_file)
