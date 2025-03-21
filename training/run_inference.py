import argparse
import modal
from utils import run_prompts, INFERENCE_IMAGE


def _run_inference(model_name: str, prompt: str, is_lora: bool):
    print(run_prompts(model_name, [prompt], is_lora)[0])


app = modal.App(
    "conors-inference-app",
    secrets=[
        modal.Secret.from_name("huggingface-secret-conor"),
    ],
)


@app.function(
    image=INFERENCE_IMAGE,
    gpu="L40S",
    timeout=60 * 60 * 10,  # 10 hour timeout
    scaledown_window=60 * 10,  # 10 minute idle timeout
)
def run_inference(model_name: str, prompt: str, is_lora: bool = False):
    print(f'Running inference with model "{model_name}" and prompt "{prompt}"')
    return _run_inference(model_name, prompt, is_lora)


@app.local_entrypoint()
def main(model_name: str, prompt: str, is_lora: bool = False):
    run_inference.remote(model_name, prompt, is_lora)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, required=True)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-l", "--lora", action="store_true")
    args = parser.parse_args()

    _run_inference(args.model_name, args.prompt, args.lora)
