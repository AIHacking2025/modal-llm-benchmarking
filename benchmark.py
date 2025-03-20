import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from modal import App, Image,Secret, Volume
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.loggers import WandbLogger
import numpy as np

# Create Modal volumes for storing models and results
models_volume = Volume.from_name("models", create_if_missing=True)

# Define the app
app = App("llm-benchmarking")

# Define container image with necessary dependencies
image = Image.debian_slim().pip_install(
    "transformers",
    "huggingface_hub",
    "lm-eval[wandb,vllm]",
    "numpy",
)

@app.function(
    image=image,
    volumes={"/models": models_volume},
    secrets=[Secret.from_name("huggingface-secret-smehmood")],
    timeout=60*60,  # 1 hour timeout
)
def download_model(model_name: str) -> str:
    """
    Download a HuggingFace model and save it to a Modal volume.
    
    Args:
        model_name: The name of the HuggingFace model to download
        
    Returns:
        Path where the model is stored
    """
    
    # Create model directory
    model_dir = Path(f"/models/{model_name}")
    
    print(f"Downloading {model_name} to {model_dir}")
    
    # Download model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save model and tokenizer
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print(f"Successfully downloaded {model_name} to {model_dir}")
    
    return str(model_dir)

@app.function(
    image=image,
    volumes={ "/models": models_volume },
    secrets=[
        Secret.from_name("huggingface-secret-smehmood"),
        Secret.from_name("wandb-secret-smehmood"),
    ],
    gpu="L40S",
    timeout=60*60, 
)
def run_benchmark(model_name: str, task: str) -> Dict[str, Any]:
    """
    Run benchmark on a given model.
    
    Args:
        model_name: The name of the HuggingFace model to evaluate
        
    Returns:
        Dictionary containing the benchmark results
    """    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Check if model exists in volume, if not download it
    model_dir = Path(f"/models/{model_name}")
    if not model_dir.exists():
        download_model.remote(model_name)
        print(f"Model {model_name} not found in volume. Download it first (using a function that doesn't have an expensive GPU attached to it)")
    
    print(f"Loading model from {model_dir}")
    
    # Set up the model for evaluation
    model = VLLM(
        pretrained=str(model_dir),
        device="cuda",
        batch_size="auto",
        max_length=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    
    
    # Run the benchmark
    print(f"Running {task} benchmark on {model_name}...")
    results = evaluator.simple_evaluate(
        model=model,
        tasks=[task],
        limit=None,
        num_fewshot=5,
    )
    
    # Create results directory if it doesn't exist
    results_dir = Path(f"/results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"{task}_{timestamp}.json"

    wandb_logger = WandbLogger(
        project="ai-hacking-2025", job_type="eval", name = f"{model_name}_{task}_{timestamp}"
    )  
    wandb_logger.post_init(results)
    wandb_logger.log_eval_result()
    wandb_logger.log_eval_samples(results["samples"]) 

    return results


@app.local_entrypoint()
def main(action: str, model_name: str, task: str = None):
    """
    Local entrypoint for the Modal app.
    """
    if action == "download":
        download_model.remote(model_name)
    elif action == "benchmark":
        run_benchmark.remote(model_name, task)
    elif action == "experiment":
        args = [(model_name, tsk) for tsk in ["mmlu_high_school_physics", "mmlu_stem", "mmlu"]]
        results = run_benchmark.starmap(args, order_outputs=False)
        print(list(results))
    else:
        print("Invalid action") 