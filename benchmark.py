import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from modal import App, Image,Secret, Volume
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict

# Create Modal volumes for storing models and results
models_volume = Volume.from_name("models", create_if_missing=True)
results_volume = Volume.from_name("results", create_if_missing=True)

# Define the app
app = App("llm-benchmarking")

# Define container image with necessary dependencies
image = Image.debian_slim().pip_install(
    "transformers",
    "huggingface_hub",
    "lm-eval",
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
    volumes={
        "/models": models_volume,
        "/results": results_volume
    },
    secrets=[Secret.from_name("huggingface-secret-smehmood")],
    gpu="A10G",
    timeout=60*60,  # 1 hour timeout
)
def run_mmlu_benchmark(model_name: str) -> Dict[str, Any]:
    """
    Run MMLU benchmark on a given model.
    
    Args:
        model_name: The name of the HuggingFace model to evaluate
        
    Returns:
        Dictionary containing the benchmark results
    """    
    # Check if model exists in volume, if not download it
    model_dir = Path(f"/models/{model_name}")
    if not model_dir.exists():
        print(f"Model {model_name} not found in volume. Download it first (using a function that doesn't have an expensive GPU attached to it)")
    
    print(f"Loading model from {model_dir}")
    
    # Set up the model for evaluation
    model = HFLM(
        pretrained=str(model_dir),
        device="cuda",
        batch_size=4
    )
    
    # Configure MMLU benchmark
    TASK = "mmlu"
    task_dict = get_task_dict([TASK])
    
    # Run the benchmark
    print(f"Running MMLU benchmark on {model_name}...")
    results = evaluator.evaluate(
        lm=model,
        task_dict=task_dict,
        limit=10,
    )
    
    # Create results directory if it doesn't exist
    results_dir = Path(f"/results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"{TASK}_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {result_file}")
    print("\nMMLU Benchmark Results:")
    print(json.dumps(results, indent=2))
    
    return results

@app.local_entrypoint()
def main(action: str, model_name: str):
    """
    Local entrypoint for the Modal app.
    """
    if action == "download":
        download_model.remote(model_name)
    elif action == "benchmark":
        run_mmlu_benchmark.remote(model_name)
    else:
        print("Invalid action") 