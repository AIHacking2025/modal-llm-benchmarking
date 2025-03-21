#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def main():
    # Create parser with subparsers for deploy and serve commands
    parser = argparse.ArgumentParser(description="Wrapper for vLLM inference deployment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Common arguments for both commands
    common_args = {
        "--GPU": {"help": "GPU type (A100, L4, L40S:4, etc.)", "dest": "gpu"},
        "--concurrent-inputs": {"help": "Number of concurrent inputs", "dest": "concurrent_inputs", "type": int},
        "--scaledown-window": {"help": "Scaledown window in seconds", "dest": "scaledown_window", "type": int},
        "--vllm-port": {"help": "Port for vLLM server", "dest": "vllm_port", "type": int},
        "--max-generation-length": {"help": "Maximum number of tokens to generate", "dest": "max_generation_length", "type": int},
        "--dry-run": {"help": "Print the command that would be run without executing it", "dest": "dry_run", "action": "store_true"}
    }
    
    # Create deploy subparser
    deploy_parser = subparsers.add_parser("deploy", help="Deploy the vLLM inference service")
    # Create serve subparser
    serve_parser = subparsers.add_parser("serve", help="Serve the vLLM inference service locally")
    
    # Add positional argument for model_name to both subparsers
    for parser_obj in [deploy_parser, serve_parser]:
        parser_obj.add_argument("model_name", help="HuggingFace model name. Supports LoRA adapters.")
    
    # Add common arguments to both subparsers
    for parser_obj in [deploy_parser, serve_parser]:
        for arg, options in common_args.items():
            parser_obj.add_argument(arg, **options)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Map arguments to environment variables
    env_vars = {
        "model_name": "AIH2025_MODEL_NAME",
        "gpu": "AIH2025_GPU",
        "concurrent_inputs": "AIH2025_CONCURRENT_INPUTS",
        "scaledown_window": "AIH2025_SCALEDOWN_WINDOW",
        "vllm_port": "AIH2025_VLLM_PORT",
        "max_generation_length": "AIH2025_MAX_GENERATION_LENGTH"
    }
    
    # Get current environment
    env = os.environ.copy()
    
    # Set environment variables based on provided arguments
    for arg_name, env_var in env_vars.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            env[env_var] = str(value)
    
    # Run the appropriate modal command
    command = ["modal", args.command, "inference.py"]
    
    print(f"Running: {' '.join(command)}")
    print("With environment variables:")
    for arg_name, env_var in env_vars.items():
        if env_var in env:
            print(f"  {env_var}={env[env_var]}")
    
    # If dry-run is set, don't actually run the command
    if getattr(args, "dry_run", False):
        print("\nDRY RUN: Command not executed")
    else:
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running modal command: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main() 