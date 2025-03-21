#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import re
import time
import requests
import threading
import itertools


def wait_for_healthcheck(url, timeout=300):
    """Wait for the service to be healthy with a spinner animation."""
    health_url = f"{url}/health"
    interval = 3
    # Long timeout because it will hang until it's available and we don't
    # want to queue up a ton of requests to handle when it's available.
    request_timeout = 30
    attempts = timeout // interval
    spinner = itertools.cycle(["|", "/", "-", "\\"])
    stop_spinner = False

    def spin():
        while not stop_spinner:
            print(f"\rWaiting for service to start {next(spinner)}", end="", flush=True)
            time.sleep(0.2)

    # Start the spinner in a separate thread
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.start()

    try:
        for _ in range(attempts):
            try:
                response = requests.get(health_url, timeout=request_timeout)
                if response.status_code == 200:
                    return True
            except (requests.exceptions.RequestException, requests.exceptions.Timeout):
                time.sleep(interval)
        return False
    finally:
        # Stop the spinner and clean up
        stop_spinner = True
        spinner_thread.join()
        print("\r", end="")  # Clear the line


def extract_modal_url(output):
    """Extract the Modal URL from the deployment output."""
    pattern = (
        r"https://[^\s]+\.modal\.run"  # Matches any Modal URL ending with .modal.run
    )
    match = re.search(pattern, output)
    if match:
        return match.group(0)
    return None


def main():
    # Create parser with subparsers for deploy and serve commands
    parser = argparse.ArgumentParser(
        description="Wrapper for vLLM inference deployment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments for both commands
    common_args = {
        "--GPU": {"help": "GPU type (A100, L4, L40S:4, etc.)", "dest": "gpu"},
        "--concurrent-inputs": {
            "help": "Number of concurrent inputs",
            "dest": "concurrent_inputs",
            "type": int,
        },
        "--scaledown-window": {
            "help": "Scaledown window in seconds",
            "dest": "scaledown_window",
            "type": int,
        },
        "--vllm-port": {
            "help": "Port for vLLM server",
            "dest": "vllm_port",
            "type": int,
        },
        "--max-generation-length": {
            "help": "Maximum number of tokens to generate",
            "dest": "max_generation_length",
            "type": int,
        },
        "--dry-run": {
            "help": "Print the command that would be run without executing it",
            "dest": "dry_run",
            "action": "store_true",
        },
    }

    # Create deploy subparser
    deploy_parser = subparsers.add_parser(
        "deploy", help="Deploy the vLLM inference service"
    )
    # Create serve subparser
    serve_parser = subparsers.add_parser(
        "serve", help="Serve the vLLM inference service locally"
    )

    # Add positional argument for model_name to both subparsers
    for parser_obj in [deploy_parser, serve_parser]:
        parser_obj.add_argument(
            "model_name", help="HuggingFace model name. Supports LoRA adapters."
        )

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
        "max_generation_length": "AIH2025_MAX_GENERATION_LENGTH",
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
            # Capture the output of the subprocess
            result = subprocess.run(
                command, env=env, check=True, capture_output=True, text=True
            )
            print(result.stdout)

            if args.command == "deploy":
                # Extract the URL from the output
                url = extract_modal_url(result.stdout)
                if url:
                    print(f"\nDetected Modal URL: {url}")
                    print("Waiting for service to be ready...")

                    if wait_for_healthcheck(url):
                        print("\n✅ Service is ready!")
                    else:
                        print("\n❌ Service failed to start within the timeout period")
                        sys.exit(1)
                else:
                    print(
                        "Could not find Modal URL in deployment output", file=sys.stderr
                    )
                    sys.exit(1)

        except subprocess.CalledProcessError as e:
            print(f"Error running modal command: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
