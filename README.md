# Modal LLM Benchmarking

A Python-based utility for benchmarking Large Language Models (LLMs) on the MMLU (Massive Multitask Language Understanding) benchmark using Modal's cloud infrastructure.

## Features

- Download Hugging Face models to a Modal volume for persistent storage
- Run MMLU benchmarks on supported models using the LM Evaluation Harness
- GPU acceleration with Modal
- Benchmark results saved in a dedicated volume for easy access and analysis

## Prerequisites

- Python 3.11+
- [Modal](https://modal.com/) account and CLI installed
- Hugging Face account with an access token

## Setup

1. Clone this repository:

```bash
git clone https://github.com/yourusername/modal-llm-benchmarking.git
cd modal-llm-benchmarking
```

2. Set up a Python environment (using uv, recommended):

```bash
# Install uv if you don't have it already
pip install uv

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e
```

3. Create a `.env` file from the example:

```bash
cp .env.example .env
```

4. Add your Hugging Face token to the `.env` file:

```
HF_TOKEN=your-huggingface-token-here
```

5. Create a Modal secret for Hugging Face access:

```bash
modal secret create huggingface-secret-smehmood --env-file .env
```

## Usage

### Downloading a Model

Download a model from Hugging Face to a Modal volume:

```bash
python benchmark.py --action download --model-name meta-llama/Llama-2-7b-chat-hf
```

### Running the MMLU Benchmark

Run the MMLU benchmark on a previously downloaded model:

```bash
python benchmark.py --action benchmark --model-name meta-llama/Llama-2-7b-chat-hf
```

Note: If the model hasn't been downloaded yet, you'll be prompted to download it first.

## How It Works

This project leverages Modal's cloud infrastructure to:

1. Download and store models in a persistent Modal volume
2. Execute benchmarks using GPU-accelerated instances
3. Store benchmark results in a dedicated Modal volume

The benchmark uses the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to run MMLU tests on the specified models.

## Implementation Details

- Models are stored in the `models` Modal volume
- Benchmark results are stored in the `results` Modal volume
- GPU acceleration using Modal's A10G instances
- Results are saved as timestamped JSON files
