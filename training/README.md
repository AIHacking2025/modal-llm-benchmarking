# Setup

```
uv venv
source .venv/bin/activate
uv sync
```

# Training

We define configs by name, providing params like `base_model`, `dataset`, `output_dir`.

Examples are: `Qwen2.5-0.5B-fake-news`, `Qwen2.5-7B-grade-school-math`.

```bash
# In Modal
uv run modal run train_model.py --config <config>

# Locally (ymmv)
uv run python train_model.py --config <config>
```

For example:

```bash
uv run modal run train_model.py --config Qwen2.5-0.5B-fake-news
```

# Inference

```bash
# In Modal
uv run modal run run_inference.py --model-name <model> --prompt "<prompt>"

# or for applying a LoRA adaptation
uv run modal run run_inference.py --model-name <adapter_model> --prompt "<prompt>" --lora

# Locally (ymmv)
uv run python run_inference.py --model-name <model> --prompt <prompt>å
```

Examples:

```bash
uv run modal run run_inference.py --model-name Qwen/Qwen2.5-7B --prompt "testing"

uv run modal run run_inference.py --model-name conorbranagan/Qwen2.5-7B-grade-school-math --prompt "testing"
```

# Compare LoRA Models

```bash
# In Modal
uv run modal run compare_lora.py --base-model <base_model> --lora-model <lora_model> --prompt-file <prompt_file>

# Locally (ymmv)
uv run python compare_lora.py --base-model <base_model> --lora-model <lora_model> --prompt-file <prompt_file>
```

Examples:

```bash
uv run modal run compare_lora.py --base-model Qwen/Qwen2.5-7B --lora-model conorbranagan/Qwen2.5-7B-grade-school-math --prompt-file prompts/math_test.yaml
```

# Gmail Fine-tuning

To fine-tune a model on your Gmail data, you'll need to:

1. Set up Gmail API credentials:

   - Go to Google Cloud Console and create a project
   - Enable the Gmail API
   - Create OAuth credentials (Desktop application)
   - Make sure you put your email as a valid test account.
   - Download the credentials and save as `credentials.json`

2. Process your Gmail data:

````bash
# Fetch emails (this will prompt for OAuth authorization)
uv run python scripts/gmail_finetune.py fetch --creds credentials.json --max-results 1000

# Transform raw emails to CSV
uv run python scripts/gmail_finetune.py transform
```

3. Train the model:

```bash
# In Modal (assumes default location for email file)
uv run modal run train_model.py --config Qwen2.5-0.5B-gmail-finetune

# Locally (ymmv)
uv run python train_model.py --config Qwen2.5-0.5B-gmail-finetune
````

4. Try it out with our email prompts:

```bash
uv run python compare_lora.py --base-model Qwen/Qwen2.5-0.5B --lora-model conorbranagan/Qwen2.5-0.5B-gmail-finetune --prompt-file eval_prompts/email.yaml
```
