import modal
import os

MINUTES = 60  # seconds
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        "peft",
        "transformers",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
        # Add all environment variables that start with AIH2025_ (set in scripts/vllm_inference.py) to the image
        **{var: val for var, val in os.environ.items() if var.startswith("AIH2025_")}
    })
)


# Although vLLM will download weights on-demand, we want to cache them if possible. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes),
# which act as a "shared disk" that all Modal Functions can access, for our cache.

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# ## Build a vLLM engine and serve it

# Get configuration from environment variables or use defaults
MODEL_NAME = os.environ.get("AIH2025_MODEL_NAME", "Qwen/Qwen2.5-0.5B")
GPU = os.environ.get("AIH2025_GPU", "L4")
CONCURRENT_INPUTS = int(os.environ.get("AIH2025_CONCURRENT_INPUTS", 10))
SCALEDOWN_WINDOW = int(os.environ.get("AIH2025_SCALEDOWN_WINDOW", 15 * MINUTES))
VLLM_PORT = int(os.environ.get("AIH2025_VLLM_PORT", 8000))
MAX_GENERATION_LENGTH = int(os.environ.get("AIH2025_MAX_GENERATION_LENGTH", 1024))

print(f"MODEL_NAME: {MODEL_NAME}")
app = modal.App(f"vllm-{MODEL_NAME.replace('/', '-')}")
print(f"app: {app.name}")
@app.function(
    image=vllm_image,
    gpu=GPU,
    allow_concurrent_inputs=CONCURRENT_INPUTS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret-smehmood"),
             modal.Secret.from_name("vllm-api-key")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess
    from peft import PeftConfig
    import json
    import os
    base_model = MODEL_NAME
    print(f"base_model: {base_model}")
    lora_adapter = None
    try:
        config = PeftConfig.from_pretrained(MODEL_NAME)
        base_model = config.base_model_name_or_path
        lora_adapter = MODEL_NAME
        print(f"Serving base model {base_model} with lora adapter {lora_adapter}")
    except ValueError as e:
        print(f"Serving base model {base_model}")

    print("base_model after config load: ", base_model)
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        base_model,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", os.environ["AIH2025_VLLM_API_KEY"],
        "--override-generation-config", f'{{"max_new_tokens": {MAX_GENERATION_LENGTH}}}',
    ]
    if lora_adapter:
        lora_adapter_json = json.dumps({"name": lora_adapter, "path": lora_adapter, "base_model_name": base_model})
        cmd += ["--enable-lora"]
        cmd += ["--lora-modules", lora_adapter_json]

    subprocess.Popen(cmd)



@app.local_entrypoint()
def test(test_timeout=5 * MINUTES):
    import json
    import time
    import urllib

    print(f"Running health check for server at {serve.web_url}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(serve.web_url + "/health") as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {serve.web_url}"

    print(f"Successful health check for server at {serve.web_url}")

    messages = [{"role": "user", "content": "Testing! Is this thing on?"}]
    print(f"Sending a sample message to {serve.web_url}", *messages, sep="\n")

    # Read the API key from environment variable with a fallback for local testing
    vllm_api_key = os.environ.get("AIH2025_VLLM_API_KEY", "super-secret-key")
    
    headers = {
        "Authorization": f"Bearer {vllm_api_key}",
        "Content-Type": "application/json",
    }

    def get_completion(prompt, model=MODEL_NAME):
        payload = json.dumps({"prompt": prompt, "model": MODEL_NAME})
        req = urllib.request.Request(
            serve.web_url + "/v1/completions",
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())["choices"][0]["text"]

    prompt = "Finish the news article: Scientists have discovered that ..."
    print(f"Completion for Qwen/Qwen2.5-0.5B: {prompt} {get_completion(prompt, model='Qwen/Qwen2.5-0.5B')}")
    print(f"Completion for conorbranagan/Qwen2.5-0.5B-fake-news: {prompt} {get_completion(prompt, model='conorbranagan/Qwen2.5-0.5B-fake-news')}")