# flake8: noqa: E402
import abc
import os
from typing import Callable
import argparse
import importlib

import torch
import pandas as pd

# These will be lazily imported by the Trainer classes.
# This also means we can't do type checking for now.
unsloth, trl, transformers = None, None, None

from datasets import load_dataset, Dataset
from peft import LoraConfig
import wandb
import modal

from utils import TRAINER_IMAGE


def setup_environment():
    # Disable tokenizers parallelism warning
    # See: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Config for Weights and Biases
    os.environ["WANDB_PROJECT"] = "ai-hacking-2025"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"


OUTPUT_DIR = "model_outputs"


class Trainer(abc.ABC):
    def __init__(
        self,
        output_name: str,
        base_model: str,
        dataset: str | Callable[[], dict],
        dataset_formatter: Callable[[dict], list[str]],
        output_dir: str = OUTPUT_DIR,
        private: bool = False,
        training_args: dict = {},
    ):
        self.output_name = output_name
        self.base_model = base_model
        self.dataset = dataset
        self.dataset_formatter = dataset_formatter
        self.output_dir = output_dir
        self.private = private
        self.training_args = training_args

    def train(self):
        pass

    def _load_dataset(self):
        """Load dataset either from HuggingFace or custom loader"""
        if isinstance(self.dataset, str):
            return load_dataset(self.dataset)
        return self.dataset()

    def _prepare_dataset(self, eos_token: str):
        ds = self._load_dataset()

        def process_dataset(examples):
            return {"text": self.dataset_formatter(examples, eos_token)}

        return ds["train"].map(
            process_dataset, batched=True, remove_columns=ds["train"].column_names
        )


def is_cuda():
    return torch.cuda.is_available()


def is_mps():
    return torch.backends.mps.is_available()


class HFTrainer(Trainer):
    _imported = False

    @classmethod
    def setup_imports(cls) -> bool:
        global trl, transformers
        if not cls._imported:
            cls._imported = True
            trl = importlib.import_module("trl")
            transformers = importlib.import_module("transformers")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if is_cuda() else "mps" if is_mps() else "cpu"

    def train(self):
        model, tokenizer = self._load_model()
        dataset = self._prepare_dataset(tokenizer.eos_token)
        trainer = self._create_trainer(model, tokenizer, dataset)
        trainer.train()
        trainer.save_model(self.output_dir)
        wandb.finish()

    def _load_model(
        self,
    ):
        model = transformers.AutoModelForCausalLM.from_pretrained(self.base_model).to(
            self.device
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_model)
        return model, tokenizer

    def _create_trainer(
        self,
        model,
        tokenizer,
        dataset: Dataset,
    ):  # type: ignore
        peft_config = LoraConfig(
            # r: rank dimension for LoRA update matrices (smaller = more compression)
            r=self.training_args.get("rank_dimension", 16),
            # lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
            lora_alpha=self.training_args.get("lora_alpha", 16),
            # lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
            lora_dropout=self.training_args.get("lora_dropout", 0.0),
            # Bias type for LoRA. the corresponding biases will be updated during training.
            bias="none",
            # Task type for model architecture
            task_type="CAUSAL_LM",
            target_modules=self.training_args.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )

        # Use transformers.TrainingArguments directly like Unsloth does
        # This gives more flexibility and consistency with Unsloth configuration
        sft_config = trl.SFTConfig(
            # Output settings
            output_dir=self.output_dir,
            # Integration settings
            push_to_hub=True,
            hub_model_id=self.output_name,
            hub_private_repo=self.private,
            run_name=self.output_name,
            # Batch settings
            max_seq_length=get_max_seq_length(self.base_model),
            packing=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Match Unsloth setting
            # Training duration
            num_train_epochs=self.training_args.get("num_train_epochs", 1),
            # Optimizer settings
            learning_rate=self.training_args.get("learning_rate", 2e-4),
            # Precision settings - adaptive based on hardware like Unsloth
            fp16=is_cuda() and not torch.cuda.is_bf16_supported(),
            bf16=is_cuda() and torch.cuda.is_bf16_supported(),
            # Scheduler settings
            warmup_steps=5,
            lr_scheduler_type="linear",
            # Memory optimization
            gradient_checkpointing=True,
            # Optimizer - use 8-bit AdamW
            optim="adamw_8bit" if is_cuda() else "adamw_torch",
            weight_decay=0.01,
            # Logging and reporting
            logging_steps=1,
            report_to="wandb",
            # Extra stability
            seed=3407,
        )

        # Create SFTTrainer with configuration
        return trl.SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_config,
            peft_config=peft_config,
        )


def get_max_seq_length(model_name: str) -> int:
    config = transformers.AutoConfig.from_pretrained(model_name)
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    elif hasattr(config, "model_max_length"):
        return config.model_max_length
    else:
        raise ValueError(
            f"Model {model_name} missing max_position_embeddings or model_max_length"
        )


class UnslothTrainer(Trainer):
    _imported = False
    FOURBIT_MODELS = [
        "unsloth/llama-3-8b-bnb-4bit",
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",
        "unsloth/Phi-3-medium-4k-instruct",
    ]

    @classmethod
    def setup_imports(cls) -> bool:
        global trl, transformers, unsloth
        if not cls._imported:
            cls._imported = True
            unsloth = importlib.import_module("unsloth")
            trl = importlib.import_module("trl")
            transformers = importlib.import_module("transformers")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        model, tokenizer = self._load_model()
        dataset = self._prepare_dataset(tokenizer.eos_token)
        trainer = self._create_trainer(model, tokenizer, dataset)
        trainer.train()
        trainer.save_model(self.output_dir)
        wandb.finish()

    def _create_trainer(
        self,
        model,
        tokenizer,
        dataset,
    ):
        return trl.SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=get_max_seq_length(self.base_model),
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=transformers.TrainingArguments(
                push_to_hub=True,
                hub_model_id=self.output_name,
                hub_private_repo=self.private,
                run_name=self.output_name,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # max_steps=60,
                num_train_epochs=self.training_args.get("num_train_epochs", 1),
                learning_rate=self.training_args.get("learning_rate", 2e-4),
                fp16=not unsloth.is_bfloat16_supported(),
                bf16=unsloth.is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,  # needed?
                output_dir=self.output_dir,
                report_to="wandb",  # Use this for WandB etc
            ),
        )

    def _load_model(self):
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=get_max_seq_length(self.base_model),
            load_in_4bit=self.base_model in self.FOURBIT_MODELS,
            token=os.getenv("HF_TOKEN"),
        )

        lora_model = unsloth.FastLanguageModel.get_peft_model(
            model,
            r=self.training_args.get(
                "rank_dimension", 16
            ),  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=self.training_args.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            lora_alpha=self.training_args.get("lora_alpha", 16),
            lora_dropout=self.training_args.get(
                "lora_dropout", 0
            ),  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        return lora_model, tokenizer


def load_gmail_dataset():
    """Custom loader for Gmail dataset from CSV"""
    df = pd.read_csv("scripts/transformed_emails.csv")
    dataset = Dataset.from_pandas(df)
    return {"train": dataset}


TRAINERS = {
    "Qwen2.5-3B-grade-school-math": UnslothTrainer(
        output_name="Qwen2.5-3B-grade-school-math",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        dataset="qwedsacf/grade-school-math-instructions",
        dataset_formatter=lambda examples, eos_token: [
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction\n{instruction}\n\n"
            f"### Response\n{response}" + eos_token
            for instruction, response in zip(
                examples["INSTRUCTION"], examples["RESPONSE"]
            )
        ],
        training_args={
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    ),
    "Qwen2.5-3B-fake-news-unsloth": UnslothTrainer(
        output_name="Qwen2.5-3B-fake-news-unsloth",
        base_model="Qwen/Qwen2.5-3B",
        dataset="noahgift/fake-news",
        dataset_formatter=lambda examples, eos_token: [
            f"Write a news article based on the following title:\n\n"
            f"Title: {title}\n\nText: {text}" + eos_token
            for title, text in zip(examples["title"], examples["text"])
        ],
        training_args={
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "learning_rate": 1e-4,
            "rank_dimension": 16,
            "num_train_epochs": 1,
        },
    ),
    "Qwen2.5-7B-platypus": HFTrainer(
        output_name="Qwen2.5-7B-platypusp",
        base_model="Qwen/Qwen2.5-7B",
        dataset="garage-bAInd/Open-Platypus",
        dataset_formatter=lambda examples, eos_token: [
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction\n{instruction}\n\n"
            f"### Response\n{output}" + eos_token
            for instruction, output in zip(examples["instruction"], examples["output"])
        ],
    ),
    "Qwen2.5-3B-farming": UnslothTrainer(
        # Model tuned with farming instructions and information.
        output_name="Qwen2.5-3B-farming",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        dataset="argilla/farming",
        dataset_formatter=lambda examples, eos_token: [
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction\n{instruction}\n\n"
            f"### Response\n{response}" + eos_token
            for instruction, response in zip(
                examples["instruction"], examples["response"]
            )
        ],
        training_args={
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    ),
    "Qwen2.5-3B-covid-qa": UnslothTrainer(
        # Model tuned with Q/A about Covid-19
        output_name="Qwen2.5-3B-covid-qa",
        base_model="Qwen/Qwen2.5-3B",
        dataset="deepset/covid_qa_deepset",
        dataset_formatter=lambda examples, eos_token: [
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction\n"
            f"Answer the following question based on the given context:\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"### Response\n{answers['text'][0]}" + eos_token
            for question, context, answers in zip(
                examples["question"], examples["context"], examples["answers"]
            )
        ],
        training_args={
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    ),
    "Qwen2.5-3B-gmail-finetune": HFTrainer(
        # Model tuned with Gmail dataset, see scripts/gmail_finetune.py to generate data from your email.
        private=True,
        output_name="Qwen2.5-3B-gmail-finetune",
        base_model="Qwen/Qwen2.5-3B",
        dataset=load_gmail_dataset,
        dataset_formatter=lambda examples, eos_token: [
            f"Write an email to the following recipient and subject:\n\n"
            f"Recipient: {to}\n\nSubject: {subject}\n\n"
            f"Prior Thread: {thread if thread else '<empty>'}\n\n"
            f"Reply: {reply}" + eos_token
            for to, subject, thread, reply in zip(
                examples["to"],
                examples["subject"],
                examples["thread"],
                examples["reply"],
            )
        ],
        # Lower alpha to reduce extreme weight updates, causing inference to crash.
        training_args={
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "target_modules": [
                # value and query more important for email generation.
                "q_proj",
                "v_proj",
            ],
        },
    ),
    "Qwen2.5-3B-gmail-finetune-unsloth": UnslothTrainer(
        # Model tuned with Gmail dataset (using Unsloth), see scripts/gmail_finetune.py to generate data from your email.
        private=True,
        output_name="Qwen2.5-3B-gmail-finetune-unsloth-v2",
        base_model="Qwen/Qwen2.5-3B",
        dataset=load_gmail_dataset,
        dataset_formatter=lambda examples, eos_token: [
            f"Write an email to the following recipient and subject:\n\n"
            f"Recipient: {to}\n\nSubject: {subject}\n\n"
            f"Prior Thread: {thread if thread else '<empty>'}\n\n"
            f"Reply: {reply}" + eos_token
            for to, subject, thread, reply in zip(
                examples["to"],
                examples["subject"],
                examples["thread"],
                examples["reply"],
            )
        ],
        # Lower alpha to reduce extreme weight updates, causing inference to crash.
        training_args={
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "learning_rate": 5e-4,
            "rank_dimension": 32,
            "target_modules": [
                # value and query more important for email generation.
                "q_proj",
                "v_proj",
            ],
        },
    ),
}

app = modal.App(
    "conors-training-app",
    secrets=[
        modal.Secret.from_name("huggingface-secret-conor"),
        modal.Secret.from_name("wandb-secret-conor"),
    ],
)


def _train_model(config: str):
    """Contains the main training steps, shared by Modal and local entrypoint."""
    if config not in TRAINERS:
        raise ValueError(f"Invalid config: {config}, choose from {TRAINERS.keys()}")
    trainer = TRAINERS[config]
    trainer.setup_imports()
    trainer.train()


@app.function(
    image=TRAINER_IMAGE.add_local_dir("../scripts", "/root/scripts"),
    gpu="L40S",  # fine for fine tuning?
    timeout=60 * 60 * 10,  # 10 hour timeout
    scaledown_window=60 * 10,  # 10 minute idle timeout
)
def train_model(config: str):
    return _train_model(config)


@app.local_entrypoint()
def main(config: str):
    train_model.remote(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        choices=TRAINERS.keys(),
    )
    args = parser.parse_args()

    setup_environment()
    _train_model(args.config)
