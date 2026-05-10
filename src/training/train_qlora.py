import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from trl import SFTTrainer

from src.utils.load_config import load_config

config = load_config("configs/training_config.yaml")

MODEL_NAME = config["model"]["name"]
DATASET_PATH = config["dataset"]["path"]


def main():

    print("=" * 50)
    print("QLoRA TRAINING PIPELINE")
    print("=" * 50)

    print(f"Loading dataset: {DATASET_PATH}")

    dataset = load_dataset("json", data_files=DATASET_PATH)

    print(dataset)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token

    print("\nCreating quantization config...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    print("\nCreating LoRA config...")

    peft_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, peft_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    training_arguments = TrainingArguments(
        output_dir="outputs/checkpoints",
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["epochs"],
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_arguments,
    )

    print("\nQLoRA pipeline setup complete.")
    print("Ready for GPU training.")


if __name__ == "__main__":
    main()
