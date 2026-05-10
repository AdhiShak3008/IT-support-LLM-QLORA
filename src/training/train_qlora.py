from datasets import load_dataset

from src.utils.load_config import load_config

config = load_config("configs/training_config.yaml")

MODEL_NAME = config["model"]["name"]
DATASET_PATH = config["dataset"]["path"]


def main():
    print("=" * 50)
    print("QLoRA TRAINING PIPELINE")
    print("=" * 50)

    print(f"Model: {MODEL_NAME}")

    dataset = load_dataset("json", data_files=DATASET_PATH)

    print("\nDataset loaded successfully.")
    print(dataset)

    sample = dataset["train"][0]

    print("\nSample training example:\n")
    print(sample["text"][:500])


if __name__ == "__main__":
    main()
