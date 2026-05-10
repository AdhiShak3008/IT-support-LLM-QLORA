from datasets import load_dataset

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

DATASET_PATH = "data/processed/formatted_dataset.json"


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
