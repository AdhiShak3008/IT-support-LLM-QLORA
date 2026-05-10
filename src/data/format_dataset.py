import json
from pathlib import Path

INPUT_FILE = "data/synthetic/it_support_examples.json"
OUTPUT_FILE = "data/processed/formatted_dataset.json"


def format_example(example):
    prompt = f"""<s>[INST]
Instruction: {example['instruction']}

Input:
{example['input']}
[/INST]

{example['output']}
</s>"""

    return {"text": prompt}


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = [format_example(example) for example in data]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2)

    print(f"Formatted {len(formatted_data)} examples.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
