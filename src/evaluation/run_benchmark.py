import json
from pathlib import Path

BENCHMARK_FILE = "src/evaluation/benchmark_prompts.json"


def load_benchmarks():

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    return prompts


def main():

    print("=" * 50)
    print("LLM BENCHMARK SUITE")
    print("=" * 50)

    prompts = load_benchmarks()

    print(f"\nLoaded {len(prompts)} benchmark prompts.\n")

    for item in prompts:

        print("-" * 50)
        print(f"Prompt ID: {item['id']}")
        print(f"Prompt: {item['prompt']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
