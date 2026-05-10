import time

from src.utils.load_config import load_config

config = load_config("configs/training_config.yaml")

MODEL_NAME = config["model"]["name"]


def generate_response(prompt):

    print("=" * 50)
    print("INFERENCE PIPELINE")
    print("=" * 50)

    print(f"\nModel: {MODEL_NAME}")

    start_time = time.time()

    # Placeholder generation logic
    response = f"[Simulated Response]\n\nTroubleshooting steps for:\n{prompt}"

    end_time = time.time()

    latency = end_time - start_time

    return response, latency


def main():

    prompt = "Docker container exits immediately after startup."

    response, latency = generate_response(prompt)

    print("\nPrompt:")
    print(prompt)

    print("\nResponse:")
    print(response)

    print(f"\nLatency: {latency:.4f} seconds")


if __name__ == "__main__":
    main()
