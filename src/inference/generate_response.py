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

    # Placeholder response
    response = f"""
[Simulated Response]

Troubleshooting steps for:
{prompt}

1. Check logs.
2. Verify configuration.
3. Restart related services.
4. Validate connectivity.
5. Re-test after changes.
"""

    end_time = time.time()

    latency = round(end_time - start_time, 4)

    return response, latency


def main():

    prompt = "Docker container exits immediately after startup."

    response, latency = generate_response(prompt)

    print("\nPROMPT:\n")
    print(prompt)

    print("\nRESPONSE:\n")
    print(response)

    print(f"\nLatency: {latency} seconds")


if __name__ == "__main__":
    main()
