import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import PeftModel

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

ADAPTER_PATH = "outputs/adapters/it-support-qlora-adapter"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    return model, tokenizer


model, tokenizer = load_model()


def generate_response(prompt):

    formatted_prompt = f"""
[INST]
{prompt}
[/INST]
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=200, temperature=0.3, do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


if __name__ == "__main__":

    prompt = "Docker container exits immediately after startup."

    response = generate_response(prompt)

    print("\n" + "=" * 50)
    print("FINE-TUNED RESPONSE")
    print("=" * 50)

    print(response)
