import torch
import transformers
import bitsandbytes as bnb

print("=" * 50)
print("TORCH INFO")
print("=" * 50)

print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {total_vram:.2f} GB")

print("\n" + "=" * 50)
print("TRANSFORMERS INFO")
print("=" * 50)

print("Transformers Version:", transformers.__version__)

print("\n" + "=" * 50)
print("BITSANDBYTES INFO")
print("=" * 50)

print("BitsAndBytes Loaded Successfully")
