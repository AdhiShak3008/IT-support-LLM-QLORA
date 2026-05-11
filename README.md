# IT Support LLM using QLoRA Fine-Tuning

## Overview

A production-style domain-specific LLM engineering project focused on fine-tuning Mistral-7B-Instruct using QLoRA for IT support and troubleshooting workflows.

This project implements a complete fine-tuning pipeline including dataset preprocessing, 4-bit quantized LoRA training, benchmark evaluation, adapter-based inference, and Gradio deployment for real-time troubleshooting generation.

The model is specialized for infrastructure and operational support scenarios such as:

- Docker failures
- Kubernetes pod crashes
- VPN connectivity issues
- Database authentication failures
- DNS resolution problems

---

## Key Features

- QLoRA-based 4-bit fine-tuning on Mistral-7B-Instruct
- LoRA adapter training using PEFT
- Quantized GPU-efficient inference pipeline
- Domain-specific instruction tuning for IT operations
- Benchmark evaluation and response comparison framework
- Base model vs fine-tuned model analysis
- Modular training, inference, and evaluation architecture
- Gradio-powered troubleshooting assistant UI
- Config-driven training workflow
- Reproducible dataset preprocessing pipeline

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- bitsandbytes
- Accelerate
- Datasets
- Gradio
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## Training Details

| Component          | Value                     |
| ------------------ | ------------------------- |
| Base Model         | Mistral-7B-Instruct-v0.2  |
| Fine-Tuning Method | QLoRA                     |
| Quantization       | 4-bit NF4                 |
| Adapter Method     | LoRA                      |
| GPU Used           | NVIDIA Tesla T4           |
| Frameworks         | Transformers + PEFT + TRL |

---

## Project Architecture

```text
Dataset
   ↓
Formatting Pipeline
   ↓
QLoRA Fine-Tuning
   ↓
LoRA Adapters
   ↓
Inference Pipeline
   ↓
Benchmark Evaluation
   ↓
Gradio Deployment
```

---

## Project Structure

```text
IT-support-llm-qlora/
│
├── configs/
│
├── data/
│   ├── processed/
│   └── synthetic/
│
├── outputs/
│   └── adapters/
│
├── src/
│   ├── data/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── utils/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Example Capabilities

### Sample Prompt

```text
User cannot connect to PostgreSQL database after password reset.
```

### Fine-Tuned Response

```text
1. Verify that the new password is entered correctly.
2. Check PostgreSQL authentication configuration.
3. Validate pg_hba.conf rules.
4. Ensure database service is running.
5. Test connectivity using alternate client tools.
6. Verify firewall and VPN routing policies.
```

---

## Future Improvements

- Expand training dataset with larger IT support scenarios
- Add automated benchmark scoring metrics
- Deploy public inference demo using Hugging Face Spaces
- Improve response consistency with larger fine-tuning datasets

---
