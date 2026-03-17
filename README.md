# Qwen 3 8B on Intel NPU with Open﻿Vino-genai

This project demonstrates running **Qwen 3 (8B)** on an **Intel Ultra Series 1 & 2 NPU** using **OpenVINO GenAI**.  
The model is optimized with **INT4 quantization**, making it faster ⚡ and smaller 🚀.

---

## ✨ Features
- Runs **Qwen LLM** locally on Intel NPU
- Uses **OpenVINO GenAI**
- **INT4 quantization** for efficient inference
- Works **offline** (no GPU required)
- Built with **Hugging Face Transformers + Optimum-Intel**

---

## 📦 Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/balaragavan2007/Qwen_on_Intel_NPU.git
cd Qwen_on_Intel_NPU
python -m venv llm
llm\Scripts\activate
pip install -r requirements.txt
```

---

## ⚡ Model Download

Download the model locally by the following command

```bash
hf download OpenVINO/Qwen3-8B-int4-cw-ov --local-dir Qwen3-8B-Instruct-NPU-Model
```
- This will create a folder Qwen3-8B-NPU-Model with the OpenVINO-optimized model.

---

## ▶️ Usage

- Run inference with:

```bash
python run.py
```

---

## Performance

Running the Qwen 3-8B model on my [ASUS Vivobook S16 OLED with Intel Core Ultra 5 125H], I achieved the following performance:

| Device | Performance      |
|--------|------------------|
| CPU    | [7.8] tok/s |
| GPU    | [13.11] tok/s |
| NPU    | [8.77] tok/s |


---

## 📸 Demo
<img width="2988" height="1455" alt="image" src="https://github.com/user-attachments/assets/c437f303-0cc2-4c8f-aab3-cb6061e33e65" />

---


