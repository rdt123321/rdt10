# Local AI Image Generator

This repository contains a **fully local AI image generator** you can run on your laptop.

It includes:
- A simple web UI (Gradio)
- Local image generation using Hugging Face Diffusers
- Automatic image saving to `outputs/`
- One-click ZIP export button for generated PNG files
- A requirements file and setup steps

## 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Run the app

```bash
python app.py
```

Then open the local URL shown in your terminal (usually `http://127.0.0.1:7860`).

## 4) Generate images

- Enter a prompt
- (Optional) Enter a negative prompt
- Choose width/height, guidance scale, steps, and seed
- Click **Generate**
- Click **Create Download ZIP** to bundle all generated PNGs and download them from the UI

Generated images are saved in `outputs/` and shown in the UI.

---

## Notes

- Default model: `stabilityai/sd-turbo`
- First run will download model weights (can take a while).
- GPU is recommended, but CPU mode is supported (it will be slower).

## Optional model override

You can change the model with an environment variable:

```bash
MODEL_ID=runwayml/stable-diffusion-v1-5 python app.py
```
