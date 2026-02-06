from __future__ import annotations

from pathlib import Path

import gradio as gr

from image_generator.engine import GenerationConfig, ImageGenerator, create_outputs_zip


generator = ImageGenerator()


def generate_image(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
):
    config = GenerationConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    path = generator.generate(config)
    return str(path), f"Saved: {path}"


def download_outputs_zip():
    try:
        zip_path = create_outputs_zip()
        return str(zip_path), f"ZIP ready: {zip_path}"
    except ValueError as exc:
        return None, str(exc)


with gr.Blocks(title="Local AI Image Generator") as demo:
    gr.Markdown("# Local AI Image Generator")
    gr.Markdown("Generate images on your laptop using Stable Diffusion.")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="A cinematic portrait of a futuristic astronaut")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="blurry, low quality")

            with gr.Row():
                width = gr.Slider(256, 1280, step=64, value=1024, label="Width")
                height = gr.Slider(256, 1280, step=64, value=1024, label="Height")

            with gr.Row():
                steps = gr.Slider(1, 50, step=1, value=1, label="Inference Steps")
                guidance = gr.Slider(0.0, 20.0, step=0.1, value=0.0, label="Guidance Scale")

            seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
            btn = gr.Button("Generate", variant="primary")
            zip_btn = gr.Button("Create Download ZIP", variant="secondary")
            status = gr.Markdown()

        with gr.Column(scale=3):
            output_image = gr.Image(label="Generated Image", type="filepath")
            zip_file = gr.File(label="Download outputs ZIP")

    btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, width, height, steps, guidance, seed],
        outputs=[output_image, status],
    )

    zip_btn.click(
        fn=download_outputs_zip,
        inputs=[],
        outputs=[zip_file, status],
    )


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7860)
