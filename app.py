import gradio as gr

try:
    from backend.backend_core import get_available_models, load_checkpoint_model, generate_image, get_available_samplers
except ImportError:
    print("FATAL ERROR: Make sure 'backend_core.py' is in the same folder.")
    exit()

with gr.Blocks(theme='soft', css="footer {display: none !important}") as demo:
    
    with gr.Row():
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 10px;">
                <h1 style="margin: 0; font-size: 2em; font-weight: 700;">🎨 Custom Diffusion Studio (Intel IPEX Edition)</h1>
                <p style="margin: 0; color: #6B7280;">Developed exclusively for IPEX</p>
            </div>
            """
        )
    
    with gr.Row():
        with gr.Group():
            gr.Markdown("<h3 style='text-align: center;'>Load Model from List</h3>")
            with gr.Row():
                checkpoint_dropdown = gr.Dropdown(
                    label="Select Model from 'model' folder",
                    choices=get_available_models(),
                    interactive=True
                )
                refresh_models_button = gr.Button("🔄 Refresh List", variant="secondary")
            
            model_load_status = gr.Textbox(label="Model Status", value="No model loaded yet.", interactive=False)
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("<h3 style='text-align: center;'>Configuration & Prompts</h3>")
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="...")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="...")
                
                sampler_dropdown = gr.Dropdown(
                    label="Sampling Method",
                    choices=get_available_samplers(),
                    value="DPM++ 2M Karras",
                    interactive=True
                )

            with gr.Accordion("🔧 Advanced Options", open=False):
                steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25, step=1)
                guidance_scale = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=20.0, value=7.5, step=0.5)
                seed = gr.Number(label="Seed", value=-1, info="Use -1 for a random result.")
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=1024, value=512, step=64)
                    height = gr.Slider(label="Height", minimum=256, maximum=1024, value=512, step=64)
                clip_skip = gr.Slider(
                    label="Clip Skip", 
                    minimum=1, 
                    maximum=4, 
                    value=1, 
                    step=1, 
                    info="Use value 2 for strange or 'burnt' images. Default: 1."
                )
            
            run_button = gr.Button("Generate Image 🖼️", variant="primary")

        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("<h3 style='text-align: center;'>Result Image & Info</h3>")
                output_image = gr.Image(label="Output", type="pil")
                gen_status_text = gr.Markdown("Status: Awaiting command...")

    checkpoint_dropdown.change(fn=load_checkpoint_model, inputs=[checkpoint_dropdown], outputs=[model_load_status])
    
    def refresh_model_list():
        return gr.update(choices=get_available_models())
        
    refresh_models_button.click(fn=refresh_model_list, inputs=None, outputs=[checkpoint_dropdown])

    run_button.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance_scale, seed, width, height, clip_skip, sampler_dropdown],
        outputs=[output_image, gen_status_text]
    )

if __name__ == "__main__":
    demo.queue().launch(share=False)