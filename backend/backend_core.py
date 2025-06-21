import os
import logging
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
logging.basicConfig(level=logging.WARNING)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
)
import traceback
import gradio as gr
import time
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    print("Intel Extension for PyTorch (IPEX) detected and ready.")
    print(f"Device {torch.xpu.get_device_name(0)} will be used.")
except ImportError:
    IPEX_AVAILABLE = False
    print("FATAL: Intel Extension for PyTorch (IPEX) is not installed.")

pipe = None
current_device_of_pipe = "cpu" 
MODELS_DIR = "model" 
is_pipe_optimized = False
original_text_encoder_layers = 12 

def get_available_samplers():
    return [
        "DPM++ 2M Karras",
        "Euler a",
        "LCM",
    ]

def get_available_models():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        return [f for f in os.listdir(MODELS_DIR) if f.endswith('.safetensors')]
    except Exception as e:
        print(f"Error scanning model folder: {e}"); return []

def load_checkpoint_model(checkpoint_filename, progress=gr.Progress(track_tqdm=True)):
    global pipe, current_device_of_pipe, is_pipe_optimized, original_text_encoder_layers
    if not checkpoint_filename:
        return "⚠️ Please select a model from the list."
    full_model_path = os.path.join(MODELS_DIR, checkpoint_filename)    
    try:
        base_model_id = "runwayml/stable-diffusion-v1-5"
        progress(0.1, desc="Loading text components...")
        tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer", cache_dir=MODELS_DIR)
        text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder", cache_dir=MODELS_DIR)
        progress(0.7, desc=f"Loading visual components...")
        
        pipe = StableDiffusionPipeline.from_single_file(
            full_model_path, 
            torch_dtype=torch.float32, 
            use_safetensors=True, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder,
            safety_checker=None,
        )
        
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            original_text_encoder_layers = pipe.text_encoder.config.num_hidden_layers

        current_device_of_pipe = "cpu"; is_pipe_optimized = False 
        progress(1, desc="Model loaded successfully!")
        return f"✅ Model **'{os.path.basename(checkpoint_filename)}'** loaded successfully!"
    except Exception as e:
        pipe = None; print(f"❌ Failed to load checkpoint: {traceback.format_exc()}"); return f"❌ Failed to load model. Error: {e}"

def generate_image(prompt, negative_prompt, steps, guidance_scale, seed, width, height, clip_skip, sampler_name, progress=gr.Progress(track_tqdm=True)):
    global pipe, current_device_of_pipe, is_pipe_optimized, original_text_encoder_layers
    if pipe is None: return None, "❌ Model not loaded yet."
    
    try:
        start_time = time.time()
        
        print(f"Using sampler: {sampler_name}")
        if sampler_name == "DPM++ 2M Karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        elif sampler_name == "Euler a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sampler_name == "LCM":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        target_device = "xpu" if IPEX_AVAILABLE else "cpu"
        if current_device_of_pipe != target_device:
            progress(0, desc=f"Moving model to {target_device} device...")
            pipe.to(target_device); current_device_of_pipe = target_device
        
        if IPEX_AVAILABLE and not is_pipe_optimized:
            progress(0, desc="Optimizing model with IPEX...")
            pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True)
            is_pipe_optimized = True; print("✅ Model optimized successfully.")

        final_clip_skip = max(1, int(clip_skip))
        num_layers_to_use = original_text_encoder_layers - (final_clip_skip - 1)
        if pipe.text_encoder.config.num_hidden_layers != num_layers_to_use:
             pipe.text_encoder.config.num_hidden_layers = num_layers_to_use

        def progress_callback(pipe_obj, step, t, kwargs): progress((step + 1)/int(steps), f"Step {step+1}/{int(steps)}"); return kwargs

        used_seed = seed
        if int(seed) == -1: used_seed = torch.Generator(device="cpu").seed()
        generator = torch.Generator(device="cpu").manual_seed(int(used_seed))
        
        with torch.no_grad(), torch.autocast(device_type=target_device, dtype=torch.bfloat16):
            image = pipe(
                prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale), generator=generator,
                width=int(width), height=int(height), callback_on_step_end=progress_callback,
            ).images[0]
        
        if pipe.text_encoder.config.num_hidden_layers != original_text_encoder_layers:
            pipe.text_encoder.config.num_hidden_layers = original_text_encoder_layers

        end_time = time.time()
        duration = end_time - start_time
        details = (f"**Success!** Generation time: **{duration:.2f} seconds**\n\n"
                   f"**Seed:** `{int(used_seed)}`\n"
                   f"**Sampler:** `{sampler_name}`\n"
                   f"**Size:** `{width}x{height}`\n"
                   f"**Steps:** `{steps}`, **CFG:** `{guidance_scale}`, **Clip Skip:** `{final_clip_skip}`\n\n"
                   f"**Prompt:** {prompt}\n\n"
                   f"**Negative Prompt:** {negative_prompt}")
        return image, details

    except Exception as e:
        if pipe is not None and hasattr(pipe, 'text_encoder'):
            if pipe.text_encoder.config.num_hidden_layers != original_text_encoder_layers:
                pipe.text_encoder.config.num_hidden_layers = original_text_encoder_layers
        print(f"FATAL ERROR: {traceback.format_exc()}"); return None, f"❌ Error: {e}"