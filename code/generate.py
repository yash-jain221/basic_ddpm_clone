import streamlit as st
import torch
from torchvision.utils import make_grid
import torchvision.transforms as T
import time
import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import os
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

IMG_SIZE = 64  # or 128

st.set_page_config(layout="wide")
st.title("🌀 DDPM Generative Playground")

# Sidebar controls
st.sidebar.header("Generation Settings")
num_inference_steps = st.sidebar.slider("Inference Steps", 20, 200, 75, step=5)
seed = st.sidebar.slider("Random Seed", 0, 9999, 42)
prompt = st.text_input("Prompt")
sampler = st.sidebar.selectbox("Sampler", options=["ddpm"])  # Add more as supported
replay = st.sidebar.button("Replay Animation")
save = st.sidebar.button("Save Final Image")
generate = st.button("Generate")

sampler = "ddpm"
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.7
output = st.empty() 

if "decoded_images" not in st.session_state:
    st.session_state.decoded_images = []

if generate:
    # torch.manual_seed(seed)

    st.write("Generating image...")
    st.session_state.generation_complete = False
    tokenizer = CLIPTokenizer("C:/Users/Yash/Desktop/Projects/stable_diffusion/data/tokenizer_vocab.json", merges_file="C:/Users/Yash/Desktop/Projects/stable_diffusion/data/tokenizer_merges.txt")
    model_file = "C:/Users/Yash/Desktop/Projects/stable_diffusion/data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

    ## TEXT TO IMAGE

    # prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    # prompt = "colorful texture like art, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    # prompt = "A modern glass skyscraper of 50 floors, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    uncond_prompt = ""  # Also known as negative prompt
    do_cfg = True
    cfg_scale = 8  # min: 1, max: 14

    ## IMAGE TO IMAGE

    input_image = None
    # Comment to disable image to image
    # image_path = "../images/dog.jpg"

    images = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer
    )
    i=0
    
    for img_tensor in images:
        # print(img_tensor)
        # logging.info("My model shape: %s", img_tensor.shape) ## [N, H, W, C]
        # logging.info(f"Step {i} - min: {img_tensor.min().item()}, max: {img_tensor.max().item()}")

        img_tensor = img_tensor.permute(0, 3, 1, 2)
        # logging.info("My model shape: %s", img_tensor.shape) ## [N, C, H, W]
        grid = make_grid(img_tensor, nrow=1)
        pil_img = T.ToPILImage()(grid)
        st.session_state.decoded_images.append(pil_img)
        output.image(pil_img, caption=f"Step image {i}", use_container_width=True)
        time.sleep(0.5)
        i+=1
    st.session_state.generation_complete = True

if replay and st.session_state.decoded_images:
    for idx, img in enumerate(st.session_state.decoded_images):
        output.image(img, caption=f"Step {idx}", use_container_width=True)
        time.sleep(0.1)

# ---- SLIDER VIEW ----
if st.session_state.decoded_images:
    step = st.slider("View Step", 0, len(st.session_state.decoded_images) - 1)
    output.image(st.session_state.decoded_images[step], caption=f"Step {step}")

# ---- SAVE IMAGE ----
if save and st.session_state.generation_complete:
    final_img = st.session_state.decoded_images[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"final_output_{timestamp}.png"
    final_img.save(save_path)
    st.success(f"Final image saved as {save_path}")