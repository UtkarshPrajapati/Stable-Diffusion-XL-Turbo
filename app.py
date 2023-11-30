from diffusers import StableDiffusionXLPipeline
import torch
import os
import time
import random
import tomesd
import streamlit as st

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


st.title("SDXL Turbo")

if not os.path.exists('outputs'):
    os.makedirs('outputs')

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionXLPipeline.from_pretrained("./sdxl-turbo/",torch_dtype=torch.float16,variant="fp16")
    pipeline = pipeline.to(device)
    pipeline.upcast_vae()
    pipeline.enable_vae_tiling()
    tomesd.apply_patch(pipeline,ratio=0.5)
    return pipeline

pipe = load_model()

prompt = st.sidebar.text_area("Enter a prompt:", "A cinematic shot of a baby raccoon wearing an intricate Italian priest robe.", height=1)
num_inference_steps = st.sidebar.slider("Number of inference steps", 1, 10, 1)
guidance_scale = st.sidebar.slider("Guidance scale", 0.0, 2.0, 0.0)

if st.sidebar.button('Generate Image'):
    with st.spinner('Generating image...'):
        # Generate the image
        image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

        # Generate a unique filename
        timestamp = int(time.time())
        random_number = random.randint(0, 99999)
        filename = f"outputs/image_{timestamp}_{random_number}.png"

        # Save the image
        image.save(filename)

        # Display the image
        st.image(filename, caption='Generated Image')