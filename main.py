from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw

# Global variable to hold the current PIL image
current_image = None

generator = torch.Generator(device=torch.device("cuda"))

latents = None

def reset_latents():
    global latents
    latents = None
    generate_image()

def get_latents():
    global latents
    global generator
    if latents is not None:
        return latents
    # Get a new random seed, store it and use it as the generator state
    seed = generator.seed()
    generator = generator.manual_seed(seed)
    
    image_latents = torch.randn(
        (1, pipe.unet.in_channels, 512 // 8, 512 // 8),
        generator=generator,
        device=torch.device("cuda"),
        dtype=pipe.dtype,  # Convert to the same data type as the pipe model
    )
    latents = image_latents if latents is None else torch.cat((latents, image_latents))
    return latents
    
def generate_image():
    global current_image
    prompt = prompt_entry.get()
    result = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, latents=get_latents())
    image = result.images[0]
    current_image = image  # Store the PIL image

    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo
    # Enable the download button
    download_button.config(state="normal")

def on_type(event=None):
    global typing_delay
    if typing_delay is not None:
        window.after_cancel(typing_delay)
    typing_delay = window.after(500, generate_image)

def create_placeholder():
    # Create an image with the same dimensions as the generated images
    placeholder = Image.new('RGB', (512, 512), color = (255, 255, 255))
    draw = ImageDraw.Draw(placeholder)
    # Add text to the placeholder
    text = "Type to generate an image"
    draw.text((10, 240), text, fill=(0, 0, 0))

    return ImageTk.PhotoImage(placeholder)

def save_image():
    global current_image
    if current_image:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            current_image.save(file_path)

window = tk.Tk()
window.title("SDXL-Turbo Image Generator")

# Create a frame for the entry and button
entry_frame = tk.Frame(window)
entry_frame.pack(fill='x', expand=True)

prompt_entry = tk.Entry(entry_frame)
prompt_entry.pack(side='left', fill='x', expand=True)  # Set entry to expand
prompt_entry.bind("<KeyRelease>", on_type)

# Load the image for the button
dice_img = Image.open("dicebutton.png")
dice_img = dice_img.resize((20, 20))  # Resize if necessary
dice_photo = ImageTk.PhotoImage(dice_img)

# Create the button with the image
dice_button = tk.Button(entry_frame, image=dice_photo, command=generate_image)
dice_button.pack(side='right')
dice_button.image = dice_photo  # Keep a reference to avoid garbage collection

# reset latents with dice button
dice_button.bind("<Button-1>", lambda e: reset_latents())

floppy_img = Image.open("floppy.png")
floppy_img = floppy_img.resize((20, 20))  # Resize if necessary
floppy_photo = ImageTk.PhotoImage(floppy_img)

# Add a download button
download_button = tk.Button(entry_frame, image=floppy_photo, command=save_image, state="disabled")
download_button.pack(side='right')
download_button.image = floppy_photo  # Keep a reference to avoid garbage collection

# Initially, display the placeholder
placeholder_image = create_placeholder()
placeholder_shown = True
image_label = tk.Label(window, image=placeholder_image)
image_label.pack()

typing_delay = None

window.mainloop()