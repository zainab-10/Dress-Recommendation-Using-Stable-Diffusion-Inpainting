from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

def load_image(file_path):
    img = Image.open(file_path)
    return img

# def create_mask(image):
#     # Create a mask with the same dimensions as the input image
#     mask = Image.new("L", image.size, 0)
#     draw = ImageDraw.Draw(mask)
    
#     # Adjust these coordinates to mask the entire dress area
#     draw.polygon([(110, 150), (400, 150), (450, 512), (50, 512)], fill=255)
    
#     return mask

# def preprocess_image(image):
#     # Convert image to tensor and normalize
#     image = image.convert("RGB")
#     image = image.resize((512, 512))
#     image = np.array(image) / 255.0  # Normalize to [0, 1]
#     return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(torch.float16)

# def preprocess_mask(mask):
#     # Convert mask to tensor and normalize
#     mask = mask.resize((512, 512))
#     mask = np.array(mask) / 255.0  # Normalize to [0, 1]
#     return torch.tensor(mask).unsqueeze(0).to(torch.float16)

# # Load and preprocess the input image
# img_path = "/home/user/zainab/vfuser/stable_diffusion/sd_inpaint/image1.png"
# image = load_image(img_path)

# # Create a mask image that covers the entire dress area
# mask_image = create_mask(image)

# # Preprocess images and mask
# image_tensor = preprocess_image(image)
# mask_tensor = preprocess_mask(mask_image)

# # Initialize the inpainting pipeline
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")

# # Define your prompt to replace the dress with a fashionable red frock
# prompt = "Recomend a beautiful jeans shirt"

# # Perform inpainting
# try:
#     output_image = pipe(prompt=prompt, image=image_tensor, mask_image=mask_tensor).images[0]
#     output_image.save("inpaint_dress.png")
    
#     # Display the original image, mask, and output image in a row
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))

#     # Display original image
#     axes[0].imshow(image)
#     axes[0].set_title("Original Image")
#     axes[0].axis("off")
    
#     # Display output image
#     output_image_pil = Image.open("inpaint_red_frock.png")
#     axes[1].imshow(output_image_pil)
#     axes[1].set_title("Output Image")
#     axes[1].axis("off")
    
#     # Save the figure as an image
#     plt.savefig("inpaint_dress_recomend2.png", bbox_inches='tight', pad_inches=0.1)

#     # Show the plot
#     plt.show()

# except RuntimeError as e:
#     print(f"Error during inpainting: {e}")
#     print(f"Image tensor shape: {image_tensor.shape}")
#     print(f"Mask tensor shape: {mask_tensor.shape}")

# def create_mask(image):
#     # Create a mask with the same dimensions as the input image
#     mask = Image.new("L", image.size, 0)
#     draw = ImageDraw.Draw(mask)
    
#     # Adjust these coordinates to mask the shirt area of the child
#     # These coordinates should approximately cover the shirt
#     draw.polygon([(120, 100), (380, 100), (450, 400), (70, 400)], fill=255)
    
#     return mask

# # Other functions remain the same

# # Load and preprocess the input image
# img_path = "/home/user/zainab/vfuser/stable_diffusion/sd_inpaint/images.jpg"
# image = load_image(img_path)

# # Create a mask image that covers the entire area where you want the jeans shirt to appear
# mask_image = create_mask(image)

# # Preprocess images and mask
# image_tensor = preprocess_image(image)
# mask_tensor = preprocess_mask(mask_image)

# # Initialize the inpainting pipeline
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")

# # Define your prompt to replace the shirt with a different one
# prompt = "A trendy, fashionable red jacket"

# # Perform inpainting
# try:
#     output_image = pipe(prompt=prompt, image=image_tensor, mask_image=mask_tensor).images[0]
#     output_image.save("inpaint_new_shirt.png")
    
#     # Display the original image, mask, and output image in a row
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))

#     # Display original image
#     axes[0].imshow(image)
#     axes[0].set_title("Original Image")
#     axes[0].axis("off")
    
    
#     # Display output image
#     output_image_pil = Image.open("inpaint_new_shirt.png")
#     axes[1].imshow(output_image_pil)
#     axes[1].set_title("Output Image")
#     axes[1].axis("off")
    
#     # Save the figure as an image
#     plt.savefig("inpaint_recommendation_final.png", bbox_inches='tight', pad_inches=0.1)

#     # Show the plot
#     plt.show()

# except RuntimeError as e:
#     print(f"Error during inpainting: {e}")
#     print(f"Image tensor shape: {image_tensor.shape}")
#     print(f"Mask tensor shape: {mask_tensor.shape}")

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

def load_image(file_path):
    img = Image.open(file_path)
    return img

def create_mask(image):
    # Create a mask with the same dimensions as the input image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Adjust these coordinates to more accurately mask the shirt area
    # Example coordinates for a shirt - modify based on your image
    draw.polygon([(100, 60), (260, 60), (290, 160), (80, 300)], fill=255)  # Adjust these coordinates

    return mask

def preprocess_image(image):
    # Convert image to tensor and normalize
    image = image.convert("RGB")
    image = image.resize((512, 512))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(torch.float16)

def preprocess_mask(mask):
    # Convert mask to tensor and normalize
    mask = mask.resize((512, 512))
    mask = np.array(mask) / 255.0  # Normalize to [0, 1]
    return torch.tensor(mask).unsqueeze(0).to(torch.float16)

# Load the uploaded image
img_path = "/home/user/zainab/vfuser/stable_diffusion/sd_inpaint/images.jpg"  # Path to the uploaded image
image = load_image(img_path)

# Display the loaded image to assist in creating the mask
plt.imshow(image)
plt.title("Uploaded Image")
plt.axis("off")
plt.show()

# Create a mask image that covers the shirt area
mask_image = create_mask(image)

# Preprocess images and mask
image_tensor = preprocess_image(image)
mask_tensor = preprocess_mask(mask_image)

# Initialize the inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# Define your prompt to replace the shirt with a different one
prompt = "wear a dark blue jacket over shirt"

# Perform inpainting
try:
    output_image = pipe(prompt=prompt, image=image_tensor, mask_image=mask_tensor).images[0]
    output_image.save("inpaint_new_shirt.png")
    
    # Display the original image and output image
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Display original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Display output image
    output_image_pil = Image.open("inpaint_new_shirt.png")
    axes[1].imshow(output_image_pil)
    axes[1].set_title("Output Image")
    axes[1].axis("off")
    
    # Save the figure as an image

    plt.savefig("inpaint_recommendation_final3.png", bbox_inches='tight', pad_inches=0.1)

    # Show the plot
    plt.show()

except RuntimeError as e:
    print(f"Error during inpainting: {e}")
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Mask tensor shape: {mask_tensor.shape}")
