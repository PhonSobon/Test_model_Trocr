import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def gen_khmer_text_image(
    index, 
    content, 
    data_type, 
    bg, 
    noise_level, 
    blur_level, 
    font_path, 
    font_size, 
    data_folder
):
    """
    Generate a text image with various augmentations for OCR training.
    
    Parameters:
    -----------
    index : int
        Index number for the image filename
    content : str
        Text content to render
    data_type : str
        Type of data: 'train', 'valid', or 'test'
    bg : tuple
        Background color (R, G, B, A)
    noise_level : str
        Noise level: 'none', 'low', 'medium', 'high'
    blur_level : int
        Gaussian blur radius (0 = no blur)
    font_path : str
        Path to the font file
    font_size : int
        Font size in points
    data_folder : str
        Root folder for saving images
    """
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(data_folder, data_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return
    
    # Calculate text size using textbbox
    dummy_img = Image.new('RGBA', (1, 1), bg)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), content, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Add padding
    padding = 20
    img_width = text_width + padding * 2
    img_height = text_height + padding * 2
    
    # Create image with background color
    image = Image.new('RGBA', (img_width, img_height), bg)
    draw = ImageDraw.Draw(image)
    
    # Draw text in black
    text_position = (padding, padding)
    draw.text(text_position, content, font=font, fill=(0, 0, 0, 255))
    
    # Convert to RGB
    image = image.convert('RGB')
    
    # Apply noise
    if noise_level != 'none':
        image = add_noise(image, noise_level)
    
    # Apply blur
    if blur_level > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_level))
    
    # Save image
    output_path = os.path.join(output_dir, f"{index}.png")
    image.save(output_path)


def add_noise(image, noise_level):
    """
    Add random noise to an image.
    
    Parameters:
    -----------
    image : PIL.Image
        Input image
    noise_level : str
        Noise intensity: 'low', 'medium', 'high'
    
    Returns:
    --------
    PIL.Image
        Image with added noise
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Define noise intensity based on level
    noise_intensity_map = {
        'low': 5,
        'medium': 15,
        'high': 30
    }
    
    noise_intensity = noise_intensity_map.get(noise_level, 0)
    
    if noise_intensity > 0:
        # Generate random noise
        noise = np.random.randint(-noise_intensity, noise_intensity + 1, img_array.shape, dtype=np.int16)
        
        # Add noise to image
        noisy_img = img_array.astype(np.int16) + noise
        
        # Clip values to valid range [0, 255]
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    return image


def test_generation():
    """
    Test function to verify the image generation works correctly.
    """
    print("Testing image generation...")
    
    # Test parameters
    test_content = "តេស្ត"  # Khmer text for "test"
    
    gen_khmer_text_image(
        index=1,
        content=test_content,
        data_type="test",
        bg=(255, 255, 255, 255),  # White background
        noise_level="medium",
        blur_level=2,
        font_path="fonts/KhmerOS.ttf",  # Make sure this font exists
        font_size=16,
        data_folder="test_output"
    )
    
    print("Test image generated in test_output/test/1.png")


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_generation()