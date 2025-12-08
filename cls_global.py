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
    Image size is automatically calculated based on text content.
    
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
        Gaussian blur radius (0 = no blur, 2 = mild, 3 = moderate, 4 = strong)
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
    # Create a temporary image to measure text
    dummy_img = Image.new('RGBA', (1, 1), bg)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), content, font=font)
    
    # Calculate actual text dimensions
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Add dynamic padding based on font size
    padding_horizontal = max(10, int(font_size * 0.5))
    padding_vertical = max(10, int(font_size * 0.3))
    
    # Calculate image dimensions based on text
    img_width = text_width + padding_horizontal * 2
    img_height = text_height + padding_vertical * 2
    
    # Ensure minimum size
    img_width = max(img_width, 50)
    img_height = max(img_height, 30)
    
    # Create image with background color
    image = Image.new('RGBA', (img_width, img_height), bg)
    draw = ImageDraw.Draw(image)
    
    # Calculate text position to center it
    text_x = padding_horizontal - bbox[0]
    text_y = padding_vertical - bbox[1]
    text_position = (text_x, text_y)
    
    # Draw text in black
    draw.text(text_position, content, font=font, fill=(0, 0, 0, 255))
    
    # Convert to RGB
    image = image.convert('RGB')
    
    # Apply noise based on noise_level
    if noise_level != 'none':
        image = add_noise(image, noise_level)
    
    # Apply blur based on blur_level
    if blur_level > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_level))
    
    # Save image
    output_path = os.path.join(output_dir, f"{index}.png")
    image.save(output_path)


def add_noise(image, noise_level):
    """
    Add random noise to an image.
    Noise levels: 'low', 'medium', 'high'
    
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
        'low': 5,      # Subtle noise
        'medium': 15,  # Moderate noise
        'high': 30     # Strong noise
    }
    
    noise_intensity = noise_intensity_map.get(noise_level, 0)
    
    if noise_intensity > 0:
        # Generate random noise
        noise = np.random.randint(
            -noise_intensity, 
            noise_intensity + 1, 
            img_array.shape, 
            dtype=np.int16
        )
        
        # Add noise to image
        noisy_img = img_array.astype(np.int16) + noise
        
        # Clip values to valid range [0, 255]
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    return image


def test_generation():
    """
    Test function to verify the image generation works correctly.
    Tests all noise and blur combinations.
    """
    print("Testing image generation with different parameters...")
    
    # Test with different text lengths
    test_texts = [
        "ខ",           # Single character
        "តេស្ត",       # Short word
        "សូមស្វាគមន៍",  # Medium word
        "នេះជាការសាកល្បងវែងបន្តិច"  # Longer sentence
    ]
    
    # Test parameters
    noise_levels = ["none", "low"]
    blur_levels = [0]
    
    test_index = 1
    
    for text in test_texts:
        for noise in noise_levels:
            for blur in blur_levels:
                print(f"Generating test {test_index}: text='{text[:10]}...', noise={noise}, blur={blur}")
                
                gen_khmer_text_image(
                    index=test_index,
                    content=text,
                    data_type="test",
                    bg=(255, 255, 255, 255),  # White background
                    noise_level=noise,
                    blur_level=blur,
                    font_path="fonts/KhmerOS.ttf",  # Adjust path as needed
                    font_size=16,
                    data_folder="test_output"
                )
                
                test_index += 1
    
    print(f"\n{test_index - 1} test images generated in test_output/test/")
    print("Check the images to verify different sizes, noise, and blur levels!")


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_generation()