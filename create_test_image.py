from PIL import Image, ImageDraw, ImageFont
import os

# Create a simple image with text
img = Image.new('RGB', (400, 200), color=(255, 255, 255))
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("Arial", 20)
except:
    font = ImageFont.load_default()

d.rectangle([50, 50, 350, 150], outline='black', width=2)
d.text((100, 80), "Test Image for RAG System", fill='black', font=font)
d.text((120, 120), "2025-11-01", fill='black', font=font)

# Save the image
test_image_path = "test_image.png"
img.save(test_image_path)
print(f"Test image created at: {os.path.abspath(test_image_path)}")
