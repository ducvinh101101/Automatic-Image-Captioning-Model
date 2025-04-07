# Import necessary libraries
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model and processor
model_dir = "blip_caption_model"  # Thay bằng đường dẫn thực tế
processor = BlipProcessor.from_pretrained(model_dir)
model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)

# Function to generate caption
def predict_caption(model, processor, image_path, max_length=50):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=max_length)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Example usage
image_path = "image/gai.jpg"
caption = predict_caption(model, processor, image_path)
print(f"Generated caption: {caption}")