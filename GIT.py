from transformers import GitProcessor, GitForCausalLM
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình và processor từ Hugging Face
model_dir = "git_caption_model"  # Use a valid Hugging Face model ID
processor = GitProcessor.from_pretrained(model_dir)
model = GitForCausalLM.from_pretrained(model_dir).to(device)

# Hàm dự đoán caption cho ảnh
def predict_caption(model, processor, image_path, max_length=50):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=max_length)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption


image_path = "image/download.jpg"
caption = predict_caption(model, processor, image_path)
img = Image.open(image_path)
img.show()
print(f"Generated caption: {caption}")