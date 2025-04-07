# Import modules
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader

plt.rcParams['font.size'] = 12
sns.set_style("dark")

# Import Dataset
image_path = '/kaggle/input/flickr8k/Images'
data = pd.read_csv("/kaggle/input/flickr8k/captions.txt")


def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    n = 0
    for i in range(15):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        image = Image.open(os.path.join(image_path, temp_df.image[i])).convert('RGB')
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))
        plt.axis("off")


display_images(data.sample(15))


# Text Preprocessing
def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]", ""))
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+", " "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    return data


data = text_preprocessing(data)

# Split dataset
images = data['image'].unique().tolist()
nimages = len(images)
split_index = round(0.85 * nimages)
train_images = images[:split_index]
val_images = images[split_index:]

train = data[data['image'].isin(train_images)]
test = data[data['image'].isin(val_images)]

train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)


# Custom Dataset for BLIP
class Flickr8kDataset(Dataset):
    def __init__(self, df, image_dir, processor):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_file = self.df['image'][idx]
        caption = self.df['caption'][idx]

        # Load image
        image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')

        # Process image and caption
        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding=True)

        # Remove batch dimension
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs


# Custom collate function to handle variable-length inputs
def custom_collate_fn(batch):
    keys = batch[0].keys()
    batched = {key: [] for key in keys}

    for item in batch:
        for key in keys:
            batched[key].append(item[key])

    # Pad sequences and stack tensors
    batched['input_ids'] = torch.nn.utils.rnn.pad_sequence(batched['input_ids'], batch_first=True,
                                                           padding_value=processor.tokenizer.pad_token_id)
    batched['attention_mask'] = torch.nn.utils.rnn.pad_sequence(batched['attention_mask'], batch_first=True,
                                                                padding_value=0)
    batched['pixel_values'] = torch.stack(batched['pixel_values'])  # Pixel values are uniform

    return batched


# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create datasets and dataloaders with custom collate function
train_dataset = Flickr8kDataset(train, image_path, processor)
test_dataset = Flickr8kDataset(test, image_path, processor)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 5

# Training loop
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        # Move batch to device
        inputs = {key: val.to(device) for key, val in batch.items()}

        # Forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Validation"):
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(test_dataloader)
    print(f"Average validation loss: {avg_val_loss:.4f}")
    model.train()

# Save the model
model.save_pretrained("blip_caption_model")
processor.save_pretrained("blip_caption_model")
print("Model saved as 'blip_caption_model'")


# Caption Generation Function
def predict_caption(model, processor, image_path, max_length=50):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=max_length)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption


# Test on samples
samples = test.sample(20)
samples.reset_index(drop=True, inplace=True)

for index, record in samples.iterrows():
    image_file = os.path.join(image_path, record['image'])
    caption = predict_caption(model, processor, image_file)
    samples.loc[index, 'caption'] = caption

display_images(samples)