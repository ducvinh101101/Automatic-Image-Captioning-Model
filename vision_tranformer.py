import numpy as np
import pandas as pd
import os
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from textwrap import wrap


# Đường dẫn
BASE_DIR = ''
image_path = os.path.join(BASE_DIR, 'image')

# Load dữ liệu captions
data = pd.read_csv(os.path.join(BASE_DIR, "captions.txt"), sep=',')


# Tiền xử lý captions
def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]", ""))
    data['caption'] = data['caption'].apply(lambda x: x.replace(r"\s+", " "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data


data = text_preprocessing(data)
captions = data['caption'].tolist()

# Tạo tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)
print(f"Vocab size: {vocab_size}, Max length: {max_length}")

# Load hoặc khởi tạo feature extractor
feature_extractor = None


from transformers import ViTImageProcessor, ViTModel

vit_model = None  # Khởi tạo biến toàn cục

def get_feature_extractor():
    global vit_model
    if vit_model is None:
        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    return vit_model



from PIL import Image
import numpy as np

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


def extract_feature(image_path, image_name, model):
    img = Image.open(os.path.join(image_path, image_name)).convert("RGB")

    # Tiền xử lý ảnh thành tensor
    inputs = feature_extractor(images=img, return_tensors="pt")  # Chuyển ảnh thành tensor

    # Trích xuất đặc trưng
    with torch.no_grad():
        outputs = model(**inputs)

    feature = outputs.pooler_output  # Lấy output từ lớp pooler
    feature = feature.numpy()  # Chuyển sang NumPy array nếu cần dùng với TensorFlow/Keras

    return feature


# Load đặc trưng ảnh từ file (nếu có) hoặc trích xuất cho ảnh tùy chỉnh
features_path = os.path.join(BASE_DIR, 'features.npy')
if os.path.exists(features_path):
    features = np.load(features_path, allow_pickle=True).item()
else:
    features = {}

# Load model đã train
model_path = os.path.join(BASE_DIR, 'best_vit_caption_model.keras')
caption_model = load_model(model_path)
print("Model loaded successfully!")


# Hàm chuyển index thành từ
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Hàm dự đoán caption
def predict_caption(model, image_name, tokenizer, max_length, features, image_path):
    if image_name not in features:
        fe = get_feature_extractor()
        features[image_name] = extract_feature(image_path, image_name, fe)
    feature = features[image_name]
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text


# Hàm hiển thị ảnh và caption
def display_image_and_caption(image_path, image_name, caption, actual_captions=None):
    img = load_img(os.path.join(image_path, image_name))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')

    title = f"Predicted Caption: {caption}"
    if actual_captions:
        wrapped_captions = [wrap(str(caption), 30) for caption in actual_captions]
        title += "\nActual Captions:\n" + "\n".join(["\n".join(lines) for lines in wrapped_captions])
    plt.title(title, fontsize=12)
    plt.show()


# Ví dụ sử dụng
image_name = 'anh-cho-meo.jpg'  # Thay bằng tên ảnh muốn dự đoán

# Dự đoán caption
caption = predict_caption(caption_model, image_name, tokenizer, max_length, features, image_path)

# Lấy các caption thực tế từ data (nếu có)
actual_captions = data[data['image'] == image_name]['caption'].tolist() if image_name in data['image'].values else None

# Hiển thị ảnh và caption
display_image_and_caption(image_path, image_name, caption, actual_captions)