from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model

# Đường dẫn
BASE_DIR = ''
image_path = os.path.join(BASE_DIR, 'image')

# Load dữ liệu captions
data = pd.read_csv(os.path.join(BASE_DIR, "captions.txt"), sep=',')
# Load DenseNet201
model = DenseNet201()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)
img_size = 224
features = {}

for image in tqdm(data['image'].unique().tolist()):
    img = load_img(os.path.join(image_path, image), target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    feature = fe.predict(img, verbose=0)
    features[image] = feature

# Lưu features
np.save(os.path.join(BASE_DIR, 'features.npy'), features)