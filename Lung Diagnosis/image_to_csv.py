import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub as kg
import os
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from tqdm import tqdm  # progress bar

path = kg.dataset_download("fatemehmehrparvar/lung-disease")
scaler = MinMaxScaler()

mainframe = pd.DataFrame()
mainseries = pd.Series(dtype=str)

base_path = f"{path}/Lung X-Ray Image/Lung X-Ray Image/" # progress bar
folders = os.listdir(base_path) # progress bar
total_images = sum(len(os.listdir(os.path.join(base_path, folder))) for folder in folders) # progress bar

# Wrap the loop in tqdm for progress bar
with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
    for folder in folders:
        for filename in os.listdir(os.path.join(base_path, folder)):
            img_path = os.path.join(base_path, folder, filename)
            img = Image.open(img_path)
            nimg_rgb = img.convert("RGB")
            img_array = np.asarray(nimg_rgb)
            img_scaled = scaler.fit_transform(img_array.reshape(-1, 3))
            timg = img_scaled.reshape(img_array.shape)
            arr = timg.flatten()
            df = pd.DataFrame([arr])
            mainframe = pd.concat([mainframe, df], ignore_index=True)
            mainseries = pd.concat([mainseries, pd.Series([folder])])
            pbar.update(1)  # update progress bar

mainframe.to_csv("lung_disease_features.csv", index=False)
mainseries.to_csv("lung_disease_labels.csv", index=False)
print("Feature extraction completed and saved to CSV files.")
