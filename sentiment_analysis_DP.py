import numpy as np
import pandas as pd
import kagglehub
from PIL import Image
import os
from itertools import islice
from sklearn.preprocessing import MinMaxScaler

path = kagglehub.dataset_download("fahadullaha/facial-emotion-recognition-dataset")
scaler = MinMaxScaler()
mainframe = pd.DataFrame()
mainseries = pd.Series()

for folder in os.listdir(f"{path}/processed_data/"):
    for filename in os.listdir(f"{path}/processed_data/{folder}/"):
        img = Image.open(f"{path}/processed_data/{folder}/{filename}")
        nimg = img.convert("L")
        nimg = np.asarray(nimg)
        timg = scaler.fit_transform(nimg)
        arr = timg.flatten()
        df = pd.DataFrame([arr])
        mainframe = pd.concat([mainframe, df])
        mainseries = pd.concat([mainseries, pd.Series([folder])])
        
mainframe.to_csv('mainframe.csv', index=False)
mainseries.to_csv('mainseries.csv', index=False)