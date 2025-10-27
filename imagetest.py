import numpy as np
import pandas as pd
import kagglehub
from PIL import Image
import os
from itertools import islice
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
path = kagglehub.dataset_download("fahadullaha/facial-emotion-recognition-dataset")
img = Image.open(f"{path}/processed_data/angry/angry_00000.jpg")
scaler = MinMaxScaler()
mainframe = pd.DataFrame()
mainseries = pd.Series()
for folder in islice(os.listdir(f"{path}/processed_data/"), 2):
    for filename in islice(os.listdir(f"{path}/processed_data/{folder}/"), 5):
        img = Image.open(f"{path}/processed_data/{folder}/{filename}")
        nimg = img.convert("L")
        nimg = np.asarray(nimg)
        timg = scaler.fit_transform(nimg)
        arr = timg.flatten()
        df = pd.DataFrame([arr])
        mainframe = pd.concat([mainframe, df])
        mainseries = pd.concat([mainseries, pd.Series([folder])])
print(mainframe.size, mainframe.shape)
print(mainseries)
mainframe.to_csv('mainframe.csv', index=False)


# newimg = img.convert("L")
# nimg = np.asarray(newimg)
# scaler = MinMaxScaler()
# x = scaler.fit_transform(nimg)
# df = pd.DataFrame(x)
# print(df)