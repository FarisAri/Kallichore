import numpy as np
import pandas as pd
import kagglehub
from PIL import Image
import os
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

path = kagglehub.dataset_download("fahadullaha/facial-emotion-recognition-dataset") #download dataset and set the file path as path
scaler = MinMaxScaler() #set scaler so it squishes pixel values between 0 and 1

#initialize empty dataframe and series to hold data
mainlist = []
mainseries = []
base_path = f"{path}/processed_data/" #progress bar
folders = os.listdir(base_path)
total_images = sum(len(os.listdir(os.path.join(base_path, folder))) for folder in folders) # progress bar

#loop through each folder (emotion category) and each image file within those folders
with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
    for folder in folders: 
        for filename in islice(os.listdir(f"{path}/processed_data/{folder}/"), None):
            img = Image.open(f"{path}/processed_data/{folder}/{filename}") #open image
            nimg_grayscale = img.convert("L") #convert to grayscale
            img_array = np.asarray(nimg_grayscale) #convert to numpy array
            timg = img_array/255 #squash values to fit between 0 and 1
            arr = timg.flatten() #convert 96x96 array into 1D array of 9216 elements
            mainlist.append(arr)
            mainseries.append(folder) #add corresponding label to main series
            pbar.update(1)  # update progress bar
        
#save dataframe and series as csv files
mainframe = pd.DataFrame(mainlist)
mainframe.to_csv('mainframe.csv', index=False)
mainseries = pd.Series(mainseries)
mainseries.to_csv('mainseries.csv', index=False)
print(mainframe.shape, mainframe.size)
print(mainseries.shape, mainseries.size)
print("finished")
