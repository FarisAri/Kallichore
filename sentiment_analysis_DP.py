import numpy as np
import pandas as pd
import kagglehub
from PIL import Image
import os
from itertools import islice
from sklearn.preprocessing import MinMaxScaler

path = kagglehub.dataset_download("fahadullaha/facial-emotion-recognition-dataset") #download dataset and set the file path as path
scaler = MinMaxScaler() #set scaler so it squishes pixel values between 0 and 1

#initialize empty dataframe and series to hold data
mainframe = pd.DataFrame()
mainseries = pd.Series()

#loop through each folder (emotion category) and each image file within those folders
for folder in os.listdir(f"{path}/processed_data/"): 
    for filename in os.listdir(f"{path}/processed_data/{folder}/"):
        img = Image.open(f"{path}/processed_data/{folder}/{filename}") #open image
        nimg_grayscale = img.convert("L") #convert to grayscale
        img_array = np.asarray(nimg_grayscale) #convert to numpy array
        timg = scaler.fit_transform(nimg_grayscale) #squash values to fit between 0 and 1
        arr = timg.flatten() #convert 96x96 array into 1D array of 9216 elements
        df = pd.DataFrame([arr]) #convert to dataframe
        mainframe = pd.concat([mainframe, df]) #add to main dataframe
        mainseries = pd.concat([mainseries, pd.Series([folder])]) #add corresponding label to main series
        
#save dataframe and series as csv files
mainframe.to_csv('mainframe.csv', index=False)
mainseries.to_csv('mainseries.csv', index=False)
print("finished")
