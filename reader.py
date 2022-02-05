import gtts
from playsound import playsound

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps
import os, ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
Y = pd.read_csv("labels.csv")["labels"]
#print(pd.Series(Y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

x_train, x_test, y_train, y_test = ttsplit(X, Y, random_state = 0, train_size = 7500, test_size = 2500)

scaled_xtr = x_train/255.0
scaled_xts = x_test/255.0

classifier = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(scaled_xtr, y_train)

y_pred = classifier.predict(scaled_xts)

print("Accuracy is :", accuracy_score(y_test, y_pred))

def prediction(image):
    img_pil = Image.open(image)
    img_bw = img_pil.convert('L')
    img_resized = img_bw.resize((22, 30), Image.ANTIALIAS)
    min_pix = np.percentile(img_resized, 20)
    img_clipped = np.clip(img_resized-min_pix, 0,255)
    max_pix = np.max(img_resized)
    img_clipped = np.asarray(img_clipped)/max_pix

    test_sample = np.array(img_clipped).reshape(1, 784)
    test_pred = classifier.predict(test_sample)
    return test_pred[0]

