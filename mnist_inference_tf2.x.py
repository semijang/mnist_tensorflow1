'''This is inference code for mnist dataset '''

from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from keras.layers import TFSMLayer

import os
os.makedirs("dataset_test/testlabels", exist_ok=True)
with open("dataset_test/testlabels/t_labels.txt", "w") as f:
    for i in range(10):
        f.write(f"{i}\n")
        
# Recreate the exact same model, including its weights and the optimizer
model = TFSMLayer('saved_model/', call_endpoint='serving_default')

# Show the model architecture
model.summary()

##-- Model Test using Test datasets
print()
print("----Actual test for digits----")

mnist_label_file_path =  "dataset_test/testlabels/t_labels.txt"
mnist_label = open(mnist_label_file_path, "r")
cnt_correct = 0

for index in range(10):
   #-- read a label
   label = mnist_label.readline().strip() 

   #print(label)
   #-- formatting the input image (image data)
   img = Image.open('dataset_test/testimgs/' + str(index+1) + '.png').convert("L")
   img = img.resize((28,28))

   im2arr = np.array(img) / 255.0
   im2arr = im2arr.reshape(1,28,28,1)

   # Predicting the Test set results
   # y_pred = model.predict_classes(im2arr)   #<-- 7 or 4
   y_pred = model(im2arr, training=False).numpy()
   pred_label = np.argmax(y_pred)

   print()
   # pred_label = np.argmax(y_pred) 
   

   print("label = {} --> predicted label= {}".format(label, pred_label))

   #-- compute the accuracy of the preditcion
   if int(label)==pred_label:
      cnt_correct += 1

#-- Final accuracy
Final_acc = cnt_correct/10
print()
print("Final test accuray: %f" %Final_acc)
print()
print('****tensorflow version****:',tf.__version__)
print()

data = {
    '이름': ['장세미'],
    '학번': [2415857],
    '학과': ['인공지능공학부']
}

df = pd.DataFrame(data)
print(df)

