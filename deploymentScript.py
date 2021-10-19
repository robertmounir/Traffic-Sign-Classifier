#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from numpy import expand_dims
from PIL import Image
def predictionFunc(imgpath,loadedmodel):
      P_test_img = cv2.imread(imgpath)
      P_test_img = Image.fromarray(P_test_img, 'RGB')
      P_test_img = P_test_img.resize((32, 32))
      P_test_img=np.array(P_test_img)
      P_test_img= (P_test_img.astype('float32')) / 255.
    
      P_test_img_input=np.expand_dims(P_test_img, 0)
      prediction = loadedmodel.predict(P_test_img_input)
      #print(prediction)

      predicted_class = np.argmax(prediction, axis=None)

      plt.figure(figsize=(2, 2))
      plt.imshow(P_test_img)

      classes = ['Speed limit (20km/h)', 'Speed limit (30km/h)',
        'Speed limit (50km/h)', 'Speed limit (60km/h)',
        'Speed limit (70km/h)', 'Speed limit (80km/h)',
        'End of speed limit (80km/h)', 'Speed limit (100km/h)',
        'Speed limit (120km/h)', 'No passing',
        'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield',
        'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
        'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road',
        'Slippery road', 'Road narrows on the right', 'Road work',
        'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right',
        'Go straight or left', 'Keep right', 'Keep left',
        'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons']
      prediction_label=classes[predicted_class]
      print("Predicted class is:",predicted_class ,prediction_label)


from keras.models import load_model

my_loaded_model = load_model('traffic_signs_model.h5')

# the image path that you want to predicit
imagePath = sys.argv[1]

predictionFunc(imagePath,my_loaded_model)