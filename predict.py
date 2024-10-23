import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import plot_model
import shap
import numpy as np
import matplotlib.pyplot as plt
import sys, os

 
mutation = sys.argv[1]
curr_path = os.path.abspath(__file__)

class_names =['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
 
# Loading the MNIST dataset
test_images, test_labels = np.load('dataset/mnist_c/' + mutation + '/test_images.npy'), np.load('dataset/mnist_c/' + mutation + '/test_labels.npy')
 
# Adding a channel dimension to the images
#test_images = np.expand_dims(a=test_images, axis=-1) / 255.0
 
# Obtaining possible labels from the dataset
labels = np.unique(ar=test_labels)
 
# Converting labels to categoricals
test_labels = tf.keras.utils.to_categorical(test_labels)

# Creating a list for image indices
indices = []

# Obtaining the index of the first image of each label
for label in labels:
    label_index = np.where(np.argmax(a=test_labels, axis=1)==label)[0][0]
    indices.append(label_index)
    
imgs_to_predict = test_images[indices]
labels_to_predict = test_labels[indices]

 
def plot_image(i, predictions_array, true_label, img): # taking index and 3 arrays viz. prediction array, true label array and image array
 
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
 
  plt.imshow(img, cmap=plt.cm.binary) # showing b/w image
 
  predicted_label=np.argmax(predictions_array)
  true_label=np.argmax(true_label)
 
  # print(predicted_label)
  # print(true_label)
 
  if predicted_label == true_label: #setting up label color
    color='blue' # correct then blue colour
   
  else:
    color='red' # wrong then red colour

  print(predictions_array)
 
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                       100*np.max(predictions_array),
                                       class_names[true_label]),
             color=color)
 
# function to display bar chart showing whether image prediction is how much correct  
def plot_value_array(i, predictions_array, true_label): # taking index along with predictions and true label array
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot=plt.bar(range(10), predictions_array, color='gray')
  plt.ylim([0,1])
  predicted_label=np.argmax(predictions_array)
  true_label=np.argmax(true_label)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')
 
 
#model = load_model('Model/shap_model_new.h5')
model = load_model('Model/shap_model_new.h5')
predictions=model.predict(imgs_to_predict)
 
 
num_rows=2
num_cols=5
num_images=num_rows*num_cols
 
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i,predictions, labels_to_predict, imgs_to_predict)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, labels_to_predict)
#plt.show()
#plt.savefig('Results/SHAP/' + mutation + '/predictions.png', bbox_inches='tight')
#plt.savefig('Results/LIME/' + mutation + '/predictions.png', bbox_inches='tight')