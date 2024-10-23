import tensorflow as tf
import shap
import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
import numpy as np
import matplotlib.pyplot as plt
import sys, os

mutation = sys.argv[1]
curr_path = os.path.abspath(__file__)

# Loading the MNIST dataset
test_images, test_labels = np.load('dataset/mnist_c/' + mutation + '/test_images.npy'), np.load('dataset/mnist_c/' + mutation + '/test_labels.npy')

# Adding a channel dimension to the images
#test_images = np.expand_dims(a=test_images, axis=-1) / 255.0

# Obtaining all possible labels from the dataset, 10 in this case (0 - 9)
labels = np.unique(ar=test_labels)

# Converting labels to categoricals
test_labels = tf.keras.utils.to_categorical(test_labels)


model = load_model('Model/shap_model_new.h5')

# Creating a list for image indices
indices = []
#how many images of each label to analyze
imgs_to_analyze = 5
#init array to look through
imgs_to_analyze_arr = list(range(imgs_to_analyze))
#imgs_to_analyze_arr = [0,1,2,3,4]

# Obtaining the index of the first image of each label
for label in labels:
    for imgs in imgs_to_analyze_arr:
        label_index = np.where(np.argmax(a=test_labels, axis=1)==label)[0][imgs]
        indices.append(label_index)
    
imgs_to_explain = test_images[indices]

# Confirming that the indices are correct
# Creating a figure
plt.figure(figsize=(12, 8))

# Setting the number of rows and columns
rows, cols = 5, 10

# Iterating over the total number of spots in the figure
for i in range(rows * cols):
    # Creating a subplot
    plt.subplot(rows, cols, i + 1)    
    
    # Showing current image
    plt.imshow(X=test_images[indices[i]])    
    
    # Showing label corresponding to image
    plt.title(label=f"Label: {np.argmax(test_labels[indices[i]])}")
    
    # Hiding x and y ticks
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])  

# Using space available in the figure efficiently
plt.tight_layout()

# Showing the plot
plt.savefig('Results/SHAP/' + mutation + '/indices.png', bbox_inches='tight')
#plt.show()


# Creating an image masker
masker = shap.maskers.Image(mask_value=None, 
                            shape=test_images[0].shape)

# Creating an explainer object
explainer = shap.Explainer(model=model.predict, 
                           masker=masker, 
                           algorithm="auto", 
                           output_names=labels.astype(str))

# Function for calculating and plotting SHAP values
def plot_shap_values(imgs_to_explain, max_evals,save_counter):
    # Calculating shap values for our images
    shap_values = explainer(imgs_to_explain,
                            max_evals=max_evals,
                            outputs=shap.Explanation.argsort.flip[:3])
    print(shap_values)


    # Plotting shap values
    shap.image_plot(shap_values=shap_values, 
                    show=False)
    #plt.show()
    plt.savefig('Results/SHAP/' + mutation + '/' + str(max_evals) + '_' + str(save_counter) + '.png', bbox_inches='tight')
    

# Choosing max evaluations to try
#max_evals = [100, 500, 1000]
max_evals = [3000,5000]


# Iterating over eval values
for max_eval in max_evals:
    # Plotting SHAP values
    for count,value in enumerate(imgs_to_analyze_arr):
        start_index = 10 * count
        end_index = start_index + 10
        plot_shap_values(imgs_to_explain=imgs_to_explain[start_index:end_index],
                         max_evals=max_eval, 
                         save_counter=count)
    


