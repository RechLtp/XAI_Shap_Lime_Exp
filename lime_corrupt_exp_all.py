import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
import numpy as np
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random
from lime_explainer_class import Explainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
import sys, os

mutation = sys.argv[1]
curr_path = os.path.abspath(__file__)

x_test, y_test = np.load('dataset/mnist_c/' + mutation + '/test_images.npy'), np.load('dataset/mnist_c/' + mutation + '/test_labels.npy')
x_test = x_test.reshape((-1,28,28,1)).astype('float32') / 255.0


def to_rgb(x):
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb

#x_train = to_rgb(x_train)
x_test = to_rgb(x_test)

labels = np.unique(ar=y_test)

# Converting labels to categoricals
test_labels = tf.keras.utils.to_categorical(y_test)

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
    
imgs_to_explain = x_test[indices]


model = load_model('Model/lime_model.h5')




# Function for creating a segmenter
def create_segmenter(**kwargs):       
    # Returning an instantiated segmenter
    return SegmentationAlgorithm(**kwargs)

segmenter_quickshift = create_segmenter(algo_type="quickshift", 
                                        kernel_size=1, 
                                        max_dist=2, 
                                        random_seed=5)

for count,value in enumerate(imgs_to_analyze_arr):
    start_index = 10 * count
    end_index = start_index + 10
    my_explainer = Explainer(imgs_to_explain[start_index:end_index])

    # Choosing max evaluations to try
    #max_evals = [100, 500, 1000]
    max_evals = [3000,5000]

    for num_samples in max_evals:

        my_explainer.explain_instances(
        model.predict,
        top_labels=10,
        num_samples=num_samples,
        segmentation_fn=segmenter_quickshift,
        random_seed=5
        )
    
        my_explainer.plot_explanations(rows=10,
                               cols=3,
                               image_indices=list(range(10)),
                               top_predictions=3,
                               positive_only=False)
    
        #plt.show()
        plt.savefig('Results/LIME/' + mutation + '/' + str(num_samples) + '_' + str(count) + '.png', bbox_inches='tight')





