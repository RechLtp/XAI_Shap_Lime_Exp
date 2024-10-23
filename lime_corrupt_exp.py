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

# Obtaining the index of the first image of each label
for label in labels:
    label_index = np.where(np.argmax(test_labels, axis=1)==label)[0][0]
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

my_explainer = Explainer(imgs_to_explain)

# Choosing max evaluations to try
max_evals = [100, 500, 1000]

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
    
    plt.savefig('Results/LIME/' + mutation + '/' + str(num_samples) + '.png', bbox_inches='tight')


# explanation = explainer.explain_instance(
#          x_train[10], 
#          model.predict,
#          top_labels=10,
#          num_samples=500
# )

# plt.imshow(x_train[10])
# image, mask = explanation.get_image_and_mask(
#          model.predict(
#               x_train[10].reshape((1,28,28,3))
#          ).argmax(axis=1)[0],
#          positive_only=False, 
#          hide_rest=False)
# plt.imshow(mark_boundaries(image, mask))

#plt.show()