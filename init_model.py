import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# Loading the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Adding a channel dimension to the images
train_images, test_images = np.expand_dims(a=train_images, axis=-1) / 255.0, np.expand_dims(a=test_images, axis=-1) / 255.0

# Obtaining possible labels from the dataset
labels = np.unique(ar=test_labels)

# Converting labels to categoricals
train_labels, test_labels = tf.keras.utils.to_categorical(train_labels), tf.keras.utils.to_categorical(test_labels)

# Function for creating TF Keras model
def create_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First Conv2D + MaxPool2D block
    x = Conv2D(filters=16,
               kernel_size=(3, 3),
               activation="relu",
               padding="same")(input_layer)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Second Conv2D + MaxPool2D block
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation="relu",
               padding="same")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    # Third Conv2D + MaxPool2D block
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation="relu",
               padding="same")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Global Average Pooling and Dense output layers
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(units=10, 
                         activation="softmax")(x)

    # Returning the model
    return Model(inputs=input_layer, 
                 outputs=output_layer)


# Creating the model
model = create_model()

# Compiling the model
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Training the model
history = model.fit(x=train_images, y=train_labels,
                    batch_size=500,
                    epochs=12,
                    validation_split=0.1,
                    validation_data=(test_images, test_labels))


# Saving our model
#model.save('shap_model.h5')
model.save_weights('shap_model_weights.h5')
#model.export("saved_shap_model")

# Plotting the model
#plot_model(model=model, 
#           show_shapes=True)

print("Model Saved")