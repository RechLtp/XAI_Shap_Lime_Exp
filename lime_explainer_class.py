
# Importing dependencies
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm
from PIL import Image



# Creating an explainer
explainer = lime_image.LimeImageExplainer(random_state=5)

# Class for making explanations
class Explainer():
    # Class constructor
    def __init__(self,
                 imgs_to_explain):
        self.imgs_to_explain = imgs_to_explain
        self.explanations =  []
    
    # Function for explaining instances
    def explain_instances(self,
                          classifier_fn,
                          top_labels,
                          num_samples,
                          segmentation_fn,
                          random_seed):        
        
        # Emptying the explanation list in case
        # explanations have been generated before
        self.explanations =  []
        
        # Iterating over the images
        for i in range(len(self.imgs_to_explain)):
            # Explaining an image of each digit and saving the explainer
            self.explanations.append(explainer.explain_instance(image=self.imgs_to_explain[i],
                                                                classifier_fn=classifier_fn,
                                                                top_labels=top_labels,
                                                                num_samples=num_samples,
                                                                segmentation_fn=segmentation_fn,
                                                                random_seed=random_seed))
    
    # Function for plotting explanations
    def plot_explanations(self,
                          rows,
                          cols,
                          image_indices,
                          top_predictions,
                          mutation,
                          positive_only=True,
                          negative_only=False,
                          hide_rest=False):
                      
        # Creating a figure and subplots
        # We are adding 1 to cols to get a spot for the source image
        fig, ax = plt.subplots(rows, cols + 1, squeeze=False)
        fig.set_size_inches(4 * cols, 3 * rows)
        
        # Iterating over the provided image indices        
        for i in range(len(image_indices)):           
            # Getting the explanation for the supplied image index
            explanation = self.explanations[image_indices[i]]

            # Showing the source image in the leftmost column
            ax[i, 0].imshow(self.imgs_to_explain[image_indices[i]])
            
            # Hiding x and y ticks
            ax[i, 0].set_xticks(ticks=[])
            ax[i, 0].set_yticks(ticks=[])
            
            # Iterating over the top predicted labels
            # We are starting j from 1 and going 1 over top_predictions because we need
            # to plot after the source image
            for j in range(1, top_predictions + 1):       
                # Generating the image heatmap and the mask
                # We are subtracting 1 from j because
                # j starts from 1, whereas top_labels starts from 0
                temp, mask = explanation.get_image_and_mask(label=explanation.top_labels[j - 1],
                                                            positive_only=positive_only,
                                                            negative_only=negative_only,
                                                            hide_rest=hide_rest)                              
                #temp contains original image + red and green
                #plt.imshow(temp)
                #plt.tight_layout()
                #plt.axis('off')
                #plt.show()
                count = 0
                count += 1
                #plt.savefig('Results/LIME/' + mutation + '/' + str(explanation.top_labels[j - 1]) + '_' + str(j) + '_' + str(count) + '.png', bbox_inches='tight')
                #plt.show()
                #mask contains differentiated colours of images
                #plt.imshow(mask)
                #plt.axis('off')
                #plt.show()
                # Plotting the explanation for the current top_label
                ax[i, j].imshow(mark_boundaries(temp / 2 + 0.5, mask))

                # Showing the prediction corresponding to explanation
                ax[i, j].set_title(label=f"Prediction: {explanation.top_labels[j - 1]}")                
                
                # Hiding x and y ticks
                ax[i, j].set_xticks(ticks=[])
                ax[i, j].set_yticks(ticks=[])  

        # Using the space available in the figure efficiently
        plt.tight_layout()
        
        # Showing the plot
        #plt.show()
        
    def plot_explanations_for_single_image(self,
                                           rows,
                                           cols,
                                           image_index,
                                           labels,
                                           positive_only=True,
                                           negative_only=False,
                                           hide_rest=False):
        
        # Creating a figure for the explanations
        plt.figure(figsize=(3 * cols, 3 * rows))
        
        # Getting the explanation for the supplied image index
        explanation = self.explanations[image_index]
        
        # Iterating over the total number of spots in the figure
        for i in range(rows * cols):
            # Creating a subplot
            plt.subplot(rows, cols, i + 1)    
            
            # Generating the processed image and mask
            temp, mask = explanation.get_image_and_mask(label=labels[i],
                                                        positive_only=positive_only,
                                                        negative_only=negative_only,
                                                        hide_rest=hide_rest)
            
            # Plotting the explanation for the current label
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            
            # Showing the label that the image was explained for
            plt.title(label=f"Label: {labels[i]}")

            # Hiding x and y ticks
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])  

        # Using space available in the figure efficiently
        plt.tight_layout()

        # Showing the plot
        plt.show()