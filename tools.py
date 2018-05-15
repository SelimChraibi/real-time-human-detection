# ┌──────────────────────────────────────────────────────┐
# │             Loading and displaying data              │ 
# └──────────────────────────────────────────────────────┘

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

def load_dataset(path):
    """
    Loads the file in the path and returns: 
    - the files, 
    - their coresponding target labels (one-hot encoded)
    - a list of the class names
    """
    data = load_files(path)
    data['target'] -= min(data['target'])
    
    nb_classes = len(glob(path + "/*"))
    
    files   = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), nb_classes)
    classes = [item[len(path):] for item in sorted(glob(path + "/*"))]
    
    return files, targets, classes

#--------------------------------------------------------

from PIL.Image import open
import numpy as np
import matplotlib.pyplot as plt
import random

def show_examples(files, number=25):
    """
    Shows 'number' images selected randomly from 'files'
    """
    width  = 5
    height = number//5
    number = height*width
    
    fig = plt.figure(figsize=(20,height*2))
    for i in range(number):
        ax = fig.add_subplot(height, width, i + 1, xticks=[], yticks=[])
        image = np.asarray(open(random.choice(files)))
        ax.imshow(image)
        
#--------------------------------------------------------

import cv2

def show(img_path):
    """
    Shows the image located at 'img_path'
    """
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()
        
#--------------------------------------------------------

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    """
    Takes a string-valued file path to a color image (3 channels) as input
    and returns a 4D tensor of shape (1, 224, 224, 3) suitable for supplying to a Keras CNN
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    """
    Takes a numpy array of string-valued image paths as input 
    and returns a 4D tensor with shape (nb_samples, 224, 224, 3)
    """
    list_of_tensors = []
    for img_path in tqdm(img_paths):
        list_of_tensors += [path_to_tensor(img_path)]
    return np.vstack(list_of_tensors).astype('float32')/255

# ┌──────────────────────────────────────────────────────┐
# │                  Training the model                  │ 
# └──────────────────────────────────────────────────────┘

import tensorflow as tf
from keras.callbacks import ModelCheckpoint  
import os

    
def train(model, model_name, train_tensors, train_targets, epochs=5, batch_size=35, verbose=1):
    """
    Trains the model and saves it under 'saved_models/<model_name>.hdf5'
    returns the 'history' of the learning process (used to plot the learning curves)
    """
    
    if not os.path.exists('saved_models/'):
        os.makedirs('saved_models/')
    
    checkpointer = ModelCheckpoint(filepath='saved_models/' + model_name +'.hdf5', 
                               verbose=1, 
                               save_best_only=True,
                               save_weights_only=False)

    history = model.fit(train_tensors, train_targets, 
                        validation_split=0.2,
                        epochs=epochs, 
                        batch_size=batch_size,
                        callbacks=[checkpointer], 
                        verbose=verbose)
    return history

# ┌──────────────────────────────────────────────────────┐
# │             Analysing the trained model              │ 
# └──────────────────────────────────────────────────────┘

import matplotlib.pyplot as plt  
import seaborn as sns
import numpy as np

def learning_curves(history, filename):
    """
    The .fit() method of a keras model returns the 'history' of the learning process.
    This method plots the learning_curves saved in 'history' and saves the plots in ./'filename'.
    """
    
    with plt.style.context('seaborn-darkgrid'):
        
        plt.figure(figsize=(16, 5))

        # Plot accuracy
        plt.subplot(1,2,1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')

        # Plot loss
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig(filename)
        
        plt.suptitle('filename')
        plt.show()

#--------------------------------------------------------

def compute_accuracy(model, test_tensors, test_targets):
    """
    Accuracy of the classification model.
    """
    # get index of predicted class for each image in test set
    class_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(class_predictions)==np.argmax(test_targets, axis=1))/len(class_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
    
#--------------------------------------------------------

def predict(model, img_path):
    """
    Return the predicted class.
    """
    # preprocess image into resized tensor
    tensor = path_to_tensor(img_path)
    
    return np.argmax(model.predict(tensor))        

# ┌──────────────────────────────────────────────────────┐
# │         Retrieving the class activation maps         │ 
# └──────────────────────────────────────────────────────┘

import keras.backend as K

def get_layer(model, layer_name):
    """
    Returns the layer with the name 'layer_name'
    """
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def class_activation_map(model, img_path, name_of_final_conv_layer, name_of_dense_layer, class_number):
    """
    Displays the class activation map of the class 'class_number' for the given model.
    name_of_dense_layer and name_of_final_conv_layer can be found by calling `model.summary()`
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width, height, _ = img.shape

    tensor = path_to_tensor(img_path)

    # Get the input weights to the dense layer.
    class_weights = get_layer(model, name_of_dense_layer).get_weights()[0]
    
    
    # Get the output of the final convolutional layer for the tensor input
    final_conv_layer = get_layer(model, name_of_final_conv_layer)
    
    get_output = K.function([model.layers[0].input], [final_conv_layer.output])
    [conv_outputs] = get_output([tensor])
    
    conv_outputs = conv_outputs[0, :, :, :]

    # ---------------- Create the class activation map ----------------
    
    # Initialize with the right shape
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])

    # Sum_i (activation_map_i * weight_i)
    for i, w in enumerate(class_weights[:, class_number]):
        
            cam += w * conv_outputs[:, :, i]

    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    
    # Add to original image
    img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
    
    plt.imshow(img)
    plt.show()

#--------------------------------------------------------

import random

def quick_test(model, test_data, nb_images, name_of_final_conv_layer, name_of_dense_layer):
    """
    Displays 'nb_images' images from 'test_data', 
    the prediction made by the 'model' 
    and the class activation map for the class predicted.
    """
    for i in range(nb_images):
        
        img_path = random.choice(test_data)
        display(img_path)
        
        if predict(model, img_path):
            print('human')
            class_activation_map(model, img_path, name_of_final_conv_layer, name_of_dense_layer, class_number=0) 
        else:
            print('not human')
            class_activation_map(model, img_path, name_of_final_conv_layer, name_of_dense_layer, class_number=1) 