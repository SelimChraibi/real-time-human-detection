"""

Contain methods for:

1.  Preparing the INRIAPerson dataset
2.  Loading and displaying data
3.  Training the model
4.  Analysing the trained model
5.  Retrieving the class activation maps

"""

# ┌──────────────────────────────────────────────────────┐
# │       1.  Preparing the INRIAPerson dataset          │
# └──────────────────────────────────────────────────────┘

import os
import shutil
import numpy as np
import os
import subprocess

def cmd(command):
    """
    Executes bash command
    """
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error

#--------------------------------------------------------

def rm_unused_folders(source):
    """
    Removes the files that won't be used in the project
    """
    cmd('sudo chmod -R 777 *')

    if not source[-1] == '/':
        source += '/'

    for folder in os.listdir(source):
        if folder != 'Test' and folder != 'Train' and folder != 'Valid':
            folder = source + folder
            if os.path.isfile(folder):
                os.remove(folder)
            else:
                shutil.rmtree(folder)

    for folder in os.listdir(source + 'Train/'):
        if folder != 'neg' and folder != 'pos':
            folder = source + 'Train/' + folder
            if os.path.isfile(folder):
                os.remove(folder)
            else:
                shutil.rmtree(folder)

    for folder in os.listdir(source + 'Test/'):
        if folder != 'neg' and folder != 'pos':
            folder = source + 'Test/' + folder
            if os.path.isfile(folder):
                os.remove(folder)
            else:
                shutil.rmtree(folder)


#--------------------------------------------------------

def create_validation_data_folder(source, proportion=0.2):
    """
    If the dataset contains only a `Train` and a `Test` folder,
    this will create a `Valid` folder with a proportion of the the data from `Train` folder.

        source/
        ├── Train/
        │   ├── neg/
        │   └── pos/
        ├── Test/
        │   ├── neg/
        └── └── pos/

    """
    if not source[-1] == '/':
        source += '/'

    valid = source + 'Valid/'
    train = source + 'Train/'

    if not os.path.exists(valid):
        os.makedirs(valid)
        categories = os.listdir(train)
        for category in categories:
            category += '/'
            os.makedirs(valid + category)
            files = os.listdir(train + category)
            for file in files:
                if np.random.rand(1) < proportion:
                    shutil.move(train + category + file, valid + category + file)


# ┌──────────────────────────────────────────────────────┐
# │         2.  Loading and displaying data              │
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

    # balancing the data
    new_files = []
    new_targets = []
    for file, target in zip(files, targets):
        target = list(target)
        if target[1] == 1:
            new_files.append(file)
            new_targets.append(target)
        elif np.random.rand(1) < 0.5:
            new_files.append(file)
            new_targets.append(target)
    targets = np.array(new_targets)
    files   = np.array(new_files)

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
#     plt.imshow(img_path)
#     plt.axis('off')
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()

#--------------------------------------------------------

from keras.preprocessing import image
from tqdm import tqdm

def img_path_to_tensor(img_path):
    """
    Takes a string-valued file path to a color image (3 channels) as input
    and returns a 4D tensor of shape (1, 224, 224, 3) suitable for supplying to a Keras CNN
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0).astype('float32')/255

def paths_to_tensor(img_paths):
    """
    Takes a numpy array of string-valued image paths as input
    and returns a 4D tensor with shape (nb_samples, 224, 224, 3)
    """
    list_of_tensors = []
    for img_path in tqdm(img_paths):
        list_of_tensors += [img_path_to_tensor(img_path)]
    return np.vstack(list_of_tensors)

#--------------------------------------------------------

def nb_pos(targets):
    count = 0
    for target in targets:
        if target[1]==1:
            count+=1
    return count

def nb_neg(targets):
    count = 0
    for target in targets:
        if target[0]==1:
            count+=1
    return count

def stats(classes, train_targets, valid_targets, test_targets):
    """
    Statistics about the dataset
    """
    print('Classes:', classes)
    print('Total images \t\t: ', len(np.vstack([train_targets, valid_targets, test_targets])))
    print('Training images \t: ', len(train_targets), " \t = ", nb_pos(train_targets), "pos +", nb_neg(train_targets), "neg")
    print('Validation images \t: ', len(valid_targets), " \t = ", nb_pos(valid_targets), "pos +", nb_neg(valid_targets), "neg")
    print('Test images \t\t: ', len(test_targets), " \t = ", nb_pos(test_targets), "pos +", nb_neg(test_targets), "neg")

# ┌──────────────────────────────────────────────────────┐
# │              3.  Training the model                  │
# └──────────────────────────────────────────────────────┘

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os

#--------------------------------------------------------

def train(model, model_name, train_tensors, train_targets, valid_tensors, valid_targets,
          epochs=5, batch_size=35, verbose=[1,0]):
    """
    Trains the model and saves it under 'saved_model_weights/<model_name>.hdf5'
    returns the 'history' of the learning process (used to plot the learning curves)
    """

    if not os.path.exists('saved_model_weights/'):
        os.makedirs('saved_model_weights/')

    checkpointer = ModelCheckpoint(filepath          = 'saved_model_weights/' + model_name +'.hdf5',
                                   verbose           = verbose[0],
                                   save_best_only    = True,
                                   save_weights_only = False)

    # Training
    history = model.fit(x               = train_tensors,
                        y               = train_targets,
                        validation_data = (valid_tensors, valid_targets),
                        epochs          = epochs,
                        batch_size      = batch_size,
                        verbose         = verbose[1],
                        callbacks       = [checkpointer])

    return history


from keras.preprocessing.image import ImageDataGenerator
import os

def train_generator(model, model_name, train_tensors,
                    train_targets, valid_tensors, valid_targets,
                    train_folder, valid_folder,
                    epochs=5, batch_size=35, verbose=[1,0]):
    """
    Trains the model and saves it under 'saved_model_weights/<model_name>.hdf5'
    returns the 'history' of the learning process (used to plot the learning curves)

    **With data augmentation**
    """

    if not os.path.exists('saved_model_weights/'):
        os.makedirs('saved_model_weights/')

    checkpointer = ModelCheckpoint(filepath          = 'saved_model_weights/' + model_name +'.hdf5',
                                   verbose           = verbose[0],
                                   save_best_only    = True,
                                   save_weights_only = False)

    # Augmentation configuration for training:
    train_datagen = ImageDataGenerator(rescale            = 1./255,
                                       width_shift_range  = 0.5,
                                       height_shift_range = 0.5,
                                       shear_range        = 0.5,
                                       zoom_range         = 0.5,
                                       horizontal_flip    = True)

    # Augmentation configuration for testing: (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Generator of augmented training data from train_folder:
    train_generator = train_datagen.flow_from_directory(train_folder,
                                                        target_size = (224, 224),
                                                        batch_size  = batch_size,
                                                        class_mode  = 'categorical')
    # Generator for validation data:
    validation_generator = test_datagen.flow_from_directory(valid_folder,
                                                        target_size = (224, 224),
                                                        batch_size  = batch_size,
                                                        class_mode  = 'categorical')

    # Training
    history = model.fit_generator(generator        = train_generator,
                                  validation_data  = (valid_tensors, valid_targets),
                                  epochs           = epochs,
                                  verbose          = verbose,
                                  callbacks        = [checkpointer],
                                  steps_per_epoch  = train_tensors.shape[0] // batch_size,
                                  validation_steps = valid_tensors.shape[0] // batch_size)

    return history


# ┌──────────────────────────────────────────────────────┐
# │         4.  Analysing the trained model              │
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
        plt.subplot(1,3,1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')

        # Plot loss
        plt.subplot(1,3,3)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig(filename)

        plt.suptitle(filename)
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
    tensor = img_path_to_tensor(img_path)

    return np.argmax(model.predict(tensor))

# ┌──────────────────────────────────────────────────────┐
# │     5.  Retrieving the class activation maps         │
# └──────────────────────────────────────────────────────┘

import keras.backend as K

def get_layer(model, layer_name):
    """
    Returns the layer with the name 'layer_name'
    """
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def cam(model, img, name_of_final_conv_layer, name_of_dense_layer, class_number):
    resized_img = cv2.resize(img, (224, 224))
    # convert 3D tensor (cv2 imgage of 3 channels) into 4D tensor with shape (1, 224, 224, 3)
    tensor = np.expand_dims(resized_img, axis=0).astype('float32')/255
    # Get the input weights to the dense layer.
    class_weights = get_layer(model, name_of_dense_layer).get_weights()[0]

    # We retrieve the final conv layer
    final_conv_layer = get_layer(model, name_of_final_conv_layer)

    # This function outputs the output of the final conv layer given the tensor input of the 1st layer
    get_output = K.function([model.layers[0].input], [final_conv_layer.output])

    [conv_outputs] = get_output([tensor])
    conv_outputs = conv_outputs[0, :, :, :]

    # Original input image to which we will add the cam
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width, height, _ = img.shape

    # ---------------- Creation of the class activation map ----------------

    # Initialize with the right shape
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])

    # Weighted sum of the cam = \Sum_i (cam_i * weight_i)
    for i, w in enumerate(class_weights[:, class_number]):
            cam += w * conv_outputs[:, :, i]

    # We normalise and resize the cam
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width), interpolation = cv2.INTER_CUBIC)

    # We transform it into a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0

    # And add it to the original input image
    img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)

    return img, model.predict(tensor)[0][1]

def class_activation_map(model, img_path, name_of_final_conv_layer, name_of_dense_layer, class_number):
    """
    Displays the class activation map of the classes 'class_number' for the given model and image.
    name_of_dense_layer and name_of_final_conv_layer can be found by calling `model.summary()`
    """
    tensor = img_path_to_tensor(img_path)

    # Get the input weights to the dense layer.
    class_weights = get_layer(model, name_of_dense_layer).get_weights()[0]

    # We retrieve the final conv layer
    final_conv_layer = get_layer(model, name_of_final_conv_layer)

    # This function outputs the output of the final conv layer given the tensor input of the 1st layer
    get_output = K.function([model.layers[0].input], [final_conv_layer.output])

    [conv_outputs] = get_output([tensor])
    conv_outputs = conv_outputs[0, :, :, :]

    # Original input image to which we will add the cam
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width, height, _ = img.shape

    # ---------------- Creation of the class activation map ----------------

    # Initialize with the right shape
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])

    # Weighted sum of the cam = \Sum_i (cam_i * weight_i)
    for i, w in enumerate(class_weights[:, class_number]):
            cam += w * conv_outputs[:, :, i]

    # We normalise and resize the cam
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))

    # We transform it into a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0

    # And add it to the original input image
    img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)

    return img

#--------------------------------------------------------

# import numpy as np
# import matplotlib.cm as cm
# from vis.visualization import visualize_cam

# def cam(model, img):
#     for modifier in [None, 'guided', 'relu']:
#         # 20 is the imagenet index corresponding to `ouzel`
#         grads = visualize_cam(model, layer_idx, filter_indices=20,
#                               seed_input=img, backprop_modifier=modifier)
#         # Lets overlay the heatmap onto original image.
#         jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
#         img = overlay(jet_heatmap, img)
#     return img

#--------------------------------------------------------

import random
import numpy as np
from matplotlib import pyplot as plt
import os

def quick_test(model, model_name, test_files, test_targets, nb_images, name_of_final_conv_layer, name_of_dense_layer):
    """
    Displays 'nb_images' images from 'test_data',
    the prediction made by the 'model'
    and the class activation maps for each category.
    """
    categories = ['non-human', 'human']

    for i in range(nb_images):

        i            = random.choice(range(len(test_files)))
        img_path     = test_files[i]
        ground_truth = categories[list(test_targets[i]).index(1)]
        prediction   = categories[predict(model, img_path)]

        width,height = 2, len(categories)//2
        fig = plt.figure(figsize=(20,height*5))

        for c, category in enumerate(categories):
            ax      = fig.add_subplot(height, width, c+1, xticks=[], yticks=[], title=category + ' cam')
            image   = class_activation_map(model, img_path, name_of_final_conv_layer, name_of_dense_layer, c)
            highlight_ax(ax, category == prediction)
            ax.imshow(image)

        fig.suptitle('Model=' + model_name + '        Prediction=' + prediction + '        GroundTruth=' + ground_truth)

        save_folder = './saved_cams/'
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        success = ground_truth == prediction
        fig.savefig(save_folder + model_name + '_' + ['failure', 'success'][success] + '_' + img_path[21:-4] + '.png')

def quick_test_no_show(model, test_files, test_targets, nb_images):
    """
    Only ground truth and prediction.
    """
    categories = ['non-human', 'human']

    for i in range(nb_images):

        i            = random.choice(range(len(test_files)))
        img_path     = test_files[i]
        ground_truth = categories[list(test_targets[i]).index(1)]
        prediction   = categories[predict(model, img_path)]

        print(img_path)
        print('ground_truth = ', ground_truth)
        print('prediction   = ', prediction)
        print()

#--------------------------------------------------------

def highlight_ax(ax, activated):
    """
    Puts a blue frame aroud ax if success.
    """
    if activated:
        color = 'dodgerblue'
        linewidth = 5
        ax.spines['left'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['right'].set_linewidth(linewidth)
        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['top'].set_linewidth(linewidth)
    else:
        ax.axis('off')
