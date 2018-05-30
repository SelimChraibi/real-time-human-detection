from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.datasets import load_files
import numpy as np
from tqdm import tqdm
from PIL.Image import open
import matplotlib.pyplot as plt
import os
import shutil
from glob import glob
import math
import pandas as pd

class AbstractDataset():
    """
    Import and preprocess a dataset from a source folder
    
    Attributes:
        X_train     Training data
        y_train     Training labels
        X_valid     Validation data
        y_valid     Validation labels
        X_test      Testing data
        y_test      Testing labels
        classes     List of classes
    """
    
    # Path to the root folder of the dataset
    source = ""
    # Features
    X_train = np.array([])
    X_valid = np.array([])
    X_test  = np.array([])
    # Corresponding labels
    y_train = []
    y_valid = []
    y_test  = []
    # List of classes
    classes = []
    
    def __init__(self, source):
        self.source = source
        if not self.source[-1] == '/': self.source += '/'
        self.X_train, self.y_train = self._loadFolder(self.source + "Train")
        self.X_test, self.y_test   = self._loadFolder(self.source + "Test")
        self.X_valid, self.y_valid = self._loadFolder(self.source + "Valid")
    
    
    def examples(self, sample='Train', number=25):
        """
        Shows 'number' images selected randomly from the training/validation/testing sample
        
        Args:
            img_paths (string): Name of the sample to display examples from ('Train' or 'Valid' or 'Test')
            number    (int)   : Number of images to be shown
        """
        files  = glob(self.source + sample + "/*/*")
        
        width  = min(4, number)
        height = math.ceil(number/4)

        fig = plt.figure(figsize=(20,height*5))
        for i in range(number):
            ax = fig.add_subplot(height, width, i + 1, xticks=[], yticks=[])
            img = np.asarray(open(np.random.choice(files)))
            ax.imshow(img)

    def statistics(self):
        """
        Statistics about the dataset
        
        Returns:
            pandas.DataFrame : Dataframe containing some statistics about the data
        """
        def count(targets, class_number):
            count = 0
            for target in targets:
                if target[class_number]==1:
                    count+=1
            return count

        stat_tab = np.array([[0]*(len(self.classes)+1)]*4)

        for class_number, class_name in enumerate(self.classes):
            stat_tab[0][class_number] = count(self.y_train, class_number)
            stat_tab[1][class_number] = count(self.y_valid, class_number)
            stat_tab[2][class_number] = count(self.y_test, class_number)   
        for i in range(3):
            stat_tab[i,-1] = sum(stat_tab[i,:-1])  
        for j in range(len(self.classes)+1):
            stat_tab[-1,j] = sum(stat_tab[:-1,j])
            
        stat_tab = np.hstack([[['Train'], ['Valid'], ['Test'], ['']], stat_tab])

        return pd.DataFrame(stat_tab, columns=["Classes"]+self.classes+["Total"]).set_index('Classes')


    def _loadFolder(self, folder):
        """
        Loads the files in the given folder into a tensor.
        
        Args:
            folder (string): Path to an folder
        Returns:
            numpy.ndarray  : 4D tensor of shape (nb_samples, 224, 224, 3) containing the loded images
            numpy.ndarray  : array containing the coresponding target labels (one-hot encoded)
        """
        data = load_files(folder)
        data['target'] -= min(data['target'])
        
        if not folder[-1] == '/': folder += '/'

        nb_classes = len(glob(folder + "*"))

        files = np.array(data['filenames'])
        tensors = self._imagePaths2tensor(files)

        targets = np_utils.to_categorical(np.array(data['target']), nb_classes)
        self.classes = [item[len(folder):] for item in sorted(glob(folder + "*"))]

#             # Balancing the data so that there is roughly as many images in 'neg' as in 'pos' 
#             new_files = []
#             new_targets = []
#             for file, target in zip(files, targets):
#                 target = list(target)
#                 if target[1] == 1:
#                     new_files.append(file)
#                     new_targets.append(target)
#                 elif np.random.rand(1) < 0.5:
#                     new_files.append(file)
#                     new_targets.append(target)
#             targets = np.array(new_targets)
#             files   = np.array(new_files)

        return tensors, targets

    def _imagePath2tensor(self, img_path):
        """
        Returns a normalized 4D tensor (suitable for supplying to a Keras CNN) associated to an image.
        
        Args:
            img_path (string): Path to an image (3 channels/colors)
        Returns:
            numpy.ndarray  : 4D tensor of shape (1, 224, 224, 3) containing the loded images
        """
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3)  return 4D tensor
        tensor = np.expand_dims(x, axis=0).astype('float32')
        # return normalized 4D tensor
        return tensor/255

    def _imagePaths2tensor(self, img_paths):
        """
        Returns a normalized 4D tensor (suitable for supplying to a Keras CNN) associated to a list of images.
        
        Args:
            img_paths (*string): Paths to images (3 channels/colors)
        Returns:
            numpy.ndarray      : 4D tensor of shape (nb_samples, 224, 224, 3) containing the loded images
        """
        list_of_tensors = []
        for img_path in tqdm(img_paths):
            list_of_tensors += [self._imagePath2tensor(img_path)]
        return np.vstack(list_of_tensors)



class InriaPersonDataset(AbstractDataset):
    """
    Import and clean the INRIA Person Dataset 
    The dataset must have been downloaded and extracted first. 
    The user should also have read and write permission.
    The dataset's webpage: http://pascal.inrialpes.fr/data/human/
    """
    
    def __init__(self, source, validation_split=0.2):
        if not source[-1] == '/': source += '/'
        self._removeUnusedFolders(source)
        self._createValidationDataFolder(source, validation_split)
        super().__init__(source)
        
    def _removeUnusedFolders(self, source):
        """
        Removes the folders that won't be used.
        """
        
        for folder in os.listdir(source):
            if folder != 'Test' and folder != 'Train' and folder != 'Valid':
                folder = self.source + folder
                if os.path.isfile(folder):
                    os.remove(folder)
                else:
                    shutil.rmtree(folder)

        for folder in os.listdir(source + 'Train/'):
            if folder != 'neg' and folder != 'pos':
                folder = self.source + 'Train/' + folder
                if os.path.isfile(folder):
                    os.remove(folder)
                else:
                    shutil.rmtree(folder)

        for folder in os.listdir(source + 'Test/'):
            if folder != 'neg' and folder != 'pos':
                folder = self.source + 'Test/' + folder
                if os.path.isfile(folder):
                    os.remove(folder)
                else:
                    shutil.rmtree(folder)
                    
    def _createValidationDataFolder(self, source, validation_split=0.2):
        """
        If the dataset contains only a `Train` and a `Test` folder,
        this will create a `Valid` folder with a proportion of the the data from `Train` folder.

            self.source/
            ├── Train/
            │   ├── neg/
            │   └── pos/
            ├── Test/
            │   ├── neg/
            └── └── pos/

        """

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
                    if np.random.rand(1) < validation_split:
                        shutil.move(train + category + file, valid + category + file)

       