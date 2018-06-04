from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.datasets import load_files
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import os
import shutil
from glob import glob
import math
from google_images_download import google_images_download # https://github.com/hardikvasa/google-images-download


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
    source  = ""
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
        self.source = addSlash(source)
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

        fig = plt.figure(figsize=(20,height*3))
        for i in range(number):
            ax = fig.add_subplot(height, width, i + 1, xticks=[], yticks=[])
            img = cv2.imread(np.random.choice(files),1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    
    def merge(self, other):
        """
        Merge another dataset with the current one.
        """
        self.X_train = np.vstack([self.X_train, other.X_train])
        self.y_train = np.vstack([self.y_train, other.y_train]) 
        self.X_test  = np.vstack([self.X_test , other.X_test ])
        self.y_test  = np.vstack([self.y_test , other.y_test ]) 
        self.X_valid = np.vstack([self.X_valid, other.X_valid])
        self.y_valid = np.vstack([self.y_valid, other.y_valid]) 

    def _loadFolder(self, folder):
        """
        Loads the files in the given folder into a tensor.
        
        Args:
            folder (string): Path to an folder
        Returns:
            numpy.ndarray  : 4D tensor of shape (samples_size, 224, 224, 3) containing the loded images
            numpy.ndarray  : array containing the coresponding target labels (one-hot encoded)
        """
        data = load_files(folder)
        data['target'] -= min(data['target'])
        
        folder = addSlash(folder)
        
        nb_classes = len(glob(folder + "*"))
        self.classes = [item[len(folder):] for item in sorted(glob(folder + "*"))]
        
        files = np.array(data['filenames'])
        targets = np_utils.to_categorical(np.array(data['target']), nb_classes)
        
        tensors = imagePaths2tensor(files)      
        return tensors, targets

def image2tensor(img):
    """
    Returns a normalized 4D tensor (suitable for supplying to a Keras CNN) associated to an image.

    Args:
        img_path (BGR cv2 image): image (3 channels/colors)
    Returns:
        numpy.ndarray  : 4D tensor of shape (1, 224, 224, 3) containing the loded images
    """
    resized_img = cv2.resize(img, (224, 224))
    # convert 3D tensor (cv2 imgage of 3 channels) into 4D tensor with shape (1, 224, 224, 3)
    tensor = np.expand_dims(resized_img, axis=0).astype('float32')
    # return normalized 4D tensor
    return tensor/255

def imagePath2tensor(img_path):
    """
    Returns a normalized 4D tensor (suitable for supplying to a Keras CNN) associated to an image.

    Args:
        img_path (string): Path to an image (3 channels/colors)
    Returns:
        numpy.ndarray  : 4D tensor of shape (1, 224, 224, 3) containing the loded images
    """
    # loads BGR image as cv2 image type
    img = cv2.imread(img_path, 1) # 1 denotes the flag IMREAD_COLOR (BGR)
    return image2tensor(img)

def imagePaths2tensor(img_paths):
    """
    Returns a normalized 4D tensor (suitable for supplying to a Keras CNN) associated to a list of images.

    Args:
        img_paths (*string): Paths to images (3 channels/colors)
    Returns:
        numpy.ndarray      : 4D tensor of shape (samples_size, 224, 224, 3) containing the loded images
    """
    list_of_tensors = []
    for img_path in tqdm(img_paths):
        try:
            list_of_tensors += [imagePath2tensor(img_path)]
        except:
            list_of_tensors += [list_of_tensors[-1]]
    return np.vstack(list_of_tensors)

def mkdir(path):
    """
    Creates a directory if it doesn't already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)

def addSlash(path):
    """
    Add a slash at the end of a path name if it isn't there already
    """
    if not path[-1] == '/': path += '/'
    return path

def remove(path):
    """
    Removes folder or file
    """
    if os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)
    

class InriaPersonDataset(AbstractDataset):
    """
    Import and clean the INRIA Person Dataset 
    The dataset must have been downloaded and extracted first. 
    The user should also have read and write permission.
    The dataset's webpage: http://pascal.inrialpes.fr/data/human/
    """
    
    def __init__(self, source, validation_split=0.2):
        source = addSlash(source)
        self._removeUnusedFolders(source)
        self._createValidationDataFolder(source, validation_split)
        super().__init__(source)
        
    def _loadFolder(self, folder):
        """
        Loads the files in the given folder into a tensor.
        
        Args:
            folder (string): Path to an folder
        Returns:
            numpy.ndarray  : 4D tensor of shape (samples_size, 224, 224, 3) containing the loded images
            numpy.ndarray  : array containing the coresponding target labels (one-hot encoded)
        """
        data = load_files(folder)
        data['target'] -= min(data['target'])
        
        folder = addSlash(folder)
        
        nb_classes = len(glob(folder + "*"))
        self.classes = [item[len(folder):] for item in sorted(glob(folder + "*"))]
        
        files = np.array(data['filenames'])
        targets = np_utils.to_categorical(np.array(data['target']), nb_classes)
        
        # Balancing the data so that there is roughly as many images in 'neg' as in 'pos' 
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
        
        tensors = imagePaths2tensor(files)      
        return tensors, targets
        
    def _removeUnusedFolders(self, source):
        """
        Removes the folders that won't be used.
        """
        
        for folder in os.listdir(source):
            if folder != 'Test' and folder != 'Train' and folder != 'Valid':
                folder = self.source + folder
                remove(folder)

        for folder in os.listdir(source + 'Train/'):
            if folder != 'neg' and folder != 'pos':
                folder = self.source + 'Train/' + folder
                remove(folder)

        for folder in os.listdir(source + 'Test/'):
            if folder != 'neg' and folder != 'pos':
                folder = self.source + 'Test/' + folder
                remove(folder)
                    
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

class GoogleImageScrapedDataset(AbstractDataset):
    """
    Creates and loads a database using images scraped from Google image.
    This class uses a library made by hardikvasa (see https://github.com/hardikvasa/google-images-download).
    """
    
    def __init__(self, source, validation_split=0.2):
        self.source = addSlash(source)
        #response = google_images_download.googleimagesdownload()
        #self.download = response.download()

    def fillClass(self, class_name, sample_size, keywords, valid=0.2, test=0.2):
        """
        Downloads sample_size images and splits them into train/valid/test images.
        Uses the folowing hierarchy:

            self.source/
                ├── Train/
                │   └── class_name/
                ├── Valid/
                │   └── class_name/
                ├── Test/
                └── └── class_name/
    
        """
        
        mkdir(self.source)
        class_name = addSlash(class_name)

        response = google_images_download.googleimagesdownload()
        arguments = {"keywords"         : ",".join(keywords),
                     "limit"            : sample_size, 
                     "safe_search"      : True,
                     "output_directory" : self.source, 
                     "image_directory"  : "Train/" + class_name,
                     "size"             : "medium",
                     "format"           : "png",
                     "chromedriver"     : "/home/selimsepthuit/human-localisation/utilities/"}
        absolute_image_paths = response.download(arguments)
        
        mkdir(self.source + "Valid/")
        mkdir(self.source + "Test/")
        mkdir(self.source + "Valid/" + class_name)
        mkdir(self.source + "Test/" + class_name)
        
        files = list(absolute_image_paths.values())[0]
        for file in files:
            if file[-3:] != 'png':
                remove(file)
            else:
                r = np.random.rand(1)
                if r < valid:
                    shutil.move(file, self.source + "Valid/" + class_name + file.split("/")[-1])
                elif r < valid+test:
                    shutil.move(file, self.source + "Test/"  + class_name + file.split("/")[-1])
            
    def load(self):
        """
        Loads all the images into tensors and loads their targets.
        """
        self.X_train, self.y_train = self._loadFolder(self.source + "Train")
        self.X_test, self.y_test   = self._loadFolder(self.source + "Test")
        self.X_valid, self.y_valid = self._loadFolder(self.source + "Valid")


















