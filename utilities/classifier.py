from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import math
import cv2
import shutil
import os
from . import datasets
from .helpers import Time

class Classifier():
    """
    Can be fitted to a dataset (from the dataset library) and predict the class of an image.
    
    Attributes:
        model (keras.models): a model, already compiled
    """
    
    model            = None
    _cam_model       = None
    _training_log    = None
    _name            = None
    _datagen         = None
    _checkpointer    = None
    _tensorboard     = None
    _early_stopping  = None
    _checkpointer_settings   = dict(save_best_only      = True,
                                  save_weights_only   = True)
    _tensorboad_settings     = dict(log_dir           = './tensorboard_logs', 
                                  histogram_freq      = 0,
                                  write_graph         = True, 
                                  write_images        = True)
    _early_stopping_settings = dict(monitor           = 'val_loss', 
                                  patience            = 5)
    _datagen_settings        = dict(width_shift_range = 0.2,
                                  height_shift_range  = 0.2,
                                  shear_range         = 0.2,
                                  zoom_range          = 0.2,
                                  horizontal_flip     = True,
                                  fill_mode           = "wrap")
    
    
    def __init__(self, model, name='model'):
        self.model = model
        self._name = name
        self._checkpointer_settings['filepath'] = 'saved_model/' + self._name +'.h5'
        self._checkpointer   = ModelCheckpoint(**self._checkpointer_settings)
        self._tensorboard    = TensorBoard(**self._tensorboad_settings)
        self._early_stopping = EarlyStopping(**self._early_stopping_settings)
        self._datagen        = ImageDataGenerator(**self._datagen_settings)
        
        # Create the model that we will use to generate the class activation maps
        self._find_layers()
        self._create_cam_model()
        
    def fit(self, data, epochs=5, batch_size=32, augmentation=False, verbose=0):
        """
        Fits the model to the data, computes the validation loss and accuracy at each epoch.
        Generates the tensorbord_logs directory.
        
        Args:
            data (AbstractDataset): Dataset object generated using the dataset library.
            epochs           (int): Number of epochs to train the model. An epoch is an iteration over the entire training data. 
            batch_size       (int): Number of samples per gradient update. 
            augmentation    (bool): If True, the training will use data augmentation.
            verbose          (int): (0 = silenced) (1 = progress bar) (2 = minimal display at each epoch)
            
        """
        if   not os.path.exists('saved_model/'): os.makedirs('saved_model/')
        if os.path.exists('./tensorboard_logs'): shutil.rmtree('./tensorboard_logs')
        self._checkpointer_settings['verbose'] = verbose
        
        if augmentation:
            # Compute quantities required for feature normalization (if used)
            self._datagen.fit(data.X_train)

            # Real-time data generator
            generator = self._datagen.flow(data.X_train, data.y_train, batch_size=batch_size)
            

            # Training: fits the model on batches with real-time data augmentation
            self._training_log = self.model.fit_generator(generator= generator,
                                                validation_data    = (data.X_valid, data.y_valid),
                                                epochs             = epochs,
                                                verbose            = verbose,
                                                steps_per_epoch    = data.X_train.shape[0] // batch_size,
                                                validation_steps   = data.X_valid.shape[0] // batch_size,
                                                callbacks          = [self._checkpointer, 
                                                                      self._tensorboard, 
                                                                      self._early_stopping])
        if not augmentation:
            # Training
            self._training_log = self.model.fit(x                  = data.X_train,
                                                y                  = data.y_train,
                                                validation_data    = (data.X_valid, data.y_valid),
                                                epochs             = epochs,
                                                batch_size         = batch_size,
                                                verbose            = verbose,
                                                callbacks          = [self._checkpointer, 
                                                                      self._tensorboard, 
                                                                      self._early_stopping])
        
        # Save complete model (not only its weights)
        self.model.save('saved_model/' + self._name +'.h5')
        
        # Create the model that we will use to generate the class activation maps
        self._create_cam_model()
        
    def predict(self, img, decision=True):
        """
        Predict the class of an image.
        Args:
            img (cv2 BGR image): Image to be classified
            decision     (bool): If true returns the prediction [0, 1, 0, 0, 1...] else returns the probabilities.
        Returns:
            list: prediction
        """
        tensor = datasets.image2tensor(img)
        prediction = self.model.predict(tensor)[0]
        if decision: 
            prediction = map(lambda x: [0, 1][x>0.5], prediction)
        return list(prediction)
        
     
    def learning_curves(self):
        """
        This method plots the learning_curves saved in learning curves of the model if it has been trained.
        """
        if self._training_log is None:
            print("The model needs to be trained first.")
        else:
            history = self._training_log.history
            
            with plt.style.context('seaborn-darkgrid'):
                
                fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 5))
                
                # Plot accuracy
                ax1.plot(history['acc'])
                ax1.plot(history['val_acc'])
                ax1.set_title('Model Accuracy')
                ax1.set_ylabel('accuracy')
                ax1.set_xlabel('epoch')
                ax1.legend(['training', 'validation'], loc='upper left')
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

                # Plot loss
                ax2.plot(history['loss'])
                ax2.plot(history['val_loss'])
                ax2.set_title('Model Loss')
                ax2.set_ylabel('loss')
                ax2.set_xlabel('epoch')
                ax2.legend(['training', 'validation'], loc='upper left')
                ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

                plt.suptitle(self._name)
                plt.show()

    def cam(self, img, class_number):
        """
        Creates the Class Activation Maps associated to the model for a given class and image.
        """
        
        tensor = datasets.image2tensor(img)

        # Get the input weights to the dense layer.
        class_weights = self._get_layer(self.model.dense_layer_name).get_weights()[0]

        # The _cam_model outputs the output of the final conv layer given the tensor input of the 1st layer
        conv_outputs, prediction = self._cam_model.predict(tensor)

        # Only one image so we don't need the first dimension
        conv_outputs = conv_outputs[0, :, :, :]

        # Original input image to which we will add the cam
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width, height, _ = img.shape

        # ---------------- Creation of the class activation map ----------------

        # Initialize with the cam with right shape
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])

        # Weighted sum of the cam = \Sum_i (cam_i * weight_i)
        for i, w in enumerate(class_weights[:, class_number]):
                cam += w * conv_outputs[:, :, i]

        # We normalise and resize the cam
        cam /= np.max(cam)
        cam  = cv2.resize(cam, (height, width), interpolation = cv2.INTER_CUBIC)

        # We transform it into a heatmap
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0

        # And add it to the original input image
        img = cv2.addWeighted(heatmap, 0.5, img, 0.8, 0)
        
        # One image so one prediction -> prediction[0]
        return img, prediction[0]
    
    def _find_layers(self):
        """
        Finds the dense_layer_name and last_conv_layer_name if they are not already defined
        """
        if self.model.dense_layer_name is None or self.model.last_conv_layer_name is None:
            layer_names = [layer.name for layer in self.model.layers]
            layer_name = ''
            while 'dense' not in layer_name:
                layer_name = layer_names.pop()
            self.model.dense_layer_name = layer_name
            while 'conv' not in layer_name:
                layer_name = layer_names.pop()
            self.model.last_conv_layer_name = layer_name
    
    def _get_layer(self, layer_name):
        """
        Returns the layer with the name 'layer_name'
        """
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        layer = layer_dict[layer_name]
        
        return layer
    
    def _create_cam_model(self):
        """
        This will update the '_cam_model' attribute.
        Given an input tensor, this model will output 
        - the output of the final conv layer 
        - the output of the original model (the prediction vector)
        """
        
        # We retrieve the final conv layer and the dense layer
        last_conv_layer  = self._get_layer(self.model.last_conv_layer_name)
        dense_layer      = self._get_layer(self.model.dense_layer_name)
            
        self._cam_model = Model(inputs=self.model.layers[0].input, 
                                outputs=(last_conv_layer.output, dense_layer.output))
            