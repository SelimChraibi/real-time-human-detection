from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import math
import shutil
import os

class Classifier():
    """
    
    """
    model         = None
    training_log  = None
    _name         = '' 
    _datagen      = None
    _checkpointer = None
    _tensorboard  = None
    _checkpointer_settings = dict(verbose            = 0,
                                  save_best_only     = True,
                                  save_weights_only  = True)
    _tensorboad_settings   = dict(log_dir            = './tensorboard_logs', 
                                  histogram_freq     = 0,
                                  write_graph        = True, 
                                  write_images       = True)
    _datagen_settings     = dict(width_shift_range  = 0.3,
                                  height_shift_range = 0.3,
                                  shear_range        = 0.3,
                                  zoom_range         = 0.3,
                                  horizontal_flip    = True,
                                  fill_mode          = "wrap")
    
    
    def __init__(self, model, name='model'):
        self.model = model
        self._name = name
        self._checkpointer_settings['filepath'] = 'saved_model/' + self._name +'.h5'
        self._checkpointer = ModelCheckpoint(**self._checkpointer_settings)
        self._datagen      = ImageDataGenerator(**self._datagen_settings)
        self._tensorboard  = TensorBoard(**self._tensorboad_settings)
        
    def fit(self, data, epochs=5, batch_size=32, augmentation=False, verbose=0):
        
        if not os.path.exists('saved_model/'): os.makedirs('saved_model/')
        shutil.rmtree('./tensorboard_logs')
        
        if augmentation:
            # Compute quantities required for feature normalization (if used)
            self._datagen.fit(train_tensors)

            # Real-time data generator
            generator = self._datagen.flow(train_tensors, train_targets, batch_size=batch_size)

            # Training: fits the model on batches with real-time data augmentation
            self.training_log = self.model.fit_generator(generator= generator,
                                               validation_data    = (data.X_valid, data.y_valid),
                                               epochs             = epochs,
                                               verbose            = verbose,
                                               steps_per_epoch    = data.X_train.shape[0] // batch_size,
                                               validation_steps   = data.X_valid.shape[0] // batch_size,
                                               callbacks          = [self._checkpointer, self._tensorboard])
        if not augmentation:
            # Training
            self.training_log = self.model.fit(x                  = data.X_train,
                                               y                  = data.y_train,
                                               validation_data    = (data.X_valid, data.y_valid),
                                               epochs             = epochs,
                                               batch_size         = batch_size,
                                               verbose            = verbose,
                                               callbacks          = [self._checkpointer, self._tensorboard])
        
        # Save complete model (not only its weights)
        self.model.save('saved_model/' + self._name +'.h5')
     
    def learning_curves(self):
        """
        This method plots the learning_curves saved in learning curves of the model if it has been trained.
        """
        if self.training_log is None:
            print("The model hasen't been trained yet.")
        else:
            history = self.training_log.history
            
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
                # fig.savefig('saved_learning_curves/' + self._name +'.png')

                plt.suptitle(self._name)
                plt.show()


        
        
        