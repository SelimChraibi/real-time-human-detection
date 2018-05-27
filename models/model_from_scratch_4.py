from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD

def get_model(nb_classes, summary=False):
    """
    Returns a compiled model.
    Displays the summary if summary==True
    """
   
    input_shape=(224, 224, 3)
    
    # ---------------------------------------------------------------
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='final_conv_layer'))
    
    # The model so far outputs 3D feature maps (height, width, features)
    
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes, activation = 'softmax', name='dense_layer'))
    
    # ---------------------------------------------------------------
    
    if summary:
        model.summary()

    # Compilation of the Model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
    
    return model, "final_conv_layer", "dense_layer"