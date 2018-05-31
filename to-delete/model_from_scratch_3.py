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
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=512, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=512, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=3, name='final_conv_layer'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    # The model so far outputs 3D feature maps (height, width, features)
    
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes, activation = 'softmax', kernel_initializer='uniform', 
                    kernel_regularizer=l2(0.01), name='dense_layer'))
    
    # ---------------------------------------------------------------
    
    if summary:
        model.summary()

    # Compilation of the Model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
    
    return model, "final_conv_layer", "dense_layer"