from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Dropout
from keras.regularizers import l2

def get_model(nb_classes, summary=False):
    """
    Returns a compiled model.
    Displays the summary if summary==True
    """
    
    # ---------------------------------------------------------------
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=3, name='final_conv_layer'))
    model.add(Activation('relu'))

    # The model so far outputs 3D feature maps (height, width, features)

    model.add(GlobalAveragePooling2D())
    model.add(Dense(nb_classes, activation = 'softmax', kernel_initializer='uniform', 
                    kernel_regularizer=l2(0.01), name='dense_layer'))

    # ---------------------------------------------------------------
    
    if summary:
        model.summary()

    # Compilation of the Model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model