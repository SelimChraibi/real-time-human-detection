from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import SGD, adam

def get_model(nb_classes, summary=False):
    """
    Returns a compiled model (and its last convolutinoal layer and dense layer name).
    Displays the summary if summary==True
    """
   
    # ---------------------------------------------------------------
    model = VGG16(include_top=False,
                      weights='imagenet',
                      classes=nb_classes,
                      pooling='avg',
                      input_shape=(224, 224, 3))

    # We freeze the layers we don't want to train
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers[:]:
        layer.trainable = False    
        
    # Adding custom Layers 
    x = model.output    
    x = Dense(nb_classes, 
              activation = 'softmax', 
              kernel_regularizer=l2(0.05),
              name='dense_layer')(x)

    # We create our new model
    new_model = Model(inputs = model.input, outputs = x)
    
    # ---------------------------------------------------------------
    
    if summary:
        new_model.summary()

    # Compilation of the Model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    new_model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
    
    return new_model, new_model.layers[-3].name, model.layers[-1].name