from keras.applications.mobilenet import MobileNet
from keras.models import load_model
from keras.layers import Dense
from keras.regularizers import l2
# from keras.optimizers import SGD 

class MobileNetGAP():
    """
    Helper class to build and compile or load MobileNetGAP models.
    
    Attributes:
        model (keras.models): a Mobilenet (top off) followed by a dense layer with nb_classes neurones
        input_shape  (tuple): (224, 224, 3) for example for a 3 channel image of size 224*224
    """
    model                = None
    input_shape          = (0, 0, 0)
    last_conv_layer_name = 'conv_pw_13_relu'
    dense_layer_name     = 'dense_layer'
    
    
    def __init__(self, nb_classes=None, path=None, input_shape=(224, 224, 3)):
        self.input_shape = input_shape 
        if nb_classes:
            self._get_model(nb_classes)
        elif path:
            self.model = _load_model(path)
        else:
            print('Must be specified either:)
            print(' - a path from which to load the model')
            print(' - a number of class to create the model')
    
    def _get_model(self, nb_classes):
        """
        Creates and compiles a MobileNet GAP model.
        Args:
            nb_classes    (int): number of classes the model is supposed to classify
        """

        model = MobileNet(include_top  = False,
                          weights      = 'imagenet',
                          classes      = nb_classes,
                          input_shape  = input_shape,
                          pooling      = 'avg',
                          alpha        = 0.5)

        # We freeze the layers we don't want to train
        for layer in model.layers:
            layer.trainable = True
        for layer in model.layers[:2]:
            layer.trainable = False    

        # Adding custom Layers 
        x = model.output    
        x = Dense(nb_classes, 
                  activation         = 'softmax', 
                  kernel_regularizer = l2(0.1),
                  name               = 'dense_layer')(x)

        # We create a new model
        new_model = Model(inputs = model.input, outputs = x)

        # Compilation
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        new_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        
        self.model = new_model


    def _load_model(self, path)
        """
        Loads the MobileNet from a saved model.
        Args:
            path (string): path to an h5 file
        """
        self.model = load_model(path, custom_objects={
                           'relu6': mobilenet.relu6,
                           'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    
    
    
    
    