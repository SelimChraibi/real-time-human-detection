from keras.applications import mobilenet
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, GlobalMaxPooling2D
from keras.regularizers import l2
# from keras.optimizers import SGD 

class AbstractModelGAP():
    """
    Abstract class. 
    Instances of the subclass can be used as keras.models. 
    When instanciated, the subclass will compile or load a custom model.
    
    Subclasses 
    - have to define _get_model 
    - may have to redefine _load_model (if custom_objects are needed to load the model)
    - should assign the last_conv_layer_name and dense_layer_name attributes, as this will be used during CAM generation. 
      (Otherwise the Classifier.cam() method will try to infer those names.)
    
    Attributes:
        model          (keras.models): a keras.applications model (top off) followed by a dense layer with nb_classes neurones
        input_shape           (tuple): (224, 224, 3) for example for a 3 channel image of size 224*224
        last_conv_layer_name (string): useful to access the last convolutional layer of the model for CAM generation
        dense_layer_name     (string): useful to access the dense layer of the model for Class Aactivation Map generation
    """
    model                = None
    input_shape          = None
    last_conv_layer_name = None
    dense_layer_name     = None
    
    def __init__(self, nb_classes=None, path=None, input_shape=(224, 224, 3)):
        self.input_shape = input_shape 
        if nb_classes is not None:
            self._get_model(nb_classes)
        elif path is not None:
            self._load_model(path)
        else:
            print('Must be specified either:')
            print(' - a path from which to load the model')
            print(' - a number of class to create the model')
            
    def _get_model(self, nb_classes):
        pass


    def _load_model(self, path):
        """
        Loads from a saved model.
        
        Args:
            path (string): path to an h5 file
        """
        self.model = load_model(path)
    
    # Delegation (this allows the class' object to be used as keras models)
    def __getattr__(self, attr):
        return getattr(self.model, attr)
    
    
class MobileNetGAP(AbstractModelGAP):
    """
    Instances of this class can be used as keras.models. 
    When instanciated, this class will compile or load a custom MobileNet model, 
    top not included, followed by a GAP layer and a dense layer.
    
    Attributes:
        model          (keras.models): a Mobilenet (top off) followed by a dense layer with nb_classes neurones
        input_shape           (tuple): (224, 224, 3) for example for a 3 channel image of size 224*224
        last_conv_layer_name (string): useful to access the last convolutional layer of the model for CAM generation
        dense_layer_name     (string): useful to access the dense layer of the model for Class Aactivation Map generation
    """
    
    def __init__(self, nb_classes=None, path=None, input_shape=(224, 224, 3)):
        super().__init__(nb_classes=nb_classes, path=path, input_shape=input_shape) 
        self.last_conv_layer_name = 'conv_pw_13_relu'
        self.dense_layer_name     = 'dense_layer'
    
    def _get_model(self, nb_classes):
        """
        Creates and compiles a MobileNet GAP model.
        Args:
            nb_classes    (int): number of classes the model is supposed to classify
        """

        model = mobilenet.MobileNet(include_top  = False,
                                    weights      = 'imagenet',
                                    classes      = nb_classes,
                                    input_shape  = self.input_shape,
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
        

    def _load_model(self, path):
        """
        Loads the MobileNet from a saved model.
        Args:
            path (string): path to an h5 file
        """
        self.model = load_model(path, custom_objects={
                           'relu6': mobilenet.relu6,
                           'DepthwiseConv2D': mobilenet.DepthwiseConv2D})