import subprocess

def cmd(command):
    """
    Executes bash command
    """
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


# --------------------------------------------------------------

from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def Time():
    """
    Measures and yields elapsed time
    """
    start = default_timer()
    elapsed = lambda: default_timer() - start
    yield lambda: elapsed()
    end = default_timer()
    elapsed = lambda: end-start
    
# --------------------------------------------------------------
    
import tensorflow, keras, cv2
from keras import backend as K

def check():
    """
    Check dependencies and wether a GPU is being used
    """
    try:
        version = float(tensorflow.__version__[:3])
        assert(version>=1.7)
        print('Tensorflow version \t= {} [ok]'.format(version))
    except AssertionError: 
        print('Tensorflow version \t= {} < 1.7 \nTry running: conda install defaults::keras conda-forge::tensorflow\n'.format(version))
        
    try:
        version = float(keras.__version__[:3])
        assert(version>=2)
        print('Keras version \t\t= {} [ok]'.format(version))
    except AssertionError: 
        print('Keras version \t\t= {} < 2 \nTry running: conda install defaults::keras conda-forge::tensorflow\n'.format(version))
    
    try:
        version = float(cv2.__version__[:3])
        assert(version>=3)
        print('OpenCV version \t\t= {} [ok]'.format(version))
    except AssertionError: 
        print('OpenCV version \t\t= {} < 3 \nTry running: conda install opencv\n'.format(version))
   
    print('\n{} GPU detected'.format(len(K.tensorflow_backend._get_available_gpus())))
    
# --------------------------------------------------------------

HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'