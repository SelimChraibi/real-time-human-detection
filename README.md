
# Real Time Human Detection

<img src="https://media.giphy.com/media/3JU7xOanvSnCBKbUhK/giphy.gif" alt="drawing" width="1000"/>


<img src="https://i.imgur.com/RCcHhj8.jpg" alt="drawing" width="1000"/>

Neural Networks are often described as black boxes. In this project, however, I used a method that is based on the **interpretation of the internal parameters of a neural network** to implement an application capable of:
- Detecting the presence of humans in a live video
- Identifying *"regions of interest"* where the individuals detected are most likely to be situated

This method was introduced in the following paper: [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf).

Its originality comes from the simplicity of both the network it uses and of the training it requires.

- An explanation of this method and of the specific problem it is applied to in this project can be found in the [`Project Report`](https://selim78.github.io/real-time-human-detection/) page.

[![page](https://i.imgur.com/XP6aiLH.png)](https://selim78.github.io/real-time-human-detection/)

- A demonstration of how the code shared in this repo can be used (to create classification models capable of outputting localisation information) can be found in the jupyter notebook [`demo.ipynb`](https://nbviewer.jupyter.org/github/Selim78/real-time-human-detection/blob/master/demo.ipynb).

[![ipnb](https://i.imgur.com/mYZfwXl.png)](https://nbviewer.jupyter.org/github/Selim78/real-time-human-detection/blob/master/demo.ipynb)

## Application using the trained models

Launching the live webcam application:

``` sh
python3 webcam_cam.py --model ./saved_model/mobilenet_with_gi_data.h5
```

>**Requires:**
>Tensorflow version     >= 1.7
>Keras version         >= 2.1
>OpenCV version         >= 3.3
