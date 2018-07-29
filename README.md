
# Live Person Detection

![Screen Shot 2018-07-22 at 12.17.35](https://i.imgur.com/RCcHhj8.jpg)

Neural Networks are often described as black boxes. In this project, however, we will use a method that is based on the **interpretation of the internal parameters of a neural network** to implement an application capable of:
- Detecting the presence of humans in a live video
- Identifying *"regions of interest"* where the individuals detected are most likely to be situated

This method was introduced in the following paper: [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf).

Its originality is the simplicity of the network it uses and of the training it requires.

- An explanation of this method and of the specific problem it is applied to in this project can be found in `Project Report.pdf`.

- A demonstration of how the code shared in this repo can be used to create classification models capable of outputting localisation information can be found in the jupyter notebook `demo.ipynb`.

## Application using the trained models

Launching the live webcam application:

``` sh
python3 webcam_cam.py --model ./saved_model/mobilenet_with_gi_data.h5
```

>**Requires:**
>Tensorflow version     >= 1.7
>Keras version         >= 2.1
>OpenCV version         >= 3.3
