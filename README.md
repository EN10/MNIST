# MNIST

## Tensorflow examples using MNIST.  

**How to Install TensorFlow using pip**  
https://www.tensorflow.org/install/install_linux#determine_how_to_install_tensorflow

Tested on cs50.io

    sudo pip install -U pip 
    sudo pip install tensorflow

disable "cpu_feature_guard":  
`export TF_CPP_MIN_LOG_LEVEL=2`

**MNIST For ML Beginners**  
[MNISTsoftmax.py](https://github.com/EN10/MNIST/blob/master/MNISTsoftmax.py) from https://www.tensorflow.org/get_started/mnist/beginners

Report Accuracy [MNISTprint.py](https://github.com/EN10/MNIST/blob/master/MNISTprint.py) 

**TensorBoard Demo**  
[MNISTboard.py](https://github.com/EN10/MNIST/blob/master/MNISTboard.py) TensorBoard Image [MNISTimage.py](https://github.com/EN10/MNIST/blob/master/MNISTimage.py)  
https://www.tensorflow.org/get_started/summaries_and_tensorboard  
needs to run in python before tensorboard  
`python MNISTboard.py`  
`tensorboard --logdir=/tmp/mnist_logs --port 8080`  
tensorboard defaut port 6006 may not be open  

**Predict from Image**  
[predict.py](https://github.com/EN10/MNIST/blob/master/predict.py)
based on [MNISTsoftmax.py](https://github.com/EN10/MNIST/blob/master/MNISTsoftmax.py) 
L25 based on [cnnPredict.py](https://github.com/EN10/KerasMNIST/blob/master/cnnPredict.py) L3-11  
requires : `sudo pip install --upgrade scipy`

[MNISTexpert.py](https://github.com/EN10/MNIST/blob/master/MNISTexpert.py) from https://www.tensorflow.org/get_started/mnist/pros  

[MNIST-2L.py](https://github.com/EN10/MNIST/blob/master/MNIST-2L.py)    ReLu, Random W 
2 Layer Feed-Forward Neural Network     
http://stackoverflow.com/questions/38136961/how-to-create-2-layers-neural-network-using-tensorflow-and-python-on-mnist-data  
https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd  

[MNISTply.py](https://github.com/EN10/MNIST/blob/master/MNISTplt.py) on ipynb, view image data. See [installJupyter.txt](https://github.com/EN10/MNIST/blob/master/installJupyter.txt)   
Original:   
https://github.com/random-forests/tutorials/blob/master/ep7.ipynb   
https://www.youtube.com/watch?v=Gj0iyo265bc     
Updated:    
http://tneal.org/post/tensorflow-ipython/TensorFlowMNIST/    
http://stackoverflow.com/questions/36651704/which-cmap-colormap-to-use-with-matplotlib-imshow-to-diplay-the-mnist-datase    

Predict from Canvas see In [8]:   
https://github.com/dmlc/mxnet-notebooks/blob/master/python/tutorials/mnist.ipynb

MNIST for Handwriting Prediction:   
https://github.com/niektemme/tensorflow-mnist-predict 