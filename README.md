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
[softmax.py](https://github.com/EN10/MNIST/blob/master/softmax.py) from https://www.tensorflow.org/get_started/mnist/beginners

Report Accuracy [print.py](https://github.com/EN10/MNIST/blob/master/print.py) 

**TensorBoard Demo**  
[board.py](https://github.com/EN10/MNIST/blob/master/board.py) TensorBoard Image [image.py](https://github.com/EN10/MNIST/blob/master/MNISTimage.py)  
https://www.tensorflow.org/get_started/summaries_and_tensorboard  
needs to run in python before tensorboard  
`python board.py`  
`tensorboard --logdir=/tmp/mnist_logs --port 8080`  
tensorboard defaut port 6006 may not be open  

**Predict from Image**  
[predict.py](https://github.com/EN10/MNIST/blob/master/predict.py)
requires :  
`sudo pip install --upgrade scipy`  
`sudo pip install pillow`  
based on [softmax.py](https://github.com/EN10/MNIST/blob/master/softmax.py) 
L25 based on [cnnPredict.py](https://github.com/EN10/KerasMNIST/blob/master/cnnPredict.py) L3-11  
trained on `mnist.train.images[0]` vs image `imread('test3.png',mode='L')`  

[expert.py](https://github.com/EN10/MNIST/blob/master/MoreExamples/expert.py) from https://www.tensorflow.org/get_started/mnist/pros  

#### Older Examples

[2L.py](https://github.com/EN10/MNIST/blob/master/MoreExamples/2L.py)    ReLu, Random W 
2 Layer Feed-Forward Neural Network     
http://stackoverflow.com/questions/38136961/how-to-create-2-layers-neural-network-using-tensorflow-and-python-on-mnist-data  
https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd  

[ply.py](https://github.com/EN10/MNIST/blob/master/MoreExamples/plt.py) on ipynb, view image data. See [installJupyter.txt](https://github.com/EN10/MNIST/blob/master/installJupyter.txt)   
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