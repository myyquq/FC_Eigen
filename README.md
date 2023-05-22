# FC_Eigen

C++ implementation of fully connected neural networks library with *Keras*-like API. Contains commonly used layers, losses. Supports sequential models. The underlying data structure is based on [Eigen](https://eigen.tuxfamily.org/). Currently tested on *MNIST* dataset. 

Supported layers:

* Input
* Dense
* Dropout
* Activation

Supported optimizers:  
* SGD

Supported Losses:

- Mean Absolute Error
- Mean Squared Error
- Binary Cross Entropy

Supported Activations:

- ReLU
- Sigmoid
- Softmax
- Tanh
- Linear