# FC_Eigen

C++ implementation of fully connected neural networks library with *Keras*-like API. 
Contains commonly used layers, losses. Supports sequential models. 
The underlying data structure is based on [Eigen](https://eigen.tuxfamily.org/). 
Currently tested on *MNIST* dataset. 

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
  
**Test results:**
```     
large batch size, slow convergence:
     Dense 128 0.5 -> sigmoid -> Dropout 0.2 -> Dense 64 0.2 -> sigmoid -> Dropout 0.2 -> Dense 10 0.1-> softmax, 250 epochs, 1024 batch size
           accuracy: approximately 91%
     Dense 128 0.5 -> sigmoid -> Dropout 0.5 -> Dense 10 0.2-> softmax, 250 epochs, 1024 batch size
           accuracy: approximately 92%
     Dense 128 0.5 -> relu -> Dropout 0.2 -> Dense 32 0.2 -> relu -> Dropout 0.2 -> Dense 10 0.2-> softmax, 150 epochs, 1024 batch size
           accuracy: approximately 93%
     Dense 128 0.5 -> relu -> Dropout 0.5 -> Dense 10 0.2-> softmax, 250 epochs, 1024 batch size
           accuracy: approximately 94%
           
small batch size, fast convergence, but easily overfit:
     Dense 128 0.5 -> sigmoid -> Dropout 0.5 -> Dense 10 0.2-> softmax, 100 epochs, 32 batch size
           accuracy: approximately 96%
     Dense 128 0.5 -> relu -> Dropout 0.5 -> Dense 10 0.2-> softmax, 100 epochs, 32 batch size
           accuracy: approximately 97%
```