# deep learning
I study on how to apply deep_learning on astronomy.

# preparation
The module you need
1. tensorflow
2. sklearn
3. tkinter
4. IPython
5. prettytensor
4. TBA

If you want to install tensorflow with gpu compiled version, you need
1. bazel

The git you need
1. TensorFlow-Tutorials

# tutorial I have practiced
TensorFlow-Tutorials/01_Simple_Linear_Model.ipynb

TensorFlow-Tutorials/02_Convolutional_Neural_Network.ipynb

TensorFlow-Tutorials/03_PrettyTensor.ipynb

TensorFlow-Tutorials/04_Save_Restore.ipynb

TensorFlow-Tutorials/05_Ensemble_Learning.ipynb

TensorFlow-Tutorials/06_CIFAR-10.ipynb

# How to train AI

Usage:
    sed_04_64_8.py [source] [id]

# Result tree

+ `week, date. Month Year hh:mm`
  + test
    + index		// tracer index
    + labels		// true label
    + data set		// data
  - training
    + index             // tracer index
    + labels            // true label
    + data set          // data
  - validation
    + index             // tracer index
    + labels            // true label
    + data set          // data
  + cls true of test                        // predicted label of test set
  + cls pred of test                        // true label of test set
  + checkpoint_AI_64_8_`file_name`          // this is AI

# Tracer tree

![Data selection](https://github.com/jacob975/deep_learning/blob/master/data_selection.png)

![Data tracer](https://github.com/jacob975/deep_learning/blob/master/data_tracer.png)
