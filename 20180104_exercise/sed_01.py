#!/usr/bin/python
'''
License (MIT)
Copyright (c) 2016 by Magnus Erik Hvass Pedersen
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Abstract:
    This is a program to practice applying tensorflow on 01 Simple Linear Model on https://github.com/Hvass-Labs/TensorFlow-Tutorials.git
Usage:
    sed_01.py [source] [label]
    [source]: It should be a .npy file with m*n 2d array
    [label]: It should be a .npy file with m length 1d array
    hint: you can access .npy file with dat2npy.py
Example:
    sed_01.py source_sed.npy id.npy
Practicer:
    Jacob975

20170123
####################################
update log
20180123 version alpha 1
    the AI always predict 0 for all source.
20180124 version alpha 2
    the AI work properly
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import astro_mnist
from sys import argv
from help_func import  plot_images

def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
    return

def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    return

def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(num_classes, 1)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<num_classes:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i]
            # Set the label for the sub-plot.
            ax.set_title("Weights: {0}".format(i))
            # Plot the image.
            ax.plot(range(8), image)

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    return

#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    # Load data, if no data in given path, download data
    images_name = argv[1]
    labels_name = argv[2]
    data = astro_mnist.read_data_sets(images_name, labels_name)
    # check the type of data
    print "Size of:"
    print "- Training-set:\t\t{}".format(len(data.train.labels))
    print "- Test-set:\t\t{}".format(len(data.test.labels))
    print "- Validation-set:\t{}".format(len(data.validation.labels))
    # check labels
    print data.test.labels[0:5, :]                                                  # value in a row means the probability of matched class of data.
    data.test.cls = np.array([label.argmax() for label in data.test.labels])        
    print data.test.cls[0:5]                                                        # the number means which class is the matched class of this data.
    
    # We know that MNIST images are 28 pixels in each dimension.
    img_size = 8
    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * 1
    # Tuple with height and width of images used to reshape arrays.
    img_shape = (1, img_size)
    # Number of classes, one class for each of 10 digits.
    num_classes = 3
    # Number of neurals
    num_neurals = 128
    #--------------------------------------------------------------
    # exercise 1: plot 9 image and their labels.
    # Get the first images from the test-set.
    images = data.test.images[0:9]
    # Get the true classes for those images.
    cls_true = data.test.cls[0:9]
    # Plot the images and labels using our helper-function above.
    plot_images(images=images, cls_true=cls_true)
    #--------------------------------------------------------------
    # exercise 2: run deep learning
    # 1. placeholder, used to save input data
    x = tf.placeholder(tf.float32, [None, img_size_flat])
    y_true = tf.placeholder(tf.float32, [None, num_classes])
    y_true_cls = tf.placeholder(tf.int64, [None])
    # 2. model variable, used to save weight or so called, parameters.
    weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))
    # 3. model, mathematical statement
    logits = tf.matmul(x, weights) + biases
    #-----------------------------------------------
    # output
    # result
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)          # take the order of largest number as the answer
    # 4. Cost-function to be optimized
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    # 5. Optimization method
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 6. ready to run
    # the code won't work until session cmd.
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    batch_size = 200
    feed_dict_test = {x: data.test.images, y_true: data.test.labels, y_true_cls: data.test.cls}
    # before run
    print "before run"
    print_accuracy()
    plot_example_errors()
    # run for once
    optimize(num_iterations=1)
    print "iterate for 1 times"
    print_accuracy()
    plot_example_errors()
    plot_weights()
    # run for 10 times
    optimize(num_iterations=9)          # We have already performed 1 iteration.
    print "iterate for 10 times"
    print_accuracy()
    plot_example_errors()
    plot_weights()
    # run for 1000 times
    optimize(num_iterations=990)          # We have already performed 10 iteration.
    print "iterate for 1k times"
    print_accuracy()
    plot_example_errors()
    plot_weights()
    # run for 10k times
    optimize(num_iterations=9000)          # We have already performed 1k iteration.
    print "iterate for 10k times"
    print_accuracy()
    plot_example_errors()
    plot_weights()
    print_confusion_matrix()
    
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print "Exiting Main Program, spending ", elapsed_time, "seconds."
