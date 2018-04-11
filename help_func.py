#!/usr/bin/python3
'''
Abstract:
    This is a program to save some common help func. 
Usage:
    import help_func
Editor:
    Jacob975

20170123
####################################
update log
20180123 version alpha 1
    add plot_image
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

# the def is used to plot data and their labels
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.plot(range(len(images[i])), images[i])

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    return
