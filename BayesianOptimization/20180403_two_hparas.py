#!/usr/bin/python3
'''
Abstract:
    This is a program to exercise how to optimize deep learning with Bayesian Optimization.
    Copy from "BayesianOptimization/examples/exploitation vs exploration.ipynb" 
Usage:
    20180403_two_hparas.py
Source:
    BayesianOptimization/examples/exploitation vs exploration.ipynb

##################################
#   Python3                      #
#   This code is made in python3 #
##################################

20170403
####################################
update log
20180403 version alpha 1:
    1. I don't know

'''
# modules for Bayesian
from bayes_opt import BayesianOptimization
import pymc as pm
# modules for deep learning
import tensorflow as tf
# common modules
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.core.pylabtools import figsize

# Utility function for plotting
def plot_bo(f, bo, figname):
    xs = [x["x"] for x in bo.res["all"]["params"]]
    ys = bo.res["all"]["values"]

    mean, sigma = bo.gp.predict(np.arange(len(f)).reshape(-1, 1), return_std=True)
    
    plt.figure(figsize=(16, 9))
    plt.plot(f)
    plt.plot(np.arange(len(f)), mean)
    plt.fill_between(np.arange(len(f)), mean+sigma, mean-sigma, alpha=0.1)
    plt.scatter(bo.X.flatten(), bo.Y, c="red", s=50, zorder=10)
    plt.xlim(0, len(f))
    plt.ylim(f.min()-0.1*(f.max()-f.min()), f.max()+0.1*(f.max()-f.min()))
    plt.savefig(figname)
    return

#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    #-----------------------------------
    # load hyperparas
    # use sklearn's default parameters for theta and random_start
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    # Target function
    np.random.seed(42)
    xs = np.linspace(-2, 10, 10000)
    f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2/10) + 1/ (xs**2 + 1)
    if VERBOSE>0:
        plt.plot(f)
        plt.show()
    #-----------------------------------
    # Acquisition function 1: Upper Confidence Bound
    # Prefer exploitation (kappa=1.0)
    bo = BayesianOptimization(f=lambda x: f[int(x)],
                              pbounds={"x": (0, len(f)-1)},
                              verbose=0)
    bo.maximize(init_points=2, n_iter=25, acq="ucb", kappa=1, **gp_params) 
    plot_bo(f, bo, "ucb_exploitation.png")
    # Prefer exploration (kappa=10)
    bo = BayesianOptimization(f=lambda x: f[int(x)],
                              pbounds={"x": (0, len(f)-1)},
                              verbose=0)
    bo.maximize(init_points=2, n_iter=25, acq="ucb", kappa=10, **gp_params) 
    plot_bo(f, bo, "ucb_exploration.png")
    #-----------------------------------
    # Acquisition function 2: Expected Improvement
    # Prefer exploitation (xi=0.0)   
    bo = BayesianOptimization(f=lambda x: f[int(x)],
                              pbounds={"x": (0, len(f)-1)},
                              verbose=0)
    bo.maximize(init_points=2, n_iter=25, acq="ei", xi=1e-4, **gp_params) 
    plot_bo(f, bo, "ei_exploitation.png")

    # Prefer exploration (xi=0.1)
    bo = BayesianOptimization(f=lambda x: f[int(x)],
                              pbounds={"x": (0, len(f)-1)},
                              verbose=0)
    bo.maximize(init_points=2, n_iter=25, acq="ei", xi=0.1, **gp_params) 
    plot_bo(f, bo, "ei_exploration.png")
    #-----------------------------------
    # Acquisition function 3: Probability of Improvement
    # Prefer exploitation (xi=0.0)
    bo = BayesianOptimization(f=lambda x: f[int(x)], pbounds={"x": (0, len(f)-1)}, verbose=0)
    bo.maximize(init_points=2, n_iter=25, acq="poi", xi=1e-4, **gp_params)
    plot_bo(f, bo, "poi_exploitation.png")
    # Prefer exploration (xi=0.1)
    bo = BayesianOptimization(f=lambda x: f[int(x)], pbounds={"x": (0, len(f)-1)}, verbose=0)
    bo.maximize(init_points=2, n_iter=25, acq="poi", xi=0.1, **gp_params)
    plot_bo(f, bo, "poi_exploration.png")
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print ("Exiting Main Program, spending ", elapsed_time, "seconds.")
