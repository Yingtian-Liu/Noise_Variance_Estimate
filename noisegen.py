# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:57:57 2023

@author: yingtian
"""

import numpy as np
def noisegen(X, SNR, seed):
    """
    Add white Gaussian noise to a signal.

    Parameters:
    X (ndarray): The signal to which noise is to be added.
    SNR (float): The desired signal-to-noise ratio in dB.
    seed (int): The seed for the random number generator.

    Returns:
    Y (ndarray): The signal with added noise.
    NOISE (ndarray): The generated noise.
    """
    np.random.seed(seed)
    NOISE = np.random.randn(*X.shape)
    NOISE = NOISE - NOISE.mean()
    signal_power = np.mean(X**2)
    noise_variance = signal_power / (10**(SNR/10))
    NOISE = np.sqrt(noise_variance) * NOISE
    Y = X + NOISE
    return Y, NOISE 
