# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 22:22:54 2023

@author: yingtian
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from noisegen import  noisegen
import math
plt.close('all')


seismic_data = np.load('marmousi2.npy', allow_pickle=True)
SNR = 5 # add noise
seed = 42 
noise=np.zeros_like(seismic_data)
for m in range(seismic_data.shape[1]):
                seismic_data[:,m,:], noise[:,m,:] = noisegen(seismic_data[:,m,:], SNR, seed)

fig = plt.figure()
plt.imshow(noise.T[:,0,:], cmap='gray')
plt.colorbar()
plt.show()
plt.title('noise')

fig2 = plt.figure()
plt.imshow(seismic_data[:,0,:].T,aspect="auto", cmap='RdGy')
plt.show()
plt.colorbar()    

def estimate(data):
    # patch segmentation
    seg = nn.Unfold(kernel_size=(9, 9), dilation=1, padding=0, stride=3)
    patches = seg(data)
    patches = patches.squeeze(0)
    # principal component analysis
    out = torch.cov(patches)
    L_complex, _ = torch.linalg.eig(out)
    sv = torch.real(L_complex)
    sv, _ = torch.sort(sv, descending=True, dim=0)
    # noise estimate
    for i in range(int(81)):
        t = torch.sum(sv[i:81]) / (81 - i)
        f = int((80 + i) / 2)
        f1 = f - 1
        f2 = min(f + 1, 80)
        if (t <= sv[f1]) and (t >= sv[f2]):
            return t.numpy()  
    return 0

print('SNR:', SNR)
noise_var = np.var(noise[:,0,:])
seismic_noise_var = np.var(seismic_data[:,0,:])
print('Variance of added noise:', noise_var)
print('Estimated noise variance:', estimate(torch.tensor(seismic_data[:,0,:].squeeze()).unsqueeze(0).unsqueeze(0)))
print('Estimated SNR:', 10 * math.log10((seismic_noise_var - noise_var) / noise_var))