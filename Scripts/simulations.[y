import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from scipy import misc
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

def make_moving_square(frames, sigma = -1):
    square = np.ones((100,100))
    empty = np.zeros((200,100))
    square = np.concatenate((empty,square,empty), axis=0)
    video = []
    time = np.linspace(0,1,frames)
    for idx in time:
        left = np.zeros((500, int(np.ceil(idx*400))))
        right = np.zeros((500, int(np.floor((1-idx)*400))))
        img = np.concatenate((left, square, right), axis = 1)
        
        if sigma > 0:
            img += np.random.normal(0, sigma, (500, 500))
        
        video.append(img)
    return video

def make_rotating_square(frames,percent,theta, sigma = -1):
    square = np.ones((100,100))
    empty = np.zeros((200,100))
    square = np.concatenate((empty,square,empty), axis=0)
    idx = percent
    left = np.zeros((500, int(np.ceil(idx*400))))
    right = np.zeros((500, int(np.floor((1-idx)*400))))
    square = np.concatenate((left, square, right), axis = 1)
    
    video = []
    time = np.linspace(0,theta,frames)
    for angle in time:
        img = ndimage.interpolation.rotate(square, angle, reshape=False)
        
        if sigma > 0:
            img += np.random.normal(0, sigma, (500, 500))
        
        video.append(img)
    return video

def format_obervation_md_traj(t):
    size =  t.xyz.shape
    X = [f.reshape((1, -1))[0] for f in t.xyz]
    return X

    