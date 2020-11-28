"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ## YOUR CODE HERE
    hk = int((Hk-1)/2)
    wk = int((Wk-1)/2)
    for n in range(1,Hi-hk):
        for m in range(1,Wi-wk):
            for N in range(-hk,hk+1):
                for M in range(-wk,wk+1):
                    out[n,m] += image[n+N,m+M]*kernel[hk-N,wk-M]
    pass
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    sz = np.shape(image)
    out = np.zeros((2*pad_height+sz[0],2*pad_width+sz[1]))
    out[pad_height:pad_height+sz[0],pad_width:pad_width+sz[1]] = image
    
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
#     hk = int((Hk-1)/2)
#     wk = int((Wk-1)/2)
# #     padded_img = zero_pad(image, hk, wk)
#     kf = np.flip(np.flip(kernel,axis=0),axis=1)

#     for i in range(hk,Hi-hk):
#         for j in range(wk,Wi-wk):
#             out[i,j] = np.sum(image[i-hk:i+hk+1, j-wk:j+wk+1]*kf)

    hk = int((Hk-1)/2)
    wk = int((Wk-1)/2)
    padded_img = zero_pad(image, hk, wk)
    kf = np.flip(np.flip(kernel,axis=0),axis=1)

    for i in range(hk,Hi-hk):
        for j in range(wk,Wi-wk):
            out[i,j] = np.sum(padded_img[i:i+Hk, j:j+Wk]*kf)

    pass
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # Flip, since this is an autocorrelation
    g = np.flip(np.flip(g, axis=0), axis=1)
    out = conv_fast(f, g)
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(np.flip(g - np.mean(g), axis=0), axis=1)
    out = conv_fast(f, g)
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    hg = int((Hg-1)/2)
    wg = int((Wg-1)/2)
    padded_f = zero_pad(f, hg, wg)

    for i in range(hg,Hf-hg):
        for j in range(wg,Wf-wg):
            fmn = padded_f[i:i+Hg, j:j+Wg]
            fmn = (fmn - np.mean(fmn))/np.std(fmn)
            g = (g - np.mean(g))/np.std(g)
            out[i,j] = np.sum(fmn*g)
    
    pass
    ### END YOUR CODE

    return out
