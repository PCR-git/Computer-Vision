import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    pass
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    
    out = 0.5*np.array(image)**2
    pass
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    import skimage.color as skico
    out = skico.rgb2gray(image)
    pass
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image.copy()
    if channel == 'R':
        idx = 0
    elif channel == 'G':
        idx = 1
    elif channel == 'B':
        idx = 2
    out[:,:,idx] = 0
    
    pass
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    
    out = None

    ### YOUR CODE HERE
    lab = color.rgb2lab(image)/255.0
    
    out = lab.copy()
    if channel == 'L':
        out[:,:,1] = 0
        out[:,:,2] = 0
    elif channel == 'A':
        out[:,:,0] = 0
        out[:,:,2] = 0
    elif channel == 'B':
        out[:,:,0] = 0
        out[:,:,1] = 0
    
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    
    out = None

    ### YOUR CODE HERE
    hsv = color.rgb2hsv(image)
    
    out = hsv.copy()
    if channel == 'H':
        out[:,:,1] = 0
        out[:,:,2] = 0
    elif channel == 'S':
        out[:,:,0] = 0
        out[:,:,2] = 0
    elif channel == 'V':
        out[:,:,0] = 0
        out[:,:,1] = 0
    
    pass
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    im1 = rgb_exclusion(image1, channel1)
    im2 = rgb_exclusion(image2, channel2)
    L = np.shape(im1)[0]
    imL = im1[:,0:np.int(L/2),:]
    imR = im2[:,np.int(L/2):L,:]
    out = np.concatenate((imL,imR),axis=1)
    pass
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    im1 = rgb_exclusion(image, channel='R')
    im2 = dim_image(image)
    im3 = np.array(image)**0.5
    im4 = rgb_exclusion(image, channel='R')

    L = np.shape(im1)[0]
    Lh = np.int(L/2)
    imTL = im1[0:Lh,0:Lh,:]
    imTR = im2[0:Lh,Lh:L,:]
    imBL = im3[Lh:L,0:Lh,:]
    imBR = im4[Lh:L,Lh:L,:]

    T = np.concatenate((imTL,imTR),axis=1)
    B = np.concatenate((imBL,imBR),axis=1)
    out = np.concatenate((T,B),axis=0)
    pass
    ### END YOUR CODE

    return out
