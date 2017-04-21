"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
from keras.applications.imagenet_utils import preprocess_input

from keras import backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def standardize(x,
             rescale=None, 
            featurewise_center=None, 
            featurewise_std_normalization = None, 
            preprocessing_function= None,
            zca_whitening = None,
            channel_axis = 3):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """

        if preprocessing_function:
            x = preprocessing_function(x)
        if rescale:
            x *= rescale

        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = channel_axis - 1
        if samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if featurewise_center:
            if mean is not None:
                x -= mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if featurewise_std_normalization:
            if std is not None:
                x /= (std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if zca_whitening:
            if principal_components is not None:
                flatx = np.reshape(x, (x.size))
                whitex = np.dot(flatx, principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        model
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        #print("[INFO] channel:{}".format(data_format))
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files]
            #if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def getPics(floder,target_size=(224,224)):
    '''
    @clz
    get pics tensor for predict
    '''

    # get pic path
    picPaths = list_pictures(floder)
    #print(picPaths)

    # read
    picTensor = np.zeros((len(picPaths),) + target_size + (3,), dtype=K.floatx())
    for ind,path in enumerate(picPaths):
        img = load_img(path=path, target_size = target_size)
        picTensor[ind] = img_to_array(img)

    # process by keras preprocession   RGB->BGR   -=mean !!!
    picTensor2 = picTensor.copy()
    picTensor_change =  preprocess_input(picTensor2)

    return picPaths,picTensor_change


if __name__ == "__main__":
    import sys 
    getPics(sys.argv[1])
     
