from dash_canvas.utils import parse_jsonstring
import numpy as np
from numpy import asarray
import math
from scipy import ndimage


def rebin(arr, new_shape):
    """
    Rebin 2D array arr to shape new_shape by averaging.

    inputs:
    arr: numpy array to reduce at the size new_shape
    new_shape: tuple of the wanted size as (width, height)

    output:
    numpy array of the reduced arr

    from https://scipython.com/blog/binning-a-2d-array-in-numpy/
    """

    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def preprocess(string, canvas_width, scale, input_size, bord_size):
    '''
    Pre-processing: size reduction and centering

    inputs:
    string: drawing catch from the Canvas
    input_size: int. size of the image for prediction usable by the model
    bord_size: number of pixels around the resized draing

    Process:
        - Parse the json to make an image
        - Crop the bounding box containing only the drawing 
        (https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_find_object.html)
        - Rebin the drawing (see def rebin)

    output:
    processed_input: numpy array representing the drawn image scaled to the input_size

    Note: The center_of_mass may be an additional possibility(https://stackoverflow.com/questions/29356825/python-calculate-center-of-mass)
    '''

    # Fetch the data form the Canvas
    mask = parse_jsonstring(string)
    np_data = asarray((1 * mask).astype(np.uint8))
    # Discard the pixel outside of the canvas
    np_data = np_data[:canvas_width, :canvas_width]

    if len(scale)>0:
        # Scaled mode
        # Get the bounding box
        slice_x, slice_y = ndimage.find_objects(np_data.astype(int))[0]
        cropped_input = np_data[slice_x, slice_y]

        # Insert the cropped_input in the middle of a square of size [integer*mini_size]
        mini_size = input_size-2*bord_size  # Size of the drawing you want inside the box
        assert mini_size < input_size, 'the drawn zone must be smaller than the input_size'

        magnification = math.ceil(
            max(cropped_input.shape)/mini_size)  # math.ceil: round up
        adjusted_box_size = magnification*mini_size
        centered_input = np.zeros((adjusted_box_size, adjusted_box_size))
        # Insert the cropped_input
        x_sq=(adjusted_box_size - cropped_input.shape[0])//2
        y_sq=(adjusted_box_size - cropped_input.shape[1])//2

        centered_input[x_sq:x_sq+cropped_input.shape[0],
                    y_sq: y_sq+cropped_input.shape[1]] = cropped_input

        # Rebin to a smaller size and insert in a square of 28x28
        resized_input = rebin(centered_input, (mini_size, mini_size))
        processed_input = np.zeros((input_size, input_size))

        x_cropped_input = (input_size - mini_size)//2
        processed_input[x_cropped_input:x_cropped_input+mini_size,
                        x_cropped_input: x_cropped_input + mini_size] = resized_input
    else:
         # Not the scaled mode
        processed_input = rebin(np_data, (input_size, input_size))

    return processed_input
