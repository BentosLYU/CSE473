"""
Denoise Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to denoise image using median filter.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are suggested to use utils.zero_pad.
"""


import utils
import numpy as np
import json

def median_filter(img):
    """
    Implement median filter on the given image.
    Steps:
    (1) Pad the image with zero to ensure that the output is of the same size as the input image.
    (2) Calculate the filtered image.
    Arg: Input image. 
    Return: Filtered image.
    """
    filter_size = 3
    pad = int(filter_size/2)
    out_img = np.empty_like(img)
    # pad zeros for 3x3 filter
    padded_img = utils.zero_pad(img, pad, pad)

    # use the square median filter
    rows, columns = padded_img.shape
    for i in range(pad, rows - pad):
        for j in range(pad, columns - pad):
            out_img[i-pad, j-pad] = np.median(padded_img[i-pad:i+pad+1, j-pad:j+pad+1])
            # print(np.sort(padded_img[i-pad:i+pad+1, j-pad:j+pad]))
    return out_img

def mse(img1, img2):
    """
    Calculate mean square error of two images.
    Arg: Two images to be compared.
    Return: Mean square error.
    """
    # we are finding the mean of the square error
    square_error = np.square(img1 - img2)
    return np.mean(square_error)
    

if __name__ == "__main__":
    img = utils.read_image('lenna-noise.png')
    gt = utils.read_image('lenna-denoise.png')

    result = median_filter(img)
    error = mse(gt, result)

    with open('results/task2.json', "w") as file:
        json.dump(error, file)
    utils.write_image(result,'results/task2_result.jpg')


