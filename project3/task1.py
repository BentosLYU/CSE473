"""
K-Means Segmentation Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""


import utils
import numpy as np
import json
import time


def kmeans(img,k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """
    # Return Values
    # best_center_pts = np.random.choice(np.arange(256), 2, replace=False)
    # best_img_labels = np.empty_like(img)
    # min_error = 99999


    # Randomly choose k-centers
    # Initialize centers in range [0,255]
    rows, columns = img.shape
    x_s = np.random.choice(rows, k, replace=True)
    y_s = np.random.choice(columns, k, replace=True)
    # print(tuple(zip(x_s, y_s)))
    center_pts = img[x_s, y_s]
    tol = 0.001

    while True:
        # Calculate all the errors
        print('center points: {}'.format(center_pts))
        img_errors = np.empty((rows, columns, k))
        for i in range(rows):
            for j in range(columns):
                img_errors[i, j, :] = np.abs(img[i, j]-center_pts)

        # Calculate image labels
        img_label = np.argmin(img_errors, axis=2)

        # Calculate total error
        error = np.sum(np.min(img_errors, axis=2))
        print('error: {}'.format(error))
        # error = 0
        # for i in range(rows):
        #     for j in range(columns):
        #         error += img_errors[i, j, img_label[i,j]]

        # Calculate new center point (center of the labeled groups)
        new_center_pts = np.zeros_like(center_pts)
        counts = np.zeros_like(center_pts, dtype=int)
        for i in range(rows):
            for j in range(columns):
                new_center_pts[img_label[i, j]] += img[i, j]
                counts[img_label[i, j]] += 1

        # Need to check for divide by zero
        print(counts)
        print(new_center_pts)
        if np.any(counts == 0):
            break
        new_center_pts = new_center_pts/counts

        # Iterate until the the centers do not change
        if np.all(new_center_pts-center_pts < tol):
            # best_center_pts = new_center_pts
            # min_error = error
            # best_img_labels = img_label
            print(new_center_pts-center_pts)
            break
        else:
            center_pts = new_center_pts

    # Iterate on different initializations and output the best center

    # return clustering center values, clustering labels of all pixels, the total sum of the centers to each pixel

    return new_center_pts, img_label, error


def visualize(centers,labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    rows, columns = labels.shape
    im = np.empty_like(labels, dtype=float)
    for i in range(rows):
        for j in range(columns):
            im[i, j] = centers[labels[i, j]]

    print(centers)
    return im

     
if __name__ == "__main__":
    img = utils.read_image('lenna.png')
    img = img/255
    k = 2
    print("maximum value: {}".format(np.max(img)))
    start_time = time.time()
    centers, labels, sumdistance = kmeans(img,k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
