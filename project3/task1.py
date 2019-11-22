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


def kmeans(img, k):
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
    best_center_pts = np.random.choice(np.arange(256, dtype=np.uint8), 2, replace=False)
    best_img_labels = np.empty_like(img)
    min_error = 9999999
    iterations = 10
    for iteration in range(iterations):
        # Randomly choose k-centers
        # Initialize centers in range [0,255]
        img = img.astype(np.uint8)
        rows, columns = img.shape
        # center_pts = np.around(np.random.choice(256, k, replace=False))
        center_pts = [31, 131]
        print(center_pts)
        tol = 0.001

        while True:
            # Calculate all the errors
            img_errors = np.empty((rows, columns, k), dtype=np.uint8)
            for i in range(k):
                img_errors[:, :, i] = np.abs(img - center_pts[i])

            # Calculate image labels
            img_label = np.argmin(img_errors, axis=2)

            # Calculate total error
            error = np.sum(np.min(img_errors, axis=2))

            # Calculate new center point (center of the labeled groups)
            sums = np.zeros_like(center_pts, dtype=int)
            counts = np.zeros_like(center_pts, dtype=int)
            for i in range(rows):
                for j in range(columns):
                    sums[img_label[i, j]] += img[i, j]
                    counts[img_label[i, j]] += 1

            # Need to check for divide by zero
            if np.any(counts == 0):
                print('counts is 0')
                break

            new_center_pts = np.around(sums/counts)
            print(center_pts)
            # print(new_center_pts)

            # Iterate until the the centers do not change
            print('error: {}'.format(error))
            if error < min_error:
                best_center_pts = center_pts
                min_error = error
                best_img_labels = img_label

            if np.all(new_center_pts-center_pts < tol):
                break
            else:
                center_pts = new_center_pts

        # Iterate on different initializations and output the best center

    # return clustering center values, clustering labels of all pixels, the total sum of the centers to each pixel
    # new_center_pts = np.around(new_center_pts)
    return [int(c) for c in best_center_pts], best_img_labels, int(min_error)


def visualize(centers,labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    centers = np.array(centers, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    rows, columns = labels.shape
    im = np.empty_like(labels, dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            im[i, j] = centers[labels[i, j]]

    print('centers: {}'.format(centers))
    return im

     
if __name__ == "__main__":
    img = utils.read_image('monroe.PNG')
    k = 2
    print("maximum value: {}".format(np.max(img)))
    start_time = time.time()
    centers, labels, sumdistance = kmeans(img, k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
