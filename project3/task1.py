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
    rows, columns = img.shape
    # Return Values
    best_center_pts = np.random.choice(np.arange(256, dtype=float), k, replace=False)
    best_img_labels = np.empty_like(img)
    min_error = rows*columns*256

    if k == 2:
        center_pairs = []
        possible_center_points = np.unique(img)
        for i in range(len(possible_center_points)):
            for j in range(i+1, len(possible_center_points)):
                if i == j:
                    continue
                center_pairs.append([possible_center_points[i], possible_center_points[j]])

        # run through all possible initializations
        for i, center_pts in enumerate(center_pairs):
            if i % int(len(center_pairs)/10) == 0:
                print('{}% complete'.format(round(i/len(center_pairs)*100)))

            new_center_pts = np.empty_like(center_pts, dtype=float)
            while True:
                # Calculate all the errors
                img_errors = np.empty((rows, columns, 2), dtype=float)
                img_errors[:, :, 0] = np.abs(img - center_pts[0])
                img_errors[:, :, 1] = np.abs(img - center_pts[1])

                # Calculate image labels
                img_label = np.argmin(img_errors, axis=2)

                # Calculate total error
                error = np.sum(np.min(img_errors, axis=2))

                if error < min_error:
                    best_center_pts = center_pts
                    min_error = error
                    # print(min_error)
                    best_img_labels = img_label

                # Calculate new center point (center of the labeled groups)
                if np.all(img_label != 0):
                    # If one of the centers has no pixels in it
                    new_center_pts[0] = center_pts[0]
                    new_center_pts[1] = np.around(np.mean(img[img_label == 1]))
                elif np.all(img_label != 1):
                    new_center_pts[1] = center_pts[1]
                    new_center_pts[0] = np.around(np.mean(img[img_label == 0]))
                else:
                    new_center_pts[0] = np.around(np.mean(img[img_label == 0]))
                    new_center_pts[1] = np.around(np.mean(img[img_label == 1]))

                # Iterate until the the centers do not change
                if np.all(new_center_pts == center_pts):
                    break
                else:
                    center_pts = new_center_pts

        # return center values, center labels of all pixels, the total sum of the distance of center to its pixel
        return [int(c) for c in best_center_pts], best_img_labels, int(min_error)

    else:
        iterations = 10
        for iteration in range(iterations):
            if iteration % int(iterations/10) == 0:
                print('{}% complete'.format(round(iteration / iterations * 100)))
            # Randomly choose k-centers
            # Initialize centers in range [0,255]
            center_pts = np.random.choice(np.arange(256, dtype=float), k, replace=False)
            while True:
                # Calculate all the errors
                img_errors = np.empty((rows, columns, k), dtype=float)
                for i in range(k):
                    img_errors[:, :, i] = np.abs(img - center_pts[i])

                # Calculate image labels
                img_label = np.argmin(img_errors, axis=2)

                # Calculate total error
                error = np.sum(np.min(img_errors, axis=2))

                if error < min_error:
                    best_center_pts = center_pts
                    min_error = error
                    # print(min_error)
                    best_img_labels = img_label

                # Calculate new center point (center of the labeled groups)
                new_center_pts = np.zeros_like(center_pts, dtype=float)
                for i in range(k):
                    if np.all(img_label != i):
                        # If one of the centers has no pixels in it
                        new_center_pts[i] = center_pts[i]
                        continue
                    new_center_pts[i] = np.around(np.mean(img[img_label == i]))

                # Iterate until the the centers do not change
                if np.all(new_center_pts == center_pts):
                    break
                else:
                    center_pts = new_center_pts

        # return center values, center labels of all pixels, the total sum of the distance of center to its pixel
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

    img_out = np.empty_like(labels, dtype=np.uint8)
    for i, center in enumerate(centers):
        img_out[labels == i] = center

    return img_out

     
if __name__ == "__main__":
    img = utils.read_image('lenna.png')
    k = 2
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
