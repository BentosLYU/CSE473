"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random


def distance_sq(P1, P2, p):
    """
    This function gets the square of the perpendicular distance from the line defined by P1 and P2 to the point p
    Wikipedia has the distance  formula here: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    """
    x1, y1 = P1['value']
    x2, y2 = P2['value']
    x0, y0 = p['value']

    return ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)**2 / ((y2-y1)**2 + (x2-x1)**2)


def error_inliers_outliers(points, error, tolerance):
    """
    This function calculates the total error and the inliers and outliers of the list of points
    tolerance is the cutoff for inliers and outliers
    """
    error_total = 0
    inliers = []
    outliers = []

    for i, pt in enumerate(points):
        if error[i] <= tolerance:
            inliers.append(pt)
            error_total += error[i]
        else:
            outliers.append(pt)

    if len(inliers)-2 == 0:
        mean = 9999999999999
    else:
        # subract 2 because the starting points are included
        mean = error_total/(len(inliers)-2)

    return mean, inliers, outliers


def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """

    if len(input_points) <= 2:
        print("only two or less points given")


    least_error = 9999999999999999
    least_error_inliers = []
    least_error_outliers = []
    num_permutations = len(input_points)*(len(input_points)-1)
    prev_pairs = []
    # starting_pts = ('', '')

    for i in range(k):
        # pick the starting points
        # need unique point names

        length = len(prev_pairs)
        if len(prev_pairs) >= num_permutations:
            break

        while True:
            starting_pts = random.sample(input_points, 2)
            if starting_pts not in prev_pairs:
                prev_pairs.append(starting_pts)
                break

        # print(starting_pts)
        # need to compute the line that defines the given points
        # compute the perpendicular distances to points
        dist_sq = [distance_sq(*starting_pts, pt) for pt in input_points]

        mean_error, inliers, outliers = error_inliers_outliers(input_points,dist_sq,t**2)

        if len(inliers) - 2 < d:
            continue

        if (mean_error < least_error or
                mean_error == least_error and len(inliers) > len(least_error_inliers)):
            least_error = mean_error
            least_error_inliers = inliers
            least_error_outliers = outliers

    if len(least_error_inliers)-2 < d:
        print("could not find good model")
        return [], [n['name'] for n in input_points]

    return [n['name'] for n in least_error_inliers], [n['name'] for n in least_error_outliers]


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]

    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    assert len(inlier_points_name) + len(outlier_points_name) == 8
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()
