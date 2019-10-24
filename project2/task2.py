"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random



def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    print_results = False

    # Get grayscale images
    left_gray = np.float32(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY))
    right_gray = np.float32(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))



    # 1: Use Kaze to find the features
    kaze = cv2.KAZE_create(threshold=20)
    right_keypoints, right_desc = kaze.detectAndCompute(right_gray, None)
    left_keypoints, left_desc = kaze.detectAndCompute(left_gray, None)

    if print_results:
        # draw the keypoints on the images
        left_kps = cv2.drawKeypoints(left_img, left_keypoints, np.empty(left_img.shape),
                                     flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        right_kps = cv2.drawKeypoints(right_img, right_keypoints, np.empty(right_img.shape),
                                      flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv2.imshow('left keypoints', left_kps)
        cv2.imshow('right keypoints', right_kps)

    # 2: Match key points (brute force matching could be improved upon)
    matches = []
    nn_match_ratio = 0.5  # matching ratio

    matcher = cv2.BFMatcher()
    nn_matches = matcher.knnMatch(left_desc, right_desc, 2)
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matches.append(m)


    if print_results:
        # draw all matches
        res = np.empty((left_img.shape[0]+right_img.shape[0], left_img.shape[1] + right_img.shape[1], 3), dtype=np.uint8)
        res = cv2.drawMatches(left_img, left_keypoints, right_img, right_keypoints, matches, res,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('matched keypoints', res)

    # 3: Compute the homeography matrix using RANSAC
    left_pts = np.float32([left_keypoints[m.queryIdx].pt for m in matches])
    right_pts = np.float32([right_keypoints[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(right_pts, left_pts, cv2.RANSAC, 5.0)
    # Gives a transformation from the right coords to the left coords

    # find bounds of warped image
    # this is to make sure that the right image is not sent to negative coordinates in the left image
    xhigh = right_img.shape[1]
    yhigh = right_img.shape[0]
    corners = np.array([[0, xhigh,      0,  xhigh],
                        [0,     0,  yhigh,  yhigh],
                        [1,     1,      1,      1]])
    new_corners = np.matmul(H, corners)
    new_corners /=new_corners[2, :]
    new_yhigh = int(max(new_corners[1, :]))
    offset = 0
    if min(new_corners[1, :]) < 0:
        offset = int(-min(new_corners[1, :]))
        new_yhigh = new_yhigh + offset
        print(offset)

    T = np.array([[1, 0, 0],
                  [0, 1, offset],
                  [0, 0, 1]])
    H = np.matmul(T,H)

    # 4: Use the homographic matrix to stitch the images together
    dims = (right_img.shape[1]+left_img.shape[1], max(new_yhigh, left_img.shape[0]))
    result = cv2.warpPerspective(right_img, H, dims) # warp the right image into left image coords

    if print_results:
       cv2.imshow('transformed image', result)

    result[offset:offset+left_img.shape[0], 0:left_img.shape[1]] = left_img


    if print_results:
        cv2.imshow('result', result)
        # Clean up the images
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    return result

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)


