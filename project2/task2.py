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
    # 1 Find Key point using harris detector
    lgray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    lgray = np.float32(lgray)
    # dst = cv2.cornerHarris(lgray, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # left_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    # cv2.imshow('left', left_img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    rgray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    rgray = np.float32(rgray)
    # dst = cv2.cornerHarris(rgray, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # right_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    # cv2.imshow('right', right_img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    # 2 use SIFT to extracture featrures
    # use cv2.xfeatures2d.SIFT_create()
    kaze = cv2.KAZE_create(threshold=20)
    lkp, ldesc = kaze.detectAndCompute(lgray, None)
    left_img = cv2.drawKeypoints(left_img, lkp, left_img, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('left', left_img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    length = len(lkp)
    print(length)

    # kaze = cv2.KAZE_create(threshold=40)
    rkp, rdesc = kaze.detectAndCompute(rgray, None)
    right_img = cv2.drawKeypoints(right_img, rkp, right_img, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('right', right_img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    length = len(rkp)
    print(length)


    # 3 match key points
    matcher = cv2.BFMatcher()
    nn_matches = matcher.knnMatch(ldesc, rdesc, 2)

    # matched1 = []
    # matched2 = []
    matches = []
    nn_match_ratio = 0.5  # Nearest neighbor matching ratio
    print(len(nn_matches))

    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matches.append(m)
            # matched2.append(rkp[m.trainIdx])

    print(len(matches))

    # DMatch vectors
    # matches = []
    # for i in range(len(matched1)):
    #     matches.append(cv2.DMatch(i, i), 0)

    res = np.empty((max(left_img.shape[0], right_img.shape[0]), left_img.shape[1] + right_img.shape[1], 3), dtype=np.uint8)
    out2 = cv2.drawMatches(left_img, lkp, right_img, rkp, matches, res,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('out2', out2)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    left_pts = np.float32([lkp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    right_pts = np.float32([rkp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # 4 compute the homographymatrix using RANSAC
    M, mask = cv2.findHomography(right_pts, left_pts, cv2.RANSAC, 5.0)
    M1, mask1 = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, 5.0)
    print(M)
    print(M1)

    # 5 us the homographic matrix to stitch the images together
    # i am giong to use the left image as the base image and warp the right image so that the
    res = np.zeros((2*left_img.shape[0],2*left_img.shape[0]), dtype=np.uint8)
    x = res.shape
    print(right_img.shape)
    left = cv2.warpPerspective(left_img, M, res.shape)
    right = cv2.warpPerspective(right_img, M, res.shape)


    cv2.imshow('left', left)
    cv2.imshow('right', right)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)


