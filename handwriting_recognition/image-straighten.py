import cv2
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


def de_skew(image):
    threshold = image
    edges = cv2.Canny(threshold, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 1000, 55)

    try:
        d1 = OrderedDict()
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                deg = np.rad2deg(theta)
                print(deg)
                if deg in d1:
                    d1[deg] += 1
                else:
                    d1[deg] = 1

        t1 = OrderedDict(sorted(d1.items(), key=lambda x: x[1], reverse=False))
        print(list(t1.keys())[0], 'Angle', threshold.shape)
        non_zero_pixel = cv2.findNonZero(threshold)
        center, wh, theta = cv2.minAreaRect(non_zero_pixel)
        angle = list(t1.keys())[0]
        if angle > 160:
            angle = 180 - angle
        if 20 < angle < 160:
            angle = 12
        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols = image.shape
        rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    except:
        rotated = image
        pass
    return rotated


def un_shear(image):
    return image


def pad_with(vector, pad_width, i_axis, kwargs):
    pad_value = kwargs.get('padder', 40)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


if __name__ == '__main__':
    image = cv2.imread('./sample_images/c.png', cv2.IMREAD_GRAYSCALE)
    threshold = cv2.threshold(image, 127, 255, 1)[1]
    threshold = np.pad(threshold, 100, pad_with, padder=0)

    plt.imshow(threshold)
    plt.show()
    de_skew(threshold)
    sheared_image = un_shear(threshold)

    ret, threshold = cv2.threshold(sheared_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    plt.imshow(threshold)
    plt.show()
    cv2.imwrite('./result/data/c.png', threshold)
