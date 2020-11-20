import cv2
import numpy as np
import glob
import os

crop_image = crop_image = lambda img, x0, y0, w, h: img[x0:x0+w, y0:y0+h]


def rotate_image(img, angle, crop):
    w, h = img.shape[:2]
    angle %= 360
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

    if crop:
        if angle %90 == 0:
            return img
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180
        hw_ratio = float(h) / float(w)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator

        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated


def rotate(img, angle):
    w, h = img.shape[:2]
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    if len(img.shape) == 3:
        img_rotated = cv2.warpAffine(img, M_rotation, (w, h), flags=cv2.INTER_CUBIC)
    if len(img.shape) == 2:
        img_rotated = cv2.warpAffine(img, M_rotation, (w, h), borderValue=25, flags=cv2.INTER_NEAREST)
        img_rotated[img_rotated > 125] = 255
    return img_rotated

img_list = glob.glob('./data/nyud/train/Images/images/*.png')
for i in img_list:
    img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
    lab = cv2.imread(i.replace('Images/images', 'GT/GT'), 0)
    print(img.shape, lab.shape)
    for j in range(1,16):
        path = './data/nyud/train/Images/images_'
        filename = i.split('/')[-1]
        angle = 22.5 * j
        if angle % 90 == 0: continue
        if not os.path.isdir(path + str(angle)):
            os.makedirs(path + str(angle))
        if not os.path.isdir(path.replace('Images/images', 'GT/GT') + str(angle)):
            os.makedirs(path.replace('Images/images', 'GT/GT') + str(angle))

        image_rotated = rotate(img, angle)
        lab_rotated = rotate(lab, angle)
        cv2.imwrite(os.path.join(path + str(angle), filename), image_rotated)
        cv2.imwrite(os.path.join(path.replace('Images/images', 'GT/GT') + str(angle), filename), lab_rotated)
