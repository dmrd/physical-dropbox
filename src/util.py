import os
import cv2


def save_images(imgs, prefix, dir_name="img"):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    prefix = os.path.join(dir_name, "%s_{0}" % prefix)
    for i, img in enumerate(imgs):
        i = str(i).zfill(3)
        cv2.imwrite(prefix.format(i) + ".jpg", img)
