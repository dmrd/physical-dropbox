import cv2


def save_images(imgs, prefix="scan_{0}.jpg"):
    for i, img in enumerate(imgs):
        cv2.imwrite(prefix.format(i), img)
