import cv2


def save_images(imgs, prefix="scan_{0}"):
    for i, img in enumerate(imgs):
        i = str(i).zfill(3)
        cv2.imwrite(prefix.format(i) + ".jpg", img)
