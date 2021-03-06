import sys
import os
import cv2
from processor import Processor
from scanner import run_scan

# Set directory to write files to
directory = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(directory, "../data"))


def scan_mode(mode, prefix, rotations):
    if mode == "left" or mode == "dual":
        run_scan(rotations=rotations, prefix=prefix, right=False)
    if mode == "right" or mode == "dual":
        run_scan(rotations=rotations, prefix=prefix, right=True)


def process_mode(mode, prefix, rotations, calibration_name):
    calibration_img = cv2.imread(calibration_name)
    proc = Processor(calibration_img)

    if mode == "left" or mode == "dual":
        proc.process_scan(rotations=rotations,
                          prefix=prefix,
                          calibration_name=calibration_name,
                          right=False)
        proc.save_ply("ply/" + prefix, left=True)
    if mode == "right" or mode == "dual":
        proc.process_scan(rotations=rotations,
                          prefix=prefix,
                          calibration_name=calibration_name,
                          right=True)
        proc.save_ply("ply/" + prefix, right=True)
    if mode == "dual":
        proc.save_ply("ply/" + prefix, dual=True)


if __name__ == "__main__":
    doc_string = ("python {0} [wait|scan|process|scan_and_process] prefix "
                  "num_rotations [left|right|dual] [calibration_image]"
                  ).format(sys.argv[0])
    if len(sys.argv) < 5:
        print(doc_string)
        exit()
    action = sys.argv[1]
    prefix = sys.argv[2]
    rotations = int(sys.argv[3])
    mode = sys.argv[4]

    calibration_name = 'calibration/calibration.jpg'

    assert mode in ['dual', 'left', 'right']
    if len(sys.argv) > 5:
        calibration_name = sys.argv[5]
    if action == "wait":
        pass
    elif action == "scan":
        scan_mode(mode=mode, prefix=prefix, rotations=rotations,)
    elif action == "process":
        process_mode(mode=mode, prefix=prefix, rotations=rotations,
                     calibration_name=calibration_name)
    elif action == "scan_and_process":
        scan_mode(mode=mode, prefix=prefix, rotations=rotations)
        process_mode(mode=mode, prefix=prefix, rotations=rotations,
                     calibration_name=calibration_name)
    else:
        print(doc_string)
