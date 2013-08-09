import sys
import os
from processor import process_scan
from scanner import run_scan

# Set directory to write files to
directory = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(directory, "../data"))


def scan_mode(mode, **kwargs):
    if mode == "left" or mode == "both":
        run_scan(right=False, **kwargs)
    if mode == "right" or mode == "both":
        run_scan(right=True, **kwargs)


def process_mode(mode, **kwargs):
    if mode == "left" or mode == "dual":
        process_scan(right=False, **kwargs)
    if mode == "right" or mode == "dual":
        process_scan(right=True, **kwargs)

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

    if len(sys.argv) > 4:
        calibration_name = sys.argv[4]
    if action == "wait":
        pass
    elif action == "scan":
        scan_mode(mode, rotations=rotations, prefix=prefix)
    elif action == "process":
        process_mode(mode, rotations=rotations, prefix=prefix,
                     calibration_name=calibration_name)
    elif action == "scan_and_process":
        scan_mode(mode, rotations=rotations, prefix=prefix)
        process_mode(mode, rotations=rotations, prefix=prefix,
                     calibration_name=calibration_name)
    else:
        print(doc_string)
