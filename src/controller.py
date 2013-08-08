import sys
from processor import process_scan
from scanner import run_scan


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(("python {0} [wait|scan|process|scan_and_process] prefix "
              "num_rotations [calibration_image]").format(sys.argv[0]))
        exit()
    action = sys.argv[1]
    prefix = str(sys.argv[2])
    rotations = int(sys.argv[3])

    calibration_name = 'calibration/calibration.jpg'
    if len(sys.argv) > 4:
        calibration_name = sys.argv[4]

    if action == "wait":
        pass
    elif action == "scan":
        run_scan(rotations=rotations, prefix=prefix)
    elif action == "process":
        process_scan(rotations=rotations, prefix=prefix,
                     calibration_name=calibration_name)
    elif action == "scan_and_process":
        run_scan(rotations=rotations, prefix=prefix)
        process_scan(rotations=rotations, prefix=prefix,
                     calibration_name=calibration_name)
