import scanner
import util
import sys, os

rotations = int(sys.argv[1])
prefix = str(sys.argv[2])

s = scanner.Scanner()

result = s.do_rotation(rotations)

dir_name = os.path.join("img", prefix)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
util.save_images(result, prefix=os.path.join(dir_name, "%s_{0}" % prefix))

