import scanner
import util
import sys
import os

rotations = int(sys.argv[1])
prefix = str(sys.argv[2])

s = scanner.Scanner()
print("Scanner initialized")
#result = s.do_rotation(rotations)
result = s.continuous(rotations)
print("Images taken")

dir_name = os.path.join("img", prefix)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
util.save_images(result, prefix=os.path.join(dir_name, "%s_{0}" % prefix))
print("Images saved")
