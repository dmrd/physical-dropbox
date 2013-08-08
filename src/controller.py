import scanner
import util
import sys

rotations = int(sys.argv[1])
prefix = str(sys.argv[2])

s = scanner.Scanner()
print("Scanner initialized")
#result = s.do_rotation(rotations)
result = s.continuous(rotations)
print("Images taken")
util.save_images(result, prefix="img/"+prefix + "_{0}")
print("Images saved")
