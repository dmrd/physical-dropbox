import scanner
import util
import sys

rotations = int(sys.argv[1])
prefix = str(sys.argv[2])

s = scanner.Scanner()

result = s.do_rotation(rotations)

util.save_images(result, prefix="img/"+prefix + "_{0}")
