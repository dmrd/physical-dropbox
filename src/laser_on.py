import scanner
import sys

s = scanner.Scanner()
if sys.argv[1] == "on":
    s.laser.on()
else:
    s.laser.off()
