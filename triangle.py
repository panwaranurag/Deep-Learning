
import math as mt
import sys

def triangle(side1, hypo):
    hypo = float(hypo)
    side1 = float(side1)
    side2 = 0.0000
    side2 = round(mt.sqrt(hypo**2 - side1**2),3)
    side2 = str(side2)
    sys.stdout.write(side2)

the_string = raw_input()
l = the_string.split()
if l[0] == "_":
    triangle(l[1], l[2])
elif l[1] == "_":
    triangle(l[0], l[2])
else:
    triangle(l[0], l[1])
