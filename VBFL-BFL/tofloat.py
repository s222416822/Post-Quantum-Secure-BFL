

list = [
4.509425e-08,
1.449258e-07,
5.790360e-07,
3.019697e-06,
1.209062e-05,
3.928166e-05,
1.562762e-04,
6.310614e-04,
2.611285e-03,
9.823496e-03
]

def tofloat(value):
    return format(float(value), '.10f')

for i in list:
    print(float(tofloat(i)) * 1000000)