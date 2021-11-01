import numpy as np
long = [254, 255, 17, 41, 47, 148, 13, 254, 254, 453]
short = [80, 80, 80, 80, 63, 80, 80, 80, 80, 80]

long_array = np.array(long)
print(long_array.mean())
print(long_array.std())

short_array = np.array(short)
print(short_array.mean())
print(short_array.std())
