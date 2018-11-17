import matplotlib.pyplot as plt
import numpy as np
import math

def get_y(x):
    a = 1.0 / math.log(x+100, 100)

x = np.array(range(20*5000))

y = 1.0 / (np.log(x+100)/np.log(100))

y = 1.0 / 2**x

y = 0.99**x

plt.figure()
plt.plot(x,y)
plt.show()