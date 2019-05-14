import os
import matplotlib.pyplot as plt
import numpy as np

f = lambda x,y: max*(np.cos(x/500.0*np.pi+y*np.pi)+1)/(x/1500.0+1)

space = range(20001)
y = 0
cat = np.zeros((20001))

for x in space:
	if x>0 and f(x-1, y)<1e-7:
		y+=1
	cat[x] = f(x,y)

plt.plot(space, cat)
plt.show()
