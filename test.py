import rebound
import numpy as np

a = [[1,2,3],[4,5,6]]
b = np.sum(a,axis = 0)
print(b)
print("version:", rebound.__version__)
print("path:", rebound.__file__)
