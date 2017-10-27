import urllib
import numpy as np

fname = "boston_houses.csv"
data = np.loadtxt(fname, delimiter=',', skiprows=1)  

rows, cols  = data.shape
depend_var_index = 0
MEDV = data[..., depend_var_index].copy()
independ_vars = data[..., 1:cols]
ones = np.ones_like(MEDV).reshape((rows, 1))
independ_vars = np.hstack((ones, independ_vars));

st1 = independ_vars.T.dot(independ_vars)
st2 = np.linalg.inv(st1)
st3 = st2.dot(independ_vars.T)
B = st3.dot(MEDV)
B_formatted = map(str, B)

print('B(MEDV): ')
print(' '.join(B_formatted))