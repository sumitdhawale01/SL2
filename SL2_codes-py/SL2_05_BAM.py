import numpy as np

x1 = np.array([1, 1, 1, -1])
y1 = np.array([1, -1])

x2 = np.array([-1, -1, 1, 1])
y2 = np.array([-1, 1])

W = np.outer(y1, x1) + np.outer(y2, x2)

def bam(x):
  y = np.dot(W, x)
  y = np.where(y >= 0, 1, -1)
  return y

x_test = np.array([1, 1, 1, -1])
y_test = bam(x_test)

print("Input x: ", x_test)
print("Output y: ", y_test)

W = np.outer(x1, y1) + np.outer(x2, y2)

y_test = np.array([1, -1])
x_test = bam(y_test)

print("Input x: ", y_test)
print("Output y: ", x_test)
