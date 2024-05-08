import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, [0,2]]
y = iris.target

y = np.where(y==0, 0, 1)  #convert into binary numbers

w = np.zeros(2)
b = 0
lr = 0.1
epoch = 50


def perceptron(x, w, b):
  z = np.dot(x, w)+b
  return np.where(z>= 0, 1, 0)

for epoch in range(epoch):
  for i in range(len(X)):
    x = X[i]
    target = y[i]
    output = perceptron(x, w, b)
    error = target - output
    w += lr * error *x
    b += lr * error

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() +0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 0].max() +0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = perceptron(np.c_[xx.ravel(), yy.ravel()], w, b)
z = Z.reshape(xx.shape)
plt.contourf(xx, yy, z, cmap=plt.cm.Paired)


plt.scatter(X[:, 0], X[:, 1], c =y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('petal length')
plt.title('Perceptron Decision Region')
plt.show()

