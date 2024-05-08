import numpy as np

def mp_neuron(input, weight, threshold):
  weighted_sum = np.dot(input, weight)
  output = 1 if weighted_sum >= threshold else 0
  return output

def and_not(x1, x2):
  weight = [1, -1]
  threshold = 1
  input = np.array([x1,x2])
  output = mp_neuron(input, weight, threshold)
  return output

print(and_not(0,0))
print(and_not(0,1))
print(and_not(1,0))
print(and_not(1,1))
