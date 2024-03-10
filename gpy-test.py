import numpy as np
import GPy


X = np.array([
    [1, 0, 0, 1, 1.0, 2.5],
    [0, 1, 0, 0, 2.0, 3.5],
    [0, 0, 1, 1, 3.0, 2.0],
    [1, 0, 0, 0, 4.0, 5.0],
    [0, 1, 0, 1, 5.0, 4.0]
])
Y = np.array([[1.0], [2.0], [3.0], [2.5], [3.5]])
kernel = GPy.kern.RBF(input_dim=2, active_dims=[4, 5]) + \
         GPy.kern.Linear(input_dim=4, active_dims=[0, 1, 2, 3])

model = GPy.models.GPRegression(X, Y, kernel)
model.optimize(messages=True)
X_new = np.array([[0, 0, 1, 0, 3.5, 4.5]])
mean, variance = model.predict(X_new)
print("Predicted mean:", mean)
print("Predicted variance:", variance)
