import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def get_xyonehot(data):
    x = data[:, 0]
    y = data[:, 1]
    onehot = data[:, 2::]
    i = np.argmax(onehot, axis=1)
    return x, y, onehot

def create_2d_grid(x_range, y_range, x_num, y_num):
    x_values = np.linspace(x_range[0], x_range[1], x_num)
    y_values = np.linspace(y_range[0], y_range[1], y_num)
    return np.meshgrid(x_values, y_values)


def func1(data):
    x, y, onehot = get_xyonehot(data)
    i = np.argmax(onehot, axis=1)
    
    freq = 2 * np.pi
    # freq = 0.5 * np.pi + 2 * np.pi * i * 2.
    temp = np.sin(freq * x / 20.0) + np.cos(freq * y / 10.0)
    return temp*0.4 + i*4

def func2(data):
    x, y, onehot = get_xyonehot(data)
    i = np.argmax(onehot, axis=1)
    temp = 4 - 0.1 * (x - 5 - i*4) **2 - 0.1 * (y - 5 - i*3) **2
    return temp

def rosenbrock(data, a=-1, b=-0.01):
    x, y, onehot = get_xyonehot(data)
    i = np.argmax(onehot, axis=1)
    return -0.01 * (a - x - i) ** 2 + b * (y - x** 2)** 2



if __name__ == "__main__":
    x_range = (0, 10)
    y_range = (0, 10)
    x_num = 20
    y_num = 20
    N = 50
    sig = 0.1

    x_values = np.linspace(x_range[0], x_range[1], x_num)
    y_values = np.linspace(y_range[0], y_range[1], y_num)

    x_grid, y_grid = create_2d_grid(x_range, y_range, x_num, y_num)
    data_matrix = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    gt_matrix = func(data_matrix[:, 0], data_matrix[:, 1])
    # gt_grid = func(x_grid, y_grid)
    gt_grid = griddata(
        (data_matrix[:, 0], data_matrix[:, 1]), gt_matrix,
        (x_grid, y_grid),
        method='cubic'
    )

    X = np.random.uniform(x_range[0], x_range[1], N)
    Y = np.random.uniform(y_range[0], y_range[1], N)
    ob_data = np.column_stack((X, Y))
    ob_matrix = func(X, Y)

    ob_grid = griddata(
        (ob_data[:, 0], ob_data[:, 1]), ob_matrix,
        (x_grid, y_grid),
        method='cubic'
    )

    plt.figure(figsize=(20, 12))
    plt.subplot(121)
    contour = plt.contourf(x_grid, y_grid, gt_grid, levels=200, cmap='viridis')
    plt.colorbar(contour)
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], color='red', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.subplot(122)
    plt.clim(-2, 2)
    contour = plt.contourf(x_grid, y_grid, ob_grid, levels=200, cmap='viridis')
    plt.colorbar(contour)
    plt.scatter(X, Y, color='red', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.clim(-2, 2)
    plt.title('Contour Plot from Random Points')

    plt.savefig('gpy.png')
