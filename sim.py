import utils
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import GPy


def get_xyonehot(data):
    x = data[:, 0]
    y = data[:, 1]
    onehot = data[:, 2::]
    i = np.argmax(onehot, axis=1)
    return x, y, onehot

def func(data):
    x, y, onehot = get_xyonehot(data)
    i = np.argmax(onehot, axis=1)
    
    freq = 2 * np.pi
    # freq = 0.5 * np.pi + 2 * np.pi * i * 2.
    temp = np.sin(freq * x / 20.0) + np.cos(freq * y / 10.0)
    return temp*0.4 + i*4

def rosenbrock(data, a=1, b=1):
    x, y, onehot = get_xyonehot(data)
    i = np.argmax(onehot, axis=1)
    return (a - x) ** 2 + b * (y - x** 2)** 2

def func_noise(data, f):
    temp = f(data)
    if isinstance(temp, np.ndarray):
        temp = temp + sig * np.random.randn(temp.shape[0])
    else:
        temp = temp + sig * np.random.randn()
    return temp


def onehot_random(N, dim):

    one_hot_vectors = np.zeros((N, dim))
    indices = np.random.randint(0, dim, N)
    one_hot_vectors[np.arange(N), indices] = 1

    return one_hot_vectors



if  __name__ == '__main__':

    x_range = (0, 10)
    y_range = (0, 10)
    x_num = 20
    y_num = 20
    N = 20
    # N = 100
    sig = 0.1

    x_values = np.linspace(x_range[0], x_range[1], x_num)
    y_values = np.linspace(y_range[0], y_range[1], y_num)
    x_grid, y_grid = utils.create_2d_grid(x_range, y_range, x_num, y_num)
    N_grid = x_grid.ravel().shape[0]
    onehot_grid0 = np.zeros((N_grid, 2))
    onehot_grid1 = np.zeros((N_grid, 2))
    onehot_grid0[:, 0] = 1
    onehot_grid1[:, 1] = 1
    data_matrix0 = np.column_stack(
        (
            x_grid.ravel()[:, np.newaxis],
            y_grid.ravel()[:, np.newaxis],
            onehot_grid0
         )
    )
    data_matrix1 = np.column_stack(
        (
            x_grid.ravel()[:, np.newaxis],
            y_grid.ravel()[:, np.newaxis],
            onehot_grid1
         )
    )
    gt_matrix0 = func(data_matrix0)
    gt_matrix1 = func(data_matrix1)
    gt_grid0 = griddata(
        (data_matrix0[:, 0], data_matrix0[:, 1]), gt_matrix0,
        (x_grid, y_grid),
        method='cubic'
    )
    gt_grid1 = griddata(
        (data_matrix1[:, 0], data_matrix1[:, 1]), gt_matrix1,
        (x_grid, y_grid),
        method='cubic'
    )

    X0 = np.random.uniform(x_range[0], x_range[1], N//2)
    Y0 = np.random.uniform(y_range[0], y_range[1], N//2)
    onehot0 = onehot_random(N//2, 2)
    ob_data0 = np.column_stack((X0[:, np.newaxis], Y0[:, np.newaxis], onehot0))
    X1 = np.random.uniform(x_range[0], x_range[1], N//2)
    Y1 = np.random.uniform(y_range[0], y_range[1], N//2)
    onehot1 = onehot_random(N//2, 2)
    ob_data1 = np.column_stack((X1[:, np.newaxis], Y1[:, np.newaxis], onehot1))
    
    ob_data = np.vstack((ob_data0, ob_data1))
    ob_target = func(ob_data)

    kernel = GPy.kern.Linear(input_dim=2, active_dims=[2, 3]) + \
             GPy.kern.RBF(input_dim=2, active_dims=[0, 1])


    model = GPy.models.GPRegression(ob_data, ob_target[:, np.newaxis], kernel)
    model.optimize(messages=True)


    mean0, variance0 = model.predict(data_matrix0)
    mean1, variance1 = model.predict(data_matrix1)

    mean_grid0 = griddata(
        (data_matrix0[:, 0], data_matrix0[:, 1]), mean0,
        (x_grid, y_grid),
        method='cubic'
    )[:, :, 0]
    mean_grid1 = griddata(
        (data_matrix1[:, 0], data_matrix1[:, 1]), mean1,
        (x_grid, y_grid),
        method='cubic'
    )[:, :, 0]

    var_grid0 = griddata(
        (data_matrix0[:, 0], data_matrix0[:, 1]), variance0,
        (x_grid, y_grid),
        method='cubic'
    )[:, :, 0]
    var_grid1 = griddata(
        (data_matrix1[:, 0], data_matrix1[:, 1]), variance1,
        (x_grid, y_grid),
        method='cubic'
    )[:, :, 0]



    plt.figure(figsize=(20, 12))
    plt.subplot(321)
    contour = plt.contourf(x_grid, y_grid, gt_grid0, levels=50, cmap='viridis')
    # plt.colorbar(contour)
    plt.scatter(data_matrix0[:, 0], data_matrix0[:, 1], color='red', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.clim(-5, 5)
    plt.colorbar()
    plt.subplot(322)
    contour = plt.contourf(x_grid, y_grid, gt_grid1, levels=40, cmap='viridis')
    # plt.colorbar(contour)
    
    plt.scatter(data_matrix1[:, 0], data_matrix1[:, 1], color='red', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.clim(-5, 5)
    plt.colorbar()
    plt.subplot(323)
    contour = plt.contourf(x_grid, y_grid, mean_grid0, levels=40, cmap='viridis')
    # plt.colorbar(contour)
    plt.scatter(X0, Y0, color='red', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.clim(-5, 5)
    plt.colorbar()
    plt.subplot(324)
    contour = plt.contourf(x_grid, y_grid, mean_grid1, levels=40, cmap='viridis')
    # plt.colorbar(contour)
    plt.scatter(X1, Y1, color='red', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.clim(-5, 5)
    plt.colorbar()
    plt.subplot(325)
    plt.contourf(x_grid, y_grid, var_grid0, levels=40, cmap='viridis')
    # plt.clim(0, 20)
    plt.colorbar()
    plt.subplot(326)
    plt.contourf(x_grid, y_grid, var_grid1, levels=40, cmap='viridis')
    # plt.clim(0, 20)
    plt.colorbar()

    plt.savefig('gpy.png')