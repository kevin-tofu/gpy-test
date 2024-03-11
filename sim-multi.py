import utils
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import GPy




def func_noise(f, data):
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
    kern = GPy.kern.RBF(2) + GPy.kern.White(2)
    kernel = GPy.util.multioutput.ICM(
        input_dim=2,
        num_outputs=2,
        kernel=kern
    )
    # func = utils.func1
    func = utils.func2
    # func = utils.rosenbrock

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

    X = np.random.uniform(x_range[0], x_range[1], N)
    Y = np.random.uniform(y_range[0], y_range[1], N)
    X0 = X
    X1 = X
    Y0 = Y
    Y1 = Y
    onehot0 = np.zeros((N, 2))
    onehot1 = np.zeros((N, 2))
    onehot0[:, 0] = 1
    onehot1[:, 1] = 1
    ob_data0 = np.column_stack((X[:, np.newaxis], Y[:, np.newaxis], onehot0))
    ob_data1 = np.column_stack((X[:, np.newaxis], Y[:, np.newaxis], onehot1))
    # ob_data = np.vstack((ob_data0, ob_data1))
    ob_target0 = func(ob_data0)[:, np.newaxis]
    ob_target1 = func(ob_data1)[:, np.newaxis]

    model = GPy.models.GPCoregionalizedRegression(
        X_list=[ob_data0[:, 0:2], ob_data1[:, 0:2]],
        Y_list=[ob_target0, ob_target1],
        kernel=kernel
    )
    model.optimize(
        'bfgs',
        max_iters=500,
        messages=True
    )
    print(model)

    metadata_Y0 = dict(
        output_index=np.argmax(data_matrix0[:, 2::], axis=1).astype(np.int64)
    )
    metadata_Y1 = dict(
        output_index=np.argmax(data_matrix1[:, 2::], axis=1).astype(np.int64)
    )
    metadata_X0 = np.hstack(
        (
            x_grid.ravel()[:, np.newaxis],
            y_grid.ravel()[:, np.newaxis],
            np.zeros((x_num*y_num, 1))
        )
    )
    metadata_X1 = np.hstack(
        (
            x_grid.ravel()[:, np.newaxis],
            y_grid.ravel()[:, np.newaxis],
            np.ones((x_num*y_num, 1))
        )
    )
    mean0, variance0 = model.predict(
        metadata_X0,
        Y_metadata=metadata_Y0
    )
    mean1, variance1 = model.predict(
        metadata_X1,
        Y_metadata=metadata_Y1
    )
    
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

    plt.savefig('gpy-multi.png')