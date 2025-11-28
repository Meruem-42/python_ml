import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as mlr

if __name__ == "__main__" :
    data = np.genfromtxt("are_blue_pills_magics.csv", delimiter=",", skip_header=1)
    x = data[:,1]
    y = data[:,2]
    lr1 = mlr(np.array([0, 0]), alpha=0.01, max_iter=10000)
    lr1.fit_(x, y)
    # print(lr1.thetas)
    y_hat = lr1.predict_(x)
    theta0_star, theta1_star = lr1.thetas 
    # plt.scatter(x, y, label="S_true(pill)")
    # plt.xlabel("Quantity of blue pill (in milligrams)")
    # plt.ylabel("Space driving score")
    # plt.plot(x, y_hat, color="green", linestyle="--", label="S_predict(pill)")
    # plt.legend()
    # plt.show()

    # theta0_values = [
    #     theta0_star - 10,
    #     theta0_star - 5,
    #     theta0_star,
    #     theta0_star + 5,
    #     theta0_star + 10,
    #     theta0_star + 15]

    # theta1_range = np.linspace(theta1_star - 8, theta1_star + 8, 200)
    # # Plot curves
    # for i, t0 in enumerate(theta0_values):
    #     J_values = []
    #     for t1 in theta1_range :
    #         lr2 = mlr(np.array([[t0], [t1]]))
    #         J_values.append(lr2.loss_(y, lr2.predict_(x)))
    #     gray = 0.2 + 0.12 * i
    #     plt.ylim((10,150))
    #     plt.plot(theta1_range, J_values,color=str(gray), label="lala")
    # plt.grid(True)
    # plt.show()

    linear_model1 = mlr(np.array([[89.0], [-8]]))
    linear_model2 = mlr(np.array([[89.0], [-6]]))
    y_hat1 = linear_model1.predict_(x)
    y_hat2 = linear_model2.predict_(x)
    print(mlr.mse_(y, y_hat1))
    # 57.60304285714282
    print(mean_squared_error(y, y_hat1))
    # 57.603042857142825
    print(mlr.mse_(y, y_hat2))
    # 232.16344285714285
    print(mean_squared_error(y, y_hat2))
    # 232.16344285714285
