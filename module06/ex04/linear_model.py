import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as mlr

def plot_fittest_line(x, y, y_hat) :
    plt.scatter(x, y, label="S_true(pill)")
    plt.xlabel("Quantity of blue pill (in milligrams)")
    plt.ylabel("Space driving score")
    plt.plot(x, y_hat, color="green", linestyle="--", label="S_predict(pill)")
    plt.title("Linear regresssion of Space driving score as a function of the quantity of blue pill (in micrograms).")
    plt.legend()
    plt.show()

def plot_relation_theta_cost(x, y, theta0_star, theta1_star) :
    theta0_values = [
        theta0_star - 10,
        theta0_star - 5,
        theta0_star,
        theta0_star + 5,
        theta0_star + 10,
        theta0_star + 15]

    theta1_range = np.linspace(theta1_star - 8, theta1_star + 8, 200)
    # Plot curves
    for i, t0 in enumerate(theta0_values):
        J_values = []
        for t1 in theta1_range :
            lr2 = mlr(np.array([[t0], [t1]]))
            J_values.append(lr2.loss_(y, lr2.predict_(x)))
        gray = 0.2 + 0.12 * i
        plt.ylim((10,150))
        plt.plot(theta1_range, J_values,color=str(gray), label=f"vector theta{i}")
    plt.grid(True)
    plt.title("Relation and impact of the parameter theta0(intercept) and theta1(slope) on the cost function behavior")
    plt.legend()
    plt.show() 

def main() :
    data = np.genfromtxt("are_blue_pills_magics.csv", delimiter=",", skip_header=1)
    x = data[:,1]
    y = data[:,2]
    lr1 = mlr(np.array([0, 0]), alpha=0.01, max_iter=10000)
    lr1.fit_(x, y)
    y_hat = lr1.predict_(x)
    theta0_star, theta1_star = lr1.thetas
    #FITTEST PARAM LINE
    plot_fittest_line(x, y, y_hat)
    #SHOW RELATION BETWEEN THETA VECTOR AND THE COST FUNCTION
    plot_relation_theta_cost(x, y, theta0_star, theta1_star)

    #COMPARE MSE METHOD WITH REAL ONE
    linear_model1 = mlr(np.array([[89.0], [-8]]))
    linear_model2 = mlr(np.array([[89.0], [-6]]))
    y_hat1 = linear_model1.predict_(x)
    y_hat2 = linear_model2.predict_(x)
    print(mlr.mse_(y, y_hat1))
    print(mean_squared_error(y, y_hat1))
    print(mlr.mse_(y, y_hat2))
    print(mean_squared_error(y, y_hat2))

if __name__ == "__main__" :
    main()





