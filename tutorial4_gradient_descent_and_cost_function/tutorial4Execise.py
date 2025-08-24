# This program is add gradient descent flow and also to compare the result is near to sklearn linear regression or not
# we will check the coeffient and the intercept
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def get_df():
    df = pd.read_csv('tutorial4_execise_test_scores.csv')
    return df

def sklearn(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    return reg.coef_, reg.intercept_


def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0
    n = len(x)
    learning_rate = 0.0002
    iteration = 1000000
    previous_cost = 0
    for i in range(iteration):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, previous_cost, rel_tol=1e-20):  # Here we are comparing whether previous cost value is close to current cost or not. if close, I will stop the iteration
            break
        previous_cost = cost
        print("m: {}, b: {}, cost: {}, Iteration: {}".format(m_curr, b_curr, cost, i))
    return m_curr, b_curr

df = get_df()
print(df)
x = np.array(df.math)
y = np.array(df.cs)
print(x)
print(y)
grdCoef, grdInter = gradient_descent(x, y)
skCoef, skInter = sklearn(df[['math']], df.cs)

print('Gradient Descent:: Ceof: {} Intercept: {}'.format(grdCoef, grdInter))
print('Sklear:: Ceof: {} Intercept: {}'.format(skCoef, skInter))
# Check above two are nearly same or not. It should be close to the values

