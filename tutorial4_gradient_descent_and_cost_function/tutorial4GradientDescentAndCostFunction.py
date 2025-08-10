import numpy as np

def gradient_descent(x, y):
    m_curr = b_curr = 0  #initially we are taking x and y as 0
    iterations = 10000    # Here we are giving how my steps, initially we are taking with 1000. then we will fine tune
    n = len(x)
    learning_rate = 0.08  # we can take any decimal number to slowly. Eg: 0.01 or 0.001 or 0.005 and then we can fine tune using trail and error
    for i in range(iterations):
        y_predict = m_curr * x + b_curr   # giving formula y = mx+b
        cost = (1/2) * sum([val**2 for val in (y - y_predict)]) # this formula for finding cost
        md = -(2/n)*sum(x*(y - y_predict)) # here also giving other forumla
        bd = -(2/n)*sum(y - y_predict) # here also given other formula
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))
        #Initially we can give less iteration and with higher learning rate difference. Then reduce the learning rate with more decimals
        # Our goal is to reduce the cost as much. It should not increase.
        # We should start learning rate with first 0.1. 0.09, 0.08 etc. We need to find the minimum and stick to that learning rate
        # Once we stick to the learning rate, then increase the iteration count from 10 to 100, or 1000 or 10000
        # Once we reached the cost will remain nearly same

x = np.array([1, 2, 3, 4, 5])    #usually numpy array is faster
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x,y)