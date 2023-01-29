import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt 

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))  # it must have one column and as many rows as necessary
y = np.array([5, 20, 14, 32, 22, 38]) 

#model = LinearRegression()

#This statement creates the variable model as an instance of LinearRegression. You can provide several optional parameters to LinearRegression:

# fit_intercept     is a Boolean that, if True, decides to calculate the intercept 𝑏₀ or, if False, considers it equal to zero. It defaults to True.
# normalize         is a Boolean that, if True, decides to normalize the input variables. It defaults to False, in which case it doesn’t normalize the input variables.
# copy_X            is a Boolean that decides whether to copy (True) or overwrite the input variables (False). It’s True by default.
# n_jobs            is either an integer or None. It represents the number of jobs used in parallel computation. It defaults to None, which usually means one job. -1 means to use all available processors.

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
# how much x explains y 

print(f"intercept: {model.intercept_}")
intercept: 5.633333333333329
# represents the coefficient 𝑏₀

print(f"slope: {model.coef_}")
# represents 𝑏₁

# The value of 𝑏₀ is approximately 5.63. 
# This illustrates that your model predicts the response 5.63 when 𝑥 is zero. 
# The value 𝑏₁ = 0.54 means that the predicted response rises by 0.54 when 𝑥 is increased by one.

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

# When applying .predict(), you pass the regressor as the argument and get the corresponding predicted response. 
# This is a nearly identical way to predict the response:
