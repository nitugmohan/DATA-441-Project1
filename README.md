# Introduction to Locally Weighted Regression


## Locally Weighted Regression

With linear regression, one can make predictions as a weighted combination of the input feature values, with positive of negative weights; it is used for computing linear relationships between an input (X) and output (Y). To put it plainly, a straight line should be able to easily split/categorize the data (linearly seperable data). This follows the equation of:

<img src="LRequation.png" class="LR" alt=""> 


However, if there is a non-linear relationship between X and Y, it might be better to utilize a locally weighted regression. This type of algorithm assigns weights to data points to overcome the problem of non-linearly seperable data. This a non-parametric algorithm meaning that it does not have set parameters like linear regression. Parameters (tau) are calculated for each data point based on the point's location. Points that are closer to x will get a higher "preference." It is important to note that this is a generalization of k-nearest Neighbor and there is not training phase. The work is done while making predictions. Weights are assigned by kernel smoothing. The kernel function is the bell shaped function with parameter tau. Larger tau will result in a smoother curve.

Definition of the kernels: https://en.wikipedia.org/wiki/Kernel_(statistics)

There are many choices of kernels for locally weighted regression. The idea is to have a function with one local maximum that has a compact support. Below is some of the different kernels we discussed in class:

```python
# Gaussian Kernel
def Gaussian(x):
  return np.where(np.abs(x)>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))
  
# Tricubic Kernel
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)
  
# Epanechnikov Kernel
def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 
  
# Quartic Kernel
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 
```


**Main Idea:** Trends and associations are generally nonlinear; however, *locally*, trends can be interpreted linearly.

In this context, local properties are relative to a metric. A metric is a method by which we compute the distance between two observations. Observations contain multiple features, and if they are numeric, we can see them as vectors in a finite-dimensional Euclidean space.

The independent observations are the rows of the matrix $X$ . Each row has a number of columns (this is the number of features) and we can denote it by $p.$ As such, every row is a vector in $\mathbb{R}^p.$ The distance between two independent observations is the Euclidean distance between the two represented $p-$ dimensional vectors. The equation is:

<img src="LWRequation.png" class="LWR" alt="">


We shall have $n$ different weight vectors because we have $n$ different observations.

The message of this picture is that we are going to use kernels, such as Gaussian or similar shapes, for solving local linear regression problems.

<p align="center">
<img src="Loess_1.drawio.svg" class="Loess" alt="">
</p>


Drawbacks:
1. Need to evaluate whole dataset everytime
2. Higher computation cost
3. More memory required


The function below is one approach for the locally weighted regression by Alex Gramfort:

```python
# approach by Alex Gramfort

def lowess_ag(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest
```




Here is another approach for locally weighted regression: 
```python
def lowess(x, y,x_new, kern, tau=0.05):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    # tau is a hyper-parameter
    w = weights_matrix(x,x_new,kern,tau) 
    if np.isscalar(x_new):
      lm.fit(np.diag(w).dot(x.reshape(-1,1)),np.diag(w).dot(y.reshape(-1,1)))
      yest = lm.predict([[x_new]])[0][0]
    else:
      n = len(x_new)
      yest = np.zeros(n)
      #Looping through all x-points
      for i in range(n):
        lm.fit(np.diag(w[i,:]).dot(x.reshape(-1,1)),np.diag(w[i,:]).dot(y.reshape(-1,1)))
        yest[i] = lm.predict(x_new[i].reshape(-1,1)) 

    return yest
```






References: 
1. https://www.geeksforgeeks.org/ml-locally-weighted-linear-regression/
2. https://xavierbourretsicotte.github.io/loess.html
