# Introduction to Locally Weighted Regression


## Locally Weighted Regression

With linear regression, one can make predictions as a weighted combination of the input feature values, with positive of negative weights; it is used for computing linear relationships between an input (X) and output (Y). To put it plainly, a straight line should be able to easily split/categorize the data (linearly seperable data). This follows the equation of:

<img src="LRequation.png" class="LR" alt=""> 


However, if there is a non-linear relationship between X and Y, it might be better to utilize a locally weighted regression. This type of algorithm assigns weights to data points to overcome the problem of non-linearly seperable data. This a non-parametric algorithm meaning that it does not have set parameters like linear regression. Parameters (tau) are calculated for each data point based on the point's location. Points that are closer to x will get a higher "preference." It is important to note that this is a generalization of k-nearest Neighbor and there is not training phase. The work is done while making predictions. Weights are assigned by kernel smoothing. The kernel function is the bell shaped function with parameter tau. Larger tau will result in a smoother curve.

Definition of the kernels: https://en.wikipedia.org/wiki/Kernel_(statistics)

There are many choices of kernels for locally weighted regression. The idea is to have a function with one local maximum that has a compact support. Below is some of the different kernels we discussed in class:

```python
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


Drawbacks:
1. Need to evaluate whole dataset everytime
2. Higher computation cost
3. More memory required








**Main Idea:** Trends and associations are generally nonlinear; however, *locally*, trends can be interpreted linearly.

In this context, local properties are relative to a metric. A metric is a method by which we compute the distance between two observations. Observations contain multiple features, and if they are numeric, we can see them as vectors in a finite-dimensional Euclidean space.

The independent observations are the rows of the matrix $X$ . Each row has a number of columns (this is the number of features) and we can denote it by $p.$ As such, every row is a vector in $\mathbb{R}^p.$ The distance between two independent observations is the Euclidean distance between the two represented $p-$ dimensional vectors. The equation is:

<img src="LWRequation.png" class="LWR" alt="">


We shall have $n$ different weight vectors because we have $n$ different observations.

The message of this picture is that we are going to use kernels, such as Gaussian or similar shapes, for solving local linear regression problems.

<p align="center">
<img src="Loess_1.drawio.svg" class="Loess" alt="">
</p>

## Different Kernels



1.   The Exponential Kernel

$$ K(x):= e^{-\frac{\|x\|^2}{2\tau}}$$


2.   The Tricubic Kernel

$$ K(x):=\begin{cases}
(1-\|x\|^3)^3 \;\;\;if \;\;\; \|x\|<1 \\
0 \;\;\; \text{otherwise}
\end{cases}
$$

3.   The Epanechnikov Kernel

$$ K(x):=\begin{cases}
\frac{3}{4}(1-\|x\|^2) \;\;\;if \;\;\; \|x\|<1 \\
0 \;\;\; \text{otherwise}
\end{cases}
$$

3.   The Quartic Kernel

$$ K(x):=\begin{cases}
\frac{15}{16}(1-\|x\|^2)^2 \;\;\;if \;\;\; \|x\|<1 \\
0 \;\;\; \text{otherwise}
\end{cases}
$$




```python
# Libraries of functions need to be imported

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from scipy import linalg
```



References: 
https://www.geeksforgeeks.org/ml-locally-weighted-linear-regression/
https://xavierbourretsicotte.github.io/loess.html
