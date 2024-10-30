# Generalized Linear Models library from scratch

This is a python implementation of _Generalized Linear Models_ inspired by the theoretical material taken in a _Multivariate Statistics_ course and built based on _object-oriented programming_ design principles. Sources:  
- [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) by Hastie, Tibshirani, and Friedman  
- [Introduction to Generalized Linear Models]() by Dobson and Barnett   
- [Linear Models with R]() by Faraway

## Design

### Directory Structure

The library is structured in the following way:
```text
GLM-from-scratch/
│
├── glm/
│   ├── __init__.py
│   ├── glm.py
│   ├── link_functions.py
│   ├── loss_functions.py
│   ├── utils.py
│   └── validation.py
│
├── tests/
│   ├── __init__.py
│   ├── test_glm.py
│   └── test_link_functions.py
│
├── .gitignore
├── LICENSE
└── README.md   
```

### UML Diagram

_uunder construction_

## Some theory and concepts

Generalized Linear Models (GLM) are a class of models including linear, logistic and poisson regression, among others. These are generalizations of the linear model, following one of the exponential family distribution (gaussian, binomial or poisson).  
The aim of these models is to model the relationship between the response variable $y$ ( a $(n \times 1)$ column vector) and the predictors $X$ ( a $(n \times p)$ matrix representing $p$ predictors), and to make predictions based on this relationship.  
Hence, the model is defined as $y = f(X\beta) + \epsilon$, where $f$ is a link function, $\beta$ is a $(p \times 1)$ column vector of coefficients, and $\epsilon$ is the error term. What we're looking for is $\hat y$, the predicted value of $y$ given $X$ and $\beta$, such that $\hat y = f(X\beta)$, this is also denoted as $E(y|X,\beta)$, the expected value of $y$ given $X$ and $\beta$

In a linear model, $y = X\beta + \epsilon$, and the error term is assumed to be normally distributed, indpendent and homoscedastic: $\epsilon \sim N(0, \sigma^2)$. The linear model is a special case of the GLM, where the response variable follows a gaussian distribution.   


_Link functions and loss functions:_  

What distinguishes GLM from the linear model is the ___link function___ that connects the linear predictor to the expected value of the response variable.  
The linear predictor is defined as $ \eta = X\beta$, where $\beta$ is a $(p \times 1)$ column vector of coefficients. The link function is a function $g$ such that $g(\mu) = \eta$, where $\mu$ is the expected value of the response variable. 
The link function is chosen based on the distribution of the response variable. For example, in the case of the _gaussian distribution_, the link function is the _identity function_, while for the _binomial distribution_, the link function is the _logit function_.

Another important concept in GLM is the ___loss function___, which is a function that measures the difference between the predicted value and the actual value of the response variable. The loss function is used to estimate the coefficients of the model by minimizing the difference between the predicted value and the actual value of the response variable. It's usually chosen based on the distribution of $y$, for example, in the case of the _gaussian distribution_, the loss function is the _squared error function_, while for the _binomial distribution_, the loss function is the _log loss function_ (not going to dive deep in it, but it's the negative log likelihood of the observed data given the model).  

In each model, the coefficients are estimated by minimizing the loss function. The linear model uses _Ordinary Least Squares_ (OLS) to estimate the coefficients, while the logistic and poisson models use _Maximum Likelihood Estimation_ (MLE) to estimate the coefficients. The coefficients are estimated by maximizing the likelihood of the observed data given the model. The ***likelihood*** is a function of the coefficients and is maximized by finding the coefficients that maximize the likelihood of the observed data.   
So our goal is to get the $\beta$ that optimizes the performance of the model, i.e. minimizes the loss function or maximizes the likelihood function.

_Evaluating the model:_  

An important aspect of building a GLM is to understand the statistical properties of the model, like _residuals_, _deviance_, and _confidence intervals_.  
Evaluating the performance of a model, we often care to see which predictors are significant, and which are not. This is done by performing _hypothesis tests_ on the coefficients of the model.  

We can compare between models and nested models by _anova_ which comes to compare the deviance of the models. The _deviance_ is a measure of the goodness of fit of the model and is used to compare the performance of different models. Other measures of goodness of fit include the _AIC_ and _BIC_, and even $R^2$, _odd ratios_, and _incident rate ratios_ are used to evaluate the performance of the model.

_Some other important concepts in GLM are:_

### Probability distributions

_A refresher on some distributions_

- The _gaussian distribution_ is characterized by the probability density function $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, where $\mu$ is the mean and $\sigma^2$ is the variance. Here is a breakdown of this function because it's very nice to look at; ([source](https://youtu.be/UVvuwv-ne1I?si=z6PAEIGR1uOdMoG9)):  
    - The $e^{-\frac{1}{2}}$ term gives it its shape, it's a bell curve. The exponent is negative, so that the curve is concave down (particularly at $\mu ± \sigma$, this is why the 2 is added in denominator to scale it here instead of 2 times this value - check other elements of the exponent), and the $e$ is there to make it positive, so that the curve is always positive.  
    - The $(x-\mu)^2$ term is the exponent centered around the mean $\mu$  
    - The $\sigma^2$ term in the denominator of the exponent is the variance to make it evenly distributed, and controls the spread of the curve, it's put in the denominator because the exponent is negative, this the variance becomes directly proportional to the spread of the curve   
    - The $\frac{1}{\sqrt{2\pi\sigma^2}}$ term is the normalization constant, it ensures that the distribution is indeed a probability density function, i.e. $\int_{-\infty}^{\infty} e^{-\frac{(x-\mu)^2}{2\sigma^2}}dx = \sqrt{2\pi\sigma^2}$, and multiplying by the $\frac{1}{\sqrt{2\pi\sigma^2}}$ term ensures that the area under the curve is one ($\int_{-\infty}^{\infty} f(x)dx = 1$)


        | x        | $-\inf$ | $\mu - \sigma$ | $\mu$ | $\mu + \sigma$ | $+\inf$ |
        |----------|---------|----------------|-------|----------------|---------|
        | f(x)     | 0       | less than max  | max   | less than max  | 0       |
        | f'(x)    | +       |      +         | 0     |      -         | -       |
        | f''(x)   | -       |      0         | +     |      0         | -       |

    ![normal dist](./assets/gaussian.png)
- The _binomial distribution_ is characterized by the probability mass function $f(y) = \binom{n}{y}p^y(1-p)^{n-y}$, where $n$ is the number of trials, $y$ is the number of successes, and $p$ is the probability of success. The binomial distribution is used to model the _number of successes_ in a _fixed number of trials_ 
- The _poisson distribution_ is characterized by the probability mass function $f(y) = \frac{\lambda^y}{y!}e^{-\lambda}$, where $\lambda$ is the average number of events (also std. dev.) in a fixed interval of time. The poisson distribution is used to model the _number of events_ in a _fixed interval of time_

_note on density vs mass_:  
- A _probability density function_ is a function that describes the likelihood of a **continuous** random variable taking on a particular value. The area under the curve of a probability density function =1    
- A _probability mass function_ is a function that describes the likelihood of a **discrete** random variable taking on a particular value. The sum of the probabilities of all possible values of a discrete random variable =1

### Least Squares

The least squares approach allows to get the $\hat y$ that minimizes the distance between $y$ and $X\beta$ (the projection of $y$ onto the space of $X\beta$).   
The _residuals_ are the difference between the observed value of the response variable and the predicted value of the response variable, $e = y - \hat y$. The residuals are used to evaluate the performance of the model and to make predictions based on it.  

![least squares](./assets/Rsq-projection-2.png)

_In this fig it's worth noting the 2 types of errors we have_:  
* __error of prediction__ ($\hat \epsilon$, predicting a new y from a new obs of X will have an error that's based on $X\hat\beta$ and $y$)  
* __error of estimation__ (diff between $X\beta\  and\ X \hat \beta$, when we do sestimation we don't know real values of $\beta$, we will estimate $\hat\beta$ and predicted value will be $X\hat\beta$)  
$predicted\ error \gt estimation\ error$.

To estimate the coefficients of the model, we minimize the _residual sum of squares_ (RSS), which is the sum of the squared residuals, $RSS = \sum_{i=1}^{n} (y_i - \hat y_i)^2$. The coefficients are estimated by minimizing the RSS, i.e. finding the coefficients that minimize the difference between the observed value of the response variable and the predicted value of the response variable.  
Enters the _normal equation_ that gives us the $\hat \beta$ that minimizes the RSS, $\hat \beta = (X^TX)^{-1}X^Ty$. So $E(y|X,\beta) = X\beta = X(X^TX)^{-1}X^Ty = Hy$, where $H = X(X^TX)^{-1}X^T$ is the _hat matrix_ (orthogonal projection) that projects $y$ onto the space of $X\beta$.

$\hat \beta = (X^TX)^{-1}X^Ty$ is true $iff$ $X^TX$ is invertible, and $X$ is full rank, which is the case when the predictors are linearly independent. So can conclude 2 clear scenarios where this isn't met:  
  
- $p>n$, we have more predictors than observations, so we can't estimate the coefficients, the rank of $X \lt p$ thus $X^TX$ is not invertible  
- When the predictors are linearly dependent, the rank of $X \lt p$ thus $X^TX$ is not invertible  

To deal with 1, use _regularization_ (L1, L2, elastic net), or _dimensionality reduction_ (PCA), or _feature selection_ (backward, forward, stepwise selection).

To deal with the 2nd case, we can either remove colinear predictors or (do smtg else i forgot what). When X is not full rank than some predictors are linearly dependent on the others. Interestingly, in practice we might have a predictor that is close to being a linear combination of other predictors (but not exactly), in this case to test for multicollinearity we can:    
- comparing ratio between the largest and smallest eigenvalues of teh corr matrix R (the smallest eigenvalue is 0 when it's not full rank), if it's very large, then we have multicollinearity, there is no stat test to check if it's significantly greater than 0 but some empirical values are used to check for it like 500 or 1000 (arbitrary threshold); thus when ration is greater than this value, we have strong colinearity.  
- perform several Linear regressions between preductors and check the $R^2$ values, if they are close to 1. i.e., try to explain one predictor with the others, the model that shows a near 1 $R^2$ means that this predictor is very well explained by the others, it's a linear combination of them thus it's a good candidate to be removed. (e.g., explain x1 by x2 & x3, then x2 by x1 & x3, then x3 by x1 & x2, check for $R^2$ values)   
VIF (Variance Inflation Factor) is used here, as $VIF = \frac{1}{1-R^2}$, when $R^2 \approx 1$, then $VIF \approx \infty$, and when $R^2 \approx 0$, then $VIF \approx 1$ (compute a VIF for each predictor)

_But what is $R^2$?_


- $R^2$: proportion of variance explained by the model, $R^2 = \frac{ESS}{TSS}$, where $ESS$ is the explained sum of squares and $TSS$ is the total sum of squares. RSS is the residual sum of squares, $RSS = \sum_{i=1}^{n} (y_i - \hat y_i)^2$, $ESS = \sum_{i=1}^{n} (\hat y_i - \bar y)^2$, and
$TSS = \sum_{i=1}^{n} (y_i - \bar y)^2$, where $\bar y$ is the mean of the response variable. In here $RSS$ is the error that the model makes, also seen as the projection of $y$ onto the space of $X\beta$. The smaller the distance is, the closest is our $\hat y$ to $y$, the better the model is and the more $R^2 \approx 1$  
![orthogonal projraction](./assets/Rsq-projection.png)  

- Adjusted $R^2$: penalizes the $R^2$ for the number of predictors in the model, $R^2_{adj} = 1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$, where $p$ is the number of predictors in the model. When the number of predictors increases, the $R^2$ also increases, even if the predictors are not significant. It's mainly used to compare the performance of models with different numbers of predictors.