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

_A refresher on some distributions_:  
- The _gaussian distribution_ is characterized by the probability density function $f(y) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, where $\mu$ is the mean and $\sigma^2$ is the variance. Here is a breakdown of this function because it's very nice to look at; ([source](https://youtu.be/UVvuwv-ne1I?si=z6PAEIGR1uOdMoG9)):  
    - The $e^{-\frac{1}{2}}$ term gives it its shape, it's a bell curve. The exponent is negative, so that the curve is concave down (particularly at $\mu ± \sigma$), and the $e$ is there to make it positive, so that the curve is always positive.  
    - The $(x-\mu)^2$ term is the exponent centered around the mean $\mu$  
    - The $\sigma^2$ term in the denominator of teh exponent is the variance to make it evenly distributed, and controls the spread of the curve, it's put in the denominator because the exponent is negative, this the variance becomes directly proportional to the spread of the curve   
    - The $\frac{1}{\sqrt{2\pi\sigma^2}}$ term is the normalization constant, it ensures that the distribution is indeed a probability density function, i.e. $\int_{-\infty}^{\infty} e^{-\frac{(x-\mu)^2}{2\sigma^2}}dy = \sqrt{2\pi\sigma^2}$, and multiplying by the $\frac{1}{\sqrt{2\pi\sigma^2}}$ term ensures that the area under the curve is one ($\int_{-\infty}^{\infty} f(x)dx = 1$)

    | x        | $-\inf$ | $\mu - \sigma$ | $\mu$ | $\mu + \sigma$ | $+\inf$ |
    |----------|---------|----------------|-------|----------------|---------|
    | f(x)     | 0       | less than max  | max   | less than max  | 0       |
    | f'(x)    | +       |      +         | 0     |      -         | -       |
    | f''(x)   | +       |      0         | -     |      0         | +       |

    ![normal dist](./assets/gaussian.png)
- The _binomial distribution_ is characterized by the probability mass function $f(y) = \binom{n}{y}p^y(1-p)^{n-y}$, where $n$ is the number of trials, $y$ is the number of successes, and $p$ is the probability of success. The binomial distribution is used to model the _number of successes_ in a _fixed number of trials_ 
- The _poisson distribution_ is characterized by the probability mass function $f(y) = \frac{\lambda^y}{y!}e^{-\lambda}$, where $\lambda$ is the average number of events (also std. dev.) in a fixed interval of time. The poisson distribution is used to model the _number of events_ in a _fixed interval of time_

_note on probability functions_:  
- A _probability density function_ is a function that describes the likelihood of a **continuous** random variable taking on a particular value. The area under the curve of a probability density function =1    
- A _probability mass function_ is a function that describes the likelihood of a **discrete** random variable taking on a particular value. The sum of the probabilities of all possible values of a discrete random variable =1

_Link functions and loss functions:_  

What distinguishes GLM from the linear model is the ___link function___ that connects the linear predictor to the expected value of the response variable.  
The linear predictor is defined as $ \eta = X\beta$, where $\beta$ is a $(p \times 1)$ column vector of coefficients. The link function is a function $g$ such that $g(\mu) = \eta$, where $\mu$ is the expected value of the response variable. 
The link function is chosen based on the distribution of the response variable. For example, in the case of the _gaussian distribution_, the link function is the _identity function_, while for the _binomial distribution_, the link function is the _logit function_.

Another important concept in GLM is the ___loss function___, which is a function that measures the difference between the predicted value and the actual value of the response variable. The loss function is used to estimate the coefficients of the model by minimizing the difference between the predicted value and the actual value of the response variable. It's usually chosen based on the distribution of $y$, for example, in the case of the _gaussian distribution_, the loss function is the _squared error function_, while for the _binomial distribution_, the loss function is the _log loss function_ (not going to dive deep in it, but it's the negative log likelihood of the observed data given the model).  

In each model, the coefficients are estimated by minimizing the loss function. The linear model uses _Ordinary Least Squares_ (OLS) to estimate the coefficients, while the logistic and poisson models use _Maximum Likelihood Estimation_ (MLE) to estimate the coefficients. The coefficients are estimated by maximizing the likelihood of the observed data given the model. The ***likelihood*** is a function of the coefficients and is maximized by finding the coefficients that maximize the likelihood of the observed data.   
So our goal is to get the $\beta$ that optimizes the performance of the model, i.e. minimizes the loss function or maximizes the likelihood function.

_Evaluating the model:_  

An important aspect of building a GLM is to understand the statistical properties of the model. The statistical properties of the model are used to evaluate the performance of the model and to make predictions based on it, using some statistical properties like _residuals_, _deviance_, and _confidence intervals_.  
Evaluating the performance of a model, we often care to see which predictors are significant, and which are not. This is done by performing _hypothesis tests_ on the coefficients of the model.  
- $H_0: \beta_i = 0$; the _t-test_ is used to test the significance of the coefficients of the model, used to determine whether they are significantly different from zero. If the _p-value_ of the _t-test_ is less than a certain threshold (usually 0.05), then the coefficient is considered to be significant and the null hypothesis is rejected.  
- $H_0: \beta_i = \beta_j$; the _F-test_ is used to test the significance of the predictors of the model, used to determine whether the predictors are significantly different from each other. If the _p-value_ of the _F-test_ is less than a certain threshold (usually 0.05), then the predictors are considered to be significant and the null hypothesis is rejected.  
- $H_0: \beta_i = 0$; the _Wald test_, used in logistic and poisson, also tests the significance of the coefficients of the model.  

We can compare between models and nested models by _anova_ which comes to compare the deviance of the models. The _deviance_ is a measure of the goodness of fit of the model and is used to compare the performance of different models.  

Important metrics:  
- $R^2$ (linear regression): proportion of variance explained by the model, $R^2 = \frac{ESS}{TSS}$, where $ESS$ is the explained sum of squares and $TSS$ is the total sum of squares. RSS is the residual sum of squares, $RSS = \sum_{i=1}^{n} (y_i - \hat y_i)^2$, $ESS = \sum_{i=1}^{n} (\hat y_i - \bar y)^2$, and
$TSS = \sum_{i=1}^{n} (y_i - \bar y)^2$, where $\bar y$ is the mean of the response variable. In here $RSS$ is the error that the model makes, also seen as the projection of $y$ onto the space of $X\beta$. The smaller the distance is, teh closest is our $\hat y$ to $y$, the better the model is and the more $R^2 \approx 1$  
![orthogonal projraction](./assets/Rsq-projection.png)