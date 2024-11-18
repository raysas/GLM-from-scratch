# Generalized Linear Models library from scratch

This is a python implementation of _Generalized Linear Models_ inspired by the theoretical material taken in a _Multivariate Statistics_ course and built based on _object-oriented programming_ design principles. Sources:  
- [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) textbook by Hastie, Tibshirani, and Friedman  
- [Introduction to Statistical Learning](https://hastie.su.domains/ISLR/) textbook by James, Witten, Hastie, and Tibshirani
- [Introduction to Generalized Linear Models]() textbook by Dobson and Barnett   
- [Linear Models with R]() textbook by Faraway   
- [Generalized Linear Models (GLM's)](https://www.youtube.com/playlist?list=PLJ71tqAZr197DkSiGT7DD9dMYxkyZX0ti) youtube playlist by Meerkat Statstics

## Design

### Directory Structure

_under construction_

<!-- The library is structured in the following way:
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
``` -->

### UML Diagram

_under construction_

## Some theory and concepts

_beware some math and stats ahead :p_

### Main ideas:

Linear models were concieved originally by Gauss in the aim of finding a relationship between a response variable $y$ and explanatory variables $X$. This was based on 3 assumptions:  
- $\epsilon_i$ are independent (ensures that $\hat\beta$ are unbiased estimators of the true coefficients by having $\mathbb{E}(\epsilon) =0 \iff \mathbb{E}(y) = X\beta$), this makes the off-diagonal elements of the covariance matrix of $\hat\beta$ to be 0
- $\epsilon_i$ are identically distributed (ensures that the variance of the residuals is constant, and it's $\sigma^2$)  
- $\epsilon \sim N(0, \sigma^2)$ (ensures that the residuals are normally distributed)  

So i.i.d $\epsilon \sim N(0, \sigma^2)$ with this covariance matrix:  
$\begin{bmatrix} \sigma^2 & 0 & 0 & \cdots & 0 \\ 0 & \sigma^2 & 0 & \cdots & 0 \\ 0 & 0 & \sigma^2 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & \sigma^2 \end{bmatrix}$

This is mind, $\epsilon$ is the only random variable in the model, $X$ and $\beta$ are fixed, so the distribution of $y$ is normal, $y \sim N(\mu, \sigma^2)$, where $\mu = X\beta$.

- $y$ are independent
- $y$ are normally distributed with a mean $\mu$ so $y_i \sim N(\mu_i, \sigma^2)$  
- $\mu_i = X_i^T\beta$, the mean $\mu$ are related to the predictors $X$ by a linear model (linear in the parameters $\beta$, the $X$ can be transformed)

Then came the _[Generalized Linear Models](#generalized-linear-models-glm)_ (GLM) that are a generalization of the linear model, where the response variable $y$ follows a distribution from the _[exponential family](#exponential-family)_ (gaussian, binomial, poisson, gamma, etc.). As is the case in linear models, the aim of these models is to model the relationship between the response variable $y$ and the predictors $X$, and to make predictions based on this relationship. Here there are some few generalizations:  
- $y_i \sim exponentional\ family$  
- $g(\mu_i) = X_i^T\beta$, the mean $\mu$ are related to the predictors $X$ by a [link function](#link-functions-and-loss-functions) $g$ (not necessarily linear, logit, log, inverse...).  

In regular linear models, can use _[Least Squares](#linear-models-least-squares-and-residuals)_ or _[Maximum Likelihood Estimation](#maximum-likelihood-estimation)_ to estimate the coefficients $\beta$ (they are exactly the same in a normal distribution. In GLM only Maximum Likelihood - can think of OLS as a special case)

| Linear Model | Generalized Linear Model |
|--------------|---------------------------|
| $y_i \sim N(\mu_i, \sigma^2)$ | $y_i \sim Exponential\ family$ |
| $\mu_i = X_i^T\beta$ | $g(\mu_i) = X_i^T\beta$ |
| solved by OLS and MLE | solved by MLE |

In linear models, estimating the parameters come from solving one of these optimization problems:  
- **[Least Squares](#linear-models-least-squares-and-residuals)**: Optimization problem where we wanna minimize the sum of squared residuals, $min \sum_{i=1}^{n} (y_i - \hat y_i)^2$, where $\hat y_i = X_i^T\beta$; solved by differentiating and equating to 0. 
- **[Maximum Likelihood](#maximum-likelihood-estimation)**: Assuming some distribution on $y$, in LM $y \sim N(\mu, \sigma^2)$, where $\mu_i = X_i^T\beta$ sometimes $\mu_i$ will be above and sometimes below the line.  
Probability of obtaining that $y_i$ is $ \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i-\mu_i)^2}{2\sigma^2}}$, we want to maximize this probability (choose the $\beta's$ that do), so we take the log likelihood, it's be a ***sum*** of the logs:  
$l(\beta) = \sum_{i=1}^{n} log(\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i-\mu_i)^2}{2\sigma^2}})= \sum_{i=1}^{n} log(\frac{1}{\sqrt{2\pi\sigma^2}}) - \sum_{i=1}^{n} \frac{(y_i-\mu_i)^2}{2\sigma^2}$  
The goal is to maximize this quantity  
$\iff$ maximize $- \sum_{i=1}^{n} \frac{(y_i-\mu_i)^2}{2\sigma^2}$ WHICH IS  
$\iff$ minimize $\sum_{i=1}^{n} (y_i - \mu_i)^2$  
$\iff$ minimize the sum of squared residuals (same as OLS).

| Least Squares | Maximum Likelihood |
|---------------|--------------------|
| $min \sum_{i=1}^{n} (y_i - \hat y_i)^2$ | $max \sum_{i=1}^{n} log(\frac{1}{\sqrt{2\pi\sigma^2}}) - \sum_{i=1}^{n} \frac{(y_i-\mu_i)^2}{2\sigma^2}$ |

![ML in LM](./assets/max_likelihood.png)



### Generalized Linear Models (GLM):

Generalized Linear Models (GLM) are a class of models including linear, logistic and poisson regression, among others. These are generalizations of the linear model, following one of the exponential family distribution.  
The aim of these models is to model the relationship between the response variable $y$ ( a $(n \times 1)$ column vector) and the predictors $X$ ( a $(n \times p)$ matrix representing $p$ predictors), and to make predictions based on this relationship.  
Hence, the model is defined as $y = f(X\beta) + \epsilon$, where $f$ is a link function, $\beta$ is a $(p \times 1)$ column vector of coefficients, and $\epsilon$ is the error term. What we're looking for is $\hat y$, the predicted value of $y$ given $X$ and $\beta$, such that $\hat y = f(X\beta)$, this is also denoted as $E(y|X,\beta)$, the expected value of $y$ given $X$ and $\beta$.  

The parameters of the model are estimated by maximizing the likelihood of the observed data given the model. The likelihood is a function of the coefficients and is maximized by differentiating and equating to 0. This is equivalent to minimizing the loss function, which is a function that measures the difference between the predicted value and the actual value of the response variable (litteraly, negative log likelihood). The model is usually evaluated by deviance, a measure of the goodness of fit of the model and is used to compare the performance of different models.

### Linear Models, Least Squares, and Residuals

In a linear model, $y = X\beta + \epsilon$, and the error term is assumed to be normally distributed, indpendent and homoscedastic: $\epsilon \sim N(0, \sigma^2)$. The linear model is a special case of the GLM, where the response variable follows a gaussian distribution.  

Let $V$ be the space of X, $X\gamma \in V$, and $y$ is a vector in $\mathbb{R}^n$.  
The least squares approach allows to get the $\hat y(=X\hat\beta)$ that minimizes the distance between $y$ and $X$ (the projection of $y$ onto the space of $X$), in other words, we're searching for the $\hat\beta$ from $argmin_{\gamma} \sum_{i=1}^{n} (y_i - X_i^T\gamma)^2$ (the $\gamma$ that minimizes the sum of squared residuals).  
The _residuals_ are the difference between the observed value of the response variable and the predicted value of the response variable, $e = y - \hat y$. The residuals are used to evaluate the performance of the model and to make predictions based on it.   
$\beta$ signifies the real coefficient describing the relationship between the predictors and the response variable, while $\hat \beta$ is the estimated coefficient that best describes this relationship, $\mathbb{E}(\beta)=\hat\beta$ ([unbiased estimator*](#unbiased-estimators), i.e. $\hat\beta \approx \beta$ when $n \to \infty$).

![least squares](./assets/Rsq-projection-2.png)

_In this fig it's worth noting the 2 types of errors we have_:  
* __error of prediction__ ($\hat \epsilon$, predicting a new y from a new obs of X will have an error that's based on $X\hat\beta$ and $y$)  
* __error of estimation__ (diff between $X\beta\  and\ X \hat \beta$, when we do estimation we don't know real values of $\beta$, we will estimate $\hat\beta$ and predicted value will be $X\hat\beta$)  
$predicted\ error \gt estimation\ error$.

To estimate the coefficients of the model, we minimize the _residual sum of squares_ (RSS), which is the sum of the squared residuals, $RSS = \sum_{i=1}^{n} (y_i - \hat y_i)^2$. The coefficients are estimated by minimizing the RSS, i.e. finding the coefficients that minimize the difference between the observed value of the response variable and the predicted value of the response variable.  
Enters the _normal equation_ that gives us the $\hat \beta$ that minimizes the RSS, $\hat \beta = (X^TX)^{-1}X^Ty$.   
So $E(y|X,\beta) = X\beta = X(X^TX)^{-1}X^Ty = Hy$,  
where $H = X(X^TX)^{-1}X^T$ is the _hat matrix_ (orthogonal projection) that projects $y$ onto the space of $X\beta$.

$\hat \beta = (X^TX)^{-1}X^Ty$ is true $iff$ $X^TX$ is invertible, and $X$ is full rank, which is the case when the predictors are linearly independent. So can conclude 2 clear scenarios where this isn't met:  
  
- $p>n$, we have more predictors than observations, so we can't estimate the coefficients, the rank of $X \lt p$ thus $X^TX$ is not invertible  
- When the predictors are linearly dependent, the rank of $X \lt p$ thus $X^TX$ is not invertible  

To deal with 1, use _regularization_ (L1, L2, elastic net), or _dimensionality reduction_ (PCA), or _feature selection_ (backward, forward, stepwise selection).

To deal with the 2nd case, we can either remove colinear predictors or (do smtg else i forgot what). When $X$ is not full rank than some predictors are linearly dependent on the others. Interestingly, in practice we might have a predictor that is close to being a linear combination of other predictors (but not exactly), in this case to test for multicollinearity we can:    
- comparing ratio between the largest and smallest eigenvalues of the corr matrix R (the smallest eigenvalue is 0 when it's not full rank), if it's very large, then we have multicollinearity, there is no stat test to check if it's significantly greater than 0 but some empirical values are used to check for it like 500 or 1000 (arbitrary threshold); thus when ration is greater than this value, we have strong colinearity.  
- perform several Linear regressions between predictors and check the $R^2$ values, if they are close to 1. i.e., try to explain one predictor with the others, the model that shows a near 1 $R^2$ means that this predictor is very well explained by the others, it's a linear combination of them thus it's a good candidate to be removed. (e.g., explain x1 by x2 & x3, then x2 by x1 & x3, then x3 by x1 & x2, check for $R^2$ values)   
VIF (Variance Inflation Factor) is used here, as $VIF = \frac{1}{1-R^2}$, when $R^2 \approx 1$, then $VIF \approx \infty$, and when $R^2 \approx 0$, then $VIF \approx 1$ (compute a VIF for each predictor)

_But what is $R^2$ ?_

Some definitions:
- TSS: total variability in the response variable, $TSS = \sum_{i=1}^{n} (y_i - \bar y)^2$ (inherent variability in the response variable)   
- ESS: variability explained by the model, $ESS = \sum_{i=1}^{n} (\hat y_i - \bar y)^2$ (variability explained by the model), how much the model was able to explain the variability in the response variable (to get from TSS)  
- RSS: variability not explained by the model, $RSS = \sum_{i=1}^{n} (y_i - \hat y_i)^2$ (variability not explained by the model)  

- $R^2$: proportion of variability that can be explained using X, the closer to 1 the better the model is,  
proportion of variance explained by the model $R^2 = \frac{ESS}{TSS}$, where $ESS$ is the explained sum of squares and $TSS$ is the total sum of squares.   
RSS is the residual sum of squares, $RSS = \sum_{i=1}^{n} (y_i - \hat y_i)^2$, $ESS = \sum_{i=1}^{n} (\hat y_i - \bar y)^2$, and
$TSS = \sum_{i=1}^{n} (y_i - \bar y)^2$, where $\bar y$ is the mean of the response variable.  
In here $RSS$ is the error that the model makes, also seen as the projection of $y$ onto the space of $X\beta$. The smaller the distance is, the closest is our $\hat y$ to $y$, the better the model is and the more $R^2 \approx 1$  
![orthogonal projraction](./assets/Rsq-projection.png)  

- Adjusted $R^2$: penalizes the $R^2$ for the number of predictors in the model, $R^2_{adj} = 1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$, where $p$ is the number of predictors in the model. When the number of predictors increases, the $R^2$ also increases, even if the predictors are not significant. It's mainly used to compare the performance of models with different numbers of predictors (because when p increases, the variance of the model increases, which is translated in the equation by the increase of the RSS(?), so the adjusted $R^2$ will decrease)

### Maximum Likelihood Estimation

_Likelihood is not probability_  
- probability of getting this y given a particular model, pictured as an area under the curve of a fixed distribution: $Pr(data/distribution)$  
- likelihood is about fitting a model given known data, pictured as y-axis points of distributions that can be moved: $L(distribution/data)$ ([more here](https://www.youtube.com/watch?v=p3T-_LMrvBc))

![likelihood vs probability statquest note]

The maximum likelihood is an optimization problem revolving around maximizing the likelihood of the observed data given the model. "Likelihood" is a term often used in tandem with "probability", but they are not the same.  
By defintion, the likelihood is a measure of how well a statistical model explains the observed data. In here this optimization problem is about maximizing the probability of observing the data to estimate the parameters given that it follows a particular distribution, with the assumption that the data is independent and identically distributed (i.i.d).  
In other words having $f(y_i|\mu_i)$, the probability of observing $y_i$ given $\mu_i$ with $f$ being the distribution, and $\mu_i = X_i^T\beta$, the probability of observing $y_i$ given $X_i$ and $\beta$, the likelihood of observing the data is $L = \prod_{i=1}^{n} f(y_i|\mu_i)$, we want to maximize $L$ to get the $\beta$ that best explains the data.  
This product is usually hard to work with, so we take the log likelihood, $l = \sum_{i=1}^{n} log(f(y_i|\mu_i))$, and we want to maximize this quantity.   
- _gaussian distribution_, the log likelihood is $l = \sum_{i=1}^{n} log(\frac{1}{\sqrt{2\pi\sigma^2}}) - \sum_{i=1}^{n} \frac{(y_i-\mu_i)^2}{2\sigma^2}$ is equivalent the OLS problem, making this a generalized approach for different types of distributions.  
- _binomial distribution_, the log likelihood is $l = \sum_{i=1}^{n} y_i log(\mu_i) + (1-y_i)log(1-\mu_i)$, where $y_i$ is the number of successes in $n$ trials, and $\mu_i$ is the probability of success.  
- _poisson distribution_, the log likelihood is $l = \sum_{i=1}^{n} y_i log(\mu_i) - \mu_i$ and $\mu_i$ here is the $\lambda$ parameter of the poisson distribution so $\mu_i = e^{X_i^T\beta}$ and y is the number of events in a fixed interval of time.


### Link functions and loss functions: 

What distinguishes GLM from the linear model is the ___link function___ that connects the linear predictor to the expected value of the response variable.  
The linear predictor is defined as $ \eta = X\beta$, where $\beta$ is a $(p \times 1)$ column vector of coefficients. The link function is a function $g$ such that $g(\mu) = \eta$, where $\mu$ is the expected value of the response variable. 
The link function is chosen based on the distribution of the response variable. For example, in the case of the _gaussian distribution_, the link function is the _identity function_, while for the _Bernoulli distribution_, the link function is the _logit function_.

| LM | GLM |
|----|-----|
| $\mu_i = X_i^T\beta$ | $g(\mu_i) = X_i^T\beta$ |

![link function](./assets/linkfunc.png)

This link function hass to be   
- ___monotonic___ (so that the relationship between the predictors and the response variable is preserved)  
- ___differentiable___ (so that we can estimate the coefficients of the model)  
- ___invertible___ (so that we can get back to the mean from the linear predictor).

In the case of binary regression, it's the Bernoulli distribution with some probability p:  
$y_i \sim Bernoulli(p)$, $E(y_i) = \mu_i = p_i$, and we want p to be between 0 and 1, so the most common link function is the _logit function_ $logit(\mu_i) = log(\frac{\mu_i}{1-\mu_i}) = X_i^T\beta$;  where $\mu_i \in [0,1]$, $logit(\mu_i) \in \mathbb{R}$
$\iff \frac{\mu_i}{1-\mu_i} = e^{X_i^T\beta}$,   
$\iff \mu_i = \frac{e^{X_i^T\beta}}{1+e^{X_i^T\beta}} = \frac{1}{1+e^{-X_i^T\beta}}$    
$\iff \mu_i = \sigma(X_i^T\beta)$, where $\sigma$ is the sigmoid function.  

a link function is not a transformation of the response variable, only the $\mu$ is tranformed.

| Distribution | Link function name | Mathematical expression | prediction | $y \in$ |
|--------------|--------------|--|--|--|
| Gaussian     | Identity     | $g(\mu_i) = \mu_i$ | $\mu_i = X_i^T\beta$ | $\mathbb{R}$ (continuous) |
| Bernoulli    | Logit        | $g(\mu_i) = log(\frac{\mu_i}{1-\mu_i})$ | $\mu_i = \frac{1}{1+e^{-X_i^T\beta}}$ | $\{0,1\}$ (discrete) |
| Poisson      | Log          | $g(\mu_i) = log(\mu_i)$ | $\mu_i = e^{X_i^T\beta}$ |   $\mathbb{N}$ (discrete) |





Another important concept in GLM is the ___loss function___, which is a function that measures the difference between the predicted value and the actual value of the response variable. The loss function is used to estimate the coefficients of the model by minimizing the difference between the predicted value and the actual value of the response variable. It's usually chosen based on the distribution of $y$, for example, in the case of the _gaussian distribution_, the loss function is the _squared error function_, while for the _binomial distribution_, the loss function is the _log loss function_ (not going to dive deep in it, but it's the negative log likelihood of the observed data given the model).  This is seen in the MLE problem where we want to maximize the likelihood of the observed data given the model, which is equivalent to minimizing the loss function.

In each model, the coefficients are estimated by minimizing the loss function. The linear model uses _Ordinary Least Squares_ (OLS) to estimate the coefficients, while the logistic and poisson models use _Maximum Likelihood Estimation_ (MLE) to estimate the coefficients. The coefficients are estimated by maximizing the likelihood of the observed data given the model. The ***likelihood*** is a function of the coefficients and is maximized by finding the coefficients that maximize the likelihood of the observed data.   
So our goal is to get the $\beta$ that optimizes the performance of the model, i.e. minimizes the loss function or maximizes the likelihood function. (_loss function is the negative log likelihood_)

### Evaluating the model:  

An important aspect of building a GLM is to understand the statistical properties of the model, like _residuals_, _deviance_, and _confidence intervals_.  
Evaluating the performance of a model, we often care to see which predictors are significant, and which are not. This is done by performing _hypothesis tests_ on the coefficients of the model.  

We can compare between models and nested models by _anova_ which comes to compare the deviance of the models. The _deviance_ is a measure of the goodness of fit of the model and is used to compare the performance of different models. Other measures of goodness of fit include the _AIC_ and _BIC_, and even $R^2$, _odd ratios_, and _incident rate ratios_ are used to evaluate the performance of the model.

### Regularization


_Some other important concepts in statistics_  
### _Unbiased estimators:_  
The idea of bias comes from something in this example:  
If we take $\hat \mu$ (the mean) as estimator of $\mu$, for some observations we will have $\hat \mu \gt \mu$ (overestimating) and for others $\hat \mu \lt \mu$ (underestimating), but avergaing on large sets of observations we expect $\hat \mu = \mu$, so the estimator is unbiased, "unbiased estimator does not systematically over- or under-estimate the true parameter"  (ISLR). This property holds in least squares, where coefficients are unbiased estimators of the true coefficients, estimating coef on one dataset wont make our estimates exactly equal to the true values, but averaging on large sets of datasets will make them equal.

### Covariance matrix:  

This is a bit tricky everytime I read about it.  
The covariance matrix is a square matrix that contains the variances of the variables on the diagonal, and the covariances between the variables on the off-diagonal. It's a measure of how much two variables change together.  
In mathematical formulation, the covariance matrix of a random vector $X$ is defined as $Cov(X) = \mathbb{E}[(X-\mu)(X-\mu)^T]$, where $\mu$ is the mean of $X$. Let's break this down with an example:  
- $X = \begin{bmatrix} X_1 \\ X_2 \end{bmatrix}$, where $X_1$ and $X_2$ are random variables  
- $\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}$, where $\mu_1$ and $\mu_2$ are the means of $X_1$ and $X_2$  
- $Cov(X_1,X_1)=Var(X_1) =\mathbb{E}[(X_1-\mu_1)(X_1-\mu_1)]$ = $\mathbb{E}[(X_1-\mu_1)^2]$ is the variance of $X_1$ (formula of a variance from the first statistics course), by definition it's the average of the squared differences between the values of $X_1$ and the mean of $X_1$, where a large variance means that the values of $X_1$ are spread out having values far from the mean,
[probabilitycourse.com](https://www.probabilitycourse.com/chapter3/3_2_4_variance.php#:~:text=By%20definition%2C%20the%20variance%20of,values%20far%20from%20its%20mean.)  
- $Cov(X_1,X_2) = \mathbb{E}[(X_1-\mu_1)(X_2-\mu_2)]$ is the covariance between $X_1$ and $X_2$, it's a measure of how much $X_1$ and $X_2$ change together, if it's positive, then when $X_1$ is above its mean, $X_2$ is also above its mean, and if it's negative, then when $X_1$ is above its mean, $X_2$ is below its mean.  
- The cov formula above, when expanded for a matrix to include all the variables, gives the covariance matrix also denoted $Var(X)$ (ESLR) or even $\Sigma$ (ISLR).   
- $\Sigma = \begin{bmatrix} Var(X_1) & Cov(X_1,X_2) \\ Cov(X_2,X_1) & Var(X_2) \end{bmatrix}$
- Let $A$ be a Linear transformation matrix, then  
$Var(AX) = A\Sigma A^T$, this is a property of the covariance matrix, it's invariant under linear transformations (derivation:)

![cov between 2 vars]


### Probability distributions

_A refresher on some distributions_

- The _gaussian distribution_ is characterized by the probability density function $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, where $\mu$ is the mean and $\sigma^2$ is the variance. Here is a breakdown of this function because it's very nice to look at; ([source](https://youtu.be/UVvuwv-ne1I?si=z6PAEIGR1uOdMoG9)):  
    - The $e^{-\frac{1}{2}}$ term gives it its shape, it's a bell curve. The exponent is negative, so that the curve is concave down (particularly at $\mu ± \sigma$, this is why the 2 is added in denominator to scale it here instead of 2 times this value - check other elements of the exponent), and the $e$ is there to make it positive, so that the curve is always positive.  
    - The $(x-\mu)^2$ term is the exponent centered around the mean $\mu$  
    - The $\sigma^2$ term in the denominator of the exponent is the variance to make it evenly distributed, and controls the spread of the curve, it's put in the denominator because the exponent is negative, this the variance becomes directly proportional to the spread of the curve   
    - The $\frac{1}{\sqrt{2\pi\sigma^2}}$ term is the normalization constant, it ensures that the distribution is indeed a probability density function, i.e. $\int_{-\infty}^{\infty} e^{-\frac{(x-\mu)^2}{2\sigma^2}}dx = \sqrt{2\pi\sigma^2}$, and multiplying by the $\frac{1}{\sqrt{2\pi\sigma^2}}$ term ensures that the area under the curve is one ($\int_{-\infty}^{\infty} f(x)dx = 1$)


        | x        | $-\infty$ | $\mu - \sigma$ | $\mu$ | $\mu + \sigma$ | $+\infty$ |
        |----------|---------|----------------|-------|----------------|---------|
        | f(x)     | 0       | less than max  | max   | less than max  | 0       |
        | f'(x)    | +       |      +         | 0     |      -         | -       |
        | f''(x)   | +       |      0         | -     |      0         | +       |

![normal dist](./assets/gaussian.png)
- The _binomial distribution_ is characterized by the probability mass function $f(y) = \binom{n}{y}p^y(1-p)^{n-y}$, where $n$ is the number of trials, $y$ is the number of successes, and $p$ is the probability of success. The binomial distribution is used to model the _number of successes_ in a _fixed number of trials_ 
- The _poisson distribution_ is characterized by the probability mass function $f(y) = \frac{\lambda^y}{y!}e^{-\lambda}$, where $\lambda$ is the average number of events (also std. dev.) in a fixed interval of time. The poisson distribution is used to model the _number of events_ in a _fixed interval of time_

_note on density vs mass_:  
- A _probability density function_ is a function that describes the likelihood of a **continuous** random variable taking on a particular value. The area under the curve of a probability density function =1    
- A _probability mass function_ is a function that describes the likelihood of a **discrete** random variable taking on a particular value. The sum of the probabilities of all possible values of a discrete random variable =1




### Exponential Family


### Hypothesis Testing

### Multiple Testing Correction