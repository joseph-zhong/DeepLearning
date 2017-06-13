# Ch. 5: Machine Learning Basics

- Deep Learning
- 12 June 2017

## Overview

- Deep learning is a specific kind of machine learning
- Machine learning is the application of statistics with computers to approximate complicated functions
- Two common categories used to categorize ML algorithms are **supervised** and **unsupervised** learning
- **Stochastic gradient descent** is a common optimization algorithm
- To build a machine learning algorithm, we will combine optimizations, cost functions, models, and datasets

## 5.1 Learning Algorithms

A machine learning algorithm can be succinctly described as a program which 
learns from experience $E$, with respect to some class of tasks $T$ and
performance $P$

### 5.1.1 The Task $T$

Machine learning tasks are usually described in terms of how a system should
process an **example**, or a collection of **features**. 

Here are some common machine learning tasks:

- **Classification**: Predict assignment to $k$ classes for some input.
  - A classification learning algorithm usually is in the form of producing the
    function $f:R^n \to \{1, ..., k\}$ where for $y=f(x)$, the input vector $x$
    is assigned to a classifiction $y$ by the function $f$. 
  - $f$ Perhaps could also produce a probability distribution over the classes.

- **Classification with missing inputs**: Similar to above, but with lossy $x$
  input.
- **Regression**: Predict some numerical value given some input. 
  - A regression learning algorithm usually takes the form of producing the
    function $f:R^n \to R$. Similiar to classification, we are producing a
    function except, now the function produces potentially continuous values
    rather than discrete categorizations.
- **Transcription**: Transcribe some observed unstructured data into a discrete
  textual form. 
  - An example would be optical character recognition (OCR). Given some image 
    of text, return the text in the form of a sequence of characters. 
  - Another example could be speech recognition.
- **Machine Translation**: Convert a sequence of symbols in some language into a
  sequence of symbols in another language. 
- **Structured Output**: Predict a vector or data structure encoding important
  relationships with with the elements of some input. 
  - One example is parsing, or mapping a sentence into a tree encoding
    grammatical structure.
  - Pixel-wise segmentation of images is another example.
- **Anomaly detection**: Filter or flag events or objects which may be
  considered atypical. 
- **Synthesis and Sampling**: Generate new examples similar to those in training
  data. 
- **Imputation of missing values**: Predict values of missing entries $x_i$ in a
  given example of $x \in R^n$.
- **Denoising**: Predict a cleaned example $x \in R^n$ from a corrupted example
  $\tilde{x} \in R^n$. More generally, we could predict the conditional
  probability distribution $p(x|\tilde{x})$
- **Density estimation** or **Probability mass function estimation**: Learn a
  function $p_{model}: R^n \to R$ where $p_{model}$ is a probability desntiy or
  mass function on the space where examples are drawn. 
  - We can also interpret this task as a clustering problem, where we wish to
    know the distribution density across our domain space. 

This list of tasks is intended to provide examples, and not to provide a rigid
taxonomy of tasks.

### 5.1.2 The Performance Measure $P$

...

### 5.1.3 The Experience $E$

### 5.1.4 Example: Linear Regression


## 5.2 Capacity, Overfitting and Underfitting

### 5.2.1 The No Free Lunch Theorem

### 5.2.2 Regularization


## 5.3 Hyperparameters and Validation Sets

### 5.3.1 Cross-Validation


## 5.4 Estimators, Bias and Variance

### 5.4.1 Point Estimation

Point estimation is the attempt to provide the single "best" prediction of some
quantity of interest. Let us denote a point estimate of a parameter $\theta$ by
$\hat{\theta}$

A **point estimator** is any function of the data 

$$\hat{\theta_m} = g(x^{(1)}, ..., x^{(m))}$$

#### Function Estimation

In function estimation, we are interested in approximating $f$, or $\hat{f}$. 

### 5.4.2 Bias

The bias of an estimator is defined as the following:

$$\text{bias}(\hat{\theta_m}) = \mathcal{E}(\hat{\theta_m}) - \theta$$

- $\hat{\theta_m}$ is the data from samples, and $\theta$ is the true value. 
- An estimator $\hat{\theta_m}$ is said to be **asymptotically unbiased** if 

$$\lim_{m \to \infty} \text{bias}(\hat{\theta}_m) = 0$$

#### Example: Bernoulli Distribution

Recall the distribution for a Bernoulli of mean $\theta$:

$$P(x^{(i)}; \theta) = \theta^{x^{(i)} (1-\theta)^{(1-x^{(i)})}$$

Where the mean can be estimated as the following:

$$\hat{\theta_m} = \frac{1}{m}\sum_{i=1}^mx^{(i)}$$

It can be shown that this estimator is unbiased.





