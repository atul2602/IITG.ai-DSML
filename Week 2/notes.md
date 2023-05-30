## Linear Regression
1. Starts off with all Andrew N.G.'s lectures.
2. Code Basics : [Popular Models on GitHub](https://github.com/codebasics/py/tree/master/ML)
3. [From scratch notebook](https://www.kaggle.com/code/fareselmenshawii/linear-regression-from-scratch)
4. [From scratch video](https://www.youtube.com/watch?v=VmbA0pi2cRQ)
5. Did two codewalks by codebasics - sklearn

## Logistic Regression
1. Andrew's lectures
2. Did two codewalks by codebasics
3. [from scratch video](https://www.youtube.com/watch?v=nzNp05AyBM8)

---

## Regularisation
1. Andrew's Lectures
2. codewalks by codebasics

### L1 and L2 Regularisation

- Idea of regularisation : _Prevent overfitting_ by preventing extreme values of model parameter.
- Reduces variance at expense of bias
- Bring down weights to a particular value $\tilde{w}$
#### L1 Regularisation
$$ J(w) = L(w) + \sum_{i=1}^{N}{|{w_i}|}$$
<p align = "center">
<img width="416" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/675813e2-c08b-4c09-9f14-558a9f151067">
</p>

- L1 regularisation brings sparsity to the solution : some weights may go to zero
- Used as a feature selection method: auto. rules out unnecessary features
- For direction with lower slope, large effect of regularisation and viceversa

#### L2 Regularisation
$$ J(w) = L(w) + \sum_{i=1}^{N}{|{w_i}^2|}$$
<p align = "center">
<img width="482" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/d58395d2-08fb-42da-818b-ca6fd659404e">
</p>

- Optimal value of J lies at a point where contours of L(w) and regularisation term _touch_ each other.
- L2 regularisation prevents explosion of parameters to large values
- Brings points closer to origin by larger amount at which slope of contour is minimum. 

---

## Naive-Bayes Algorithm
[scikit explanation](https://scikit-learn.org/stable/modules/naive_bayes.html)
### Multinomial Naive-Bayes 
[StatQuest](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
- Discrete Features
- Take example of spam classification:
    - Store the words with their conditional probability distributions : $P( Word | Spam ) , P( Word | Not Spam)$
    - Define **Prior Probability** : Initial guess that message is normal, using training data
    - Use Bayes Theorem
        - $P(N) * P(Dear | N) * P(Friend | N) \propto P(N | Dear Friend)$
        - $P(S) * P(Dear | S) * P(Friend | S) \propto P(S | Dear Friend)$
    - To handle zero probabilities, add `1` to each histogram (frequency of word)
- Why Naive? Doesnt consider `order` of words
- High bias, low variance algorithm

### Gaussian Naive-Bayes
[StatQuest](https://www.youtube.com/watch?v=H3EjCKtlVog)
- Continuous features
- Take binary classification of a person liking a movie or not based on popcorn/Soda/candy habits:
    - Store mean and variance for each of the features among the two classes, and create **Gaussian Distributions**
    - Prior Probability same as above
    - Use Bayes Theorem, and data from Gaussian distributions
        - $P(Likes) * P(Popcorn = x_1 | Likes) * P(Soda = x_2 | Likes) * P(Candy = x_3 | Likes) \propto P(Likes | x)$
        - $P(No Likes) * P(Popcorn = y_1 | Likes) * P(Soda = y_2 | No Likes) * P(Candy = y_3 | No Likes) \propto P(No Likes | y)$
    - Read Cross Validation for best features 

---
## Evaluation Metrics
- Need? Single residuals fail to provide an overall picture.
### Regression
#### Mean Absolute Error (MAE)

$$MAE=\frac{1}{N}\cdot\sum_{i=1}^{N} |{y_i - \hat{y_i}}|$$
- Fails to punish big errors
- MAE is not differentiable, so need additional optimisers
- Robust to outliers
```python
from sklearn.metric import mean_absolute_error
mean_absolute_error(y_predicted, y_test)
```

#### Mean Squared Error (MSE)

$$MSE=\frac{1}{N}\cdot\sum_{i=1}^{N} ({y_i - \hat{y_i}})^{2}$$
- Not robust to outliers, catches them easily
- Overestimates weakness (>1) and understimates (when <1)
```python
from sklearn.metric import mean_squared_error
mean_squared_error(y_predicted, y_test)
```

#### Root Mean Squared Error (MSE)

$$RMSE=\frac{1}{N}\cdot \sqrt{\sum_{i=1}^{N} ({y_i - \hat{y_i}})^{2}}$$
- Most common metric
```python
from sklearn.metric import mean_squared_error
mean_squared_error(y_predicted, y_test, squared = False)
```
#### Max Error

$$ME=\max_{i=1}^{N} |{y_i - \hat{y_i}}|$$
```python
from sklearn.metric import max_error
max_error(y_predicted, y_test)
```

#### R-squared Score, coefficient of determination
- Measures the proportion of variance of the dependent variable explained by the regression model
- When we add new features (even irrelevant), then R2 score increases
- R2 = 1, means good model
$$R^2 = 1 - \frac{SSE}{SST}$$
$$SSE =  \sum_{i=1}^{N} ({y_i - \hat{y_i}})^{2}$$
$$SST =  \sum_{i=1}^{N} ({y_i - \bar{y}})^{2}$$
```python
from sklearn.metric import r2_score
r2_score(y_predicted, y_test)
```

#### Adjusted R-squared
- Penalises adding more independent variables
$$R_{adj}^{2} = \left(\frac{(1-R^2)(n-1)}{n-k-1} \right)$$
where, n = no. of data points, k = no. of indep features

### Classification

#### 






