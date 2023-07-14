## Ensemble Learning

> campusX, [DuchesNay Blog](https://duchesnay.github.io/pystatsml/machine_learning/ensemble_learning.html), [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/)

- Multiple models for a single result
- Potentially averages out high bias and variance
- Individual models called _weak learners_, combine to form a _strong learner_
- Homogenous (same model, different data)
  - Bagging (decrease variance) and Boosting (decrease bias)
- Heterogenous (different models)
  - Stacking (improve prediction)

### Types of Ensemble Learning
#### Voting
  - Different learning algorithms of weak learners
  - Majority count/mean is the prediction
#### Bagging
  - Stands for "Bootstrapped Aggregation"
  - Same models get different data
  - Data is randomly picked and **replaced** in the set
  - Voting:
    - Soft (average of probabilities)
    - Hard (majority class)
  - Enables intensive parallelization
  - Eg: Random Forest

    [Basic Code](https://github.com/campusx-official/bagging-ensemble)  
  - Types: Pasting (sampling w/o replacement), Random Subspaces (column sampling), Random Patching (both row, column sampling)
#### Boosting
  - A sequential technique, usually preferred for low variance, high bias weak learners
  - On a high level, each model **learns the mistakes** of the previous one: so miscalculated data points of first models are sent to the next one for training
  - Examples:
    - AdaBoost
    - GradientBoost
    - XGBoost
#### Stacking
  - Advanced voting: a separate model combines the outputs of weak learners

---
### AdaBoost
We want to solve optimisation problem of finding _optimal weights_ to combine weak learners.  

$$s_L(.) = {\sum}_{k=1}^n c_l*w_l(.)$$

Initially, all models have a weight of $\frac{1}{L}$.
Now, at each step, we choose the best $c_l$ and $w_l$ such that minimise fitting error for the strong learner...

$$s_l(.) = s_{l-1}(.) + c_l*w_l(.)$$

Also, at each step, it updates the **observations weights** for samples wrongly predicted by current ensemble.

Note: Variants exist as per _loss function_ such as LogitsBoost, L2Boost  

### GradientBoost
Optimisation problem is the same, way of iterating is different.
- We calculate _negative of gradient of fitting error of ensemble at each step_
- Find the best weak model, and fit it to the above _pseudo-residuals_
- Find the best coefficient and add the weak learner.

I find this weirdly similar to Gradient Descent for ensemble model.

Note: XGBoost is more sophisticated. LightGBM (by Microsoft) is faster and equal better as XGBoost.

---
### Random Forest
- [StatQuest](https://youtu.be/sQ870aTKqiM) explains how we can use Random Forests to find missing values in a table, and obtain **distance matrix**.
- [Difference](https://github.com/campusx-official/100-days-of-machine-learning/blob/main/day65-random-forest/bagging_vs_random_forest.ipynb) from Bagging(with DTs) is that for splitting every node, random sampling is done. More randomness => Better results.

### More on Boosting
- [CodeEmporium](https://www.youtube.com/watch?v=MIPkK5ZAsms&t=544s) defines "strong learner" as a **P**roabably **A**pproximately **C**orrect(PAC) model.
    - Weak Learners are models that perform better than guessing.
    - Researchers proposed that "a collection of weak learners" can perform as good as a strong learner(which are complex DL models in real-life)
- Gradient boosting is similar to AdaBoost, just that everything is done in terms of gradients.
---
### Trade-offs
#### Pros
- Improved performance (potentially)
- Low bias and variance
#### Cons
- Computation
