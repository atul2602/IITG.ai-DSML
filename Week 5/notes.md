## Ensemble Learning

> campusX, [Blog](https://duchesnay.github.io/pystatsml/machine_learning/ensemble_learning.html)

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
  - Enables intensive parallelisation
  - Eg: Random Forest
#### Boosting
  - A sequential technique, usually preffered for low variance, high bias weak learners
  - On a high level, each model **learns the mistakes** of previous one: so miscalculated datapoints of first models are sent to next one for training
  - Examples:
    - AdaBoost
    - GradientBoost
    - XGBoost
#### Stacking
  - Advanced voting: a separate model combines the outputs of weak learners


### Trade-offs
#### Pros
- Improved performance (potentially)
- Low bias and variance
#### Cons
- Computation
