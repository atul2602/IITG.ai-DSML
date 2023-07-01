## Ensemble Learning
- Multiple models for a single result
- Potentially averages out high bias and variance
- Individual models called _weak learners_, combine to form a _strong learner_
- Homogenous (same model, different data)
  - Bagging and Boosting
- Heterogenous (different models)
  - Stacking

### Types of Ensemble Learning
#### Voting
  - Different learning algorithms of weak learners
  - Majority count/mean is the prediction
#### Bagging
  - Stands for "Bootstrapped Aggregation"
  - Same models get different data
  - Data is randomly picked and **replaced** in the set
  - Eg: Random Forest
#### Boosting
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
