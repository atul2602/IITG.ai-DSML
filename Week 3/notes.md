# Support Vector Machines
- Supervised
> StatQuest 

- **Margin** : Shortest distance between observations and threshold (classified completely)
- Threshold that gives _maximum margin_ is called **Maximal Margin Classifier**
    - Super-sensitive to outliers
- _Solution_ : Allow misclassification (increase bias, decrease variance)
- In this case, margin is called **soft margin**
    - Determine using Cross Validation
- Threshold is called **Support Vector Classifier**  
    - Observations on edge, within soft margin are called Support Vectors
<p align = "center">
<img width="471" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/04ef30a9-0397-40ef-b5ce-e45ba3614220">
</p> 

- Above methods work to create a _hyperplane_, what about this??
<img width="484" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/b47e604e-6a6d-4c38-ad17-ba70599023df"> 

- **Support Vector Machines**
    - Start with data in relatively low dimension
    - Move the data into higher dimension
    - Find SVC that separates data into 2 groups
<p align = "center">
<img width="412" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/98fe870c-28b6-41c7-93e8-d36558d01dd6">
</p> 

- **Kernel Functions** : Used to systematically find SVC in higher dimensions
    - Polynomial(d : degree) : Use CV to get d
    - Radial : idea is to classify based on nearest observations in training data 
- _Kernel Trick_ : the kernel function doesnt physically convert data to higher dimensions, but uses pairwise relations

### Polynomial Kernel
<p align = "center">
$$(a \cdot b + r)^{d}$$ expands to $$(a\sqrt{r}, a^2, r).(b\sqrt{r}, b^2, r)$$ where a and b are two different data points and $d=2$
</p>  

- This expression helps to calculate pair-wise relationships in higher dimensions
- `r` and `d` determined using cross validation

### Radial Kernel
- $$e^{-\gamma(a-b)^{2}}$$
- Put `r=0` in polynomial kernel, then data remains in 2-D
- Add multiple `r=0` kernel to get multidimensional data
- expand $e^{a \cdot b}$ using Taylor's to see infinite dimensional relation

---
[Playlist](https://www.youtube.com/playlist?list=PLKnIA16_RmvbOIFee-ra7U6jR2oIbCZBL)
- SVM is an improved version of Logistic Regression
    - Classify as _widely_ as possible
    - Maximise margin d
    - Touch points are called support vectors
  <img width="250" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/6f570b0e-23d4-40b3-b07f-45dc14217d71">
  
#### Hard Margin SVM  
  - Hyperplane $\vec{w} \cdot \vec{x} + w_{0} = 0$, $\vec{w}$ is normal to the hyperplane
  - **Decision Rule** : $\vec{w} \cdot \vec{v} + b \ge 0$
  <img width="262" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/58b6bd3d-5e32-47eb-bda7-bcf736eed55d">  

  - **Aim** : Maximise distance between $\pi_{+} (\vec{w} \cdot \vec{x} + b = 1)$ and $\pi_{-} (\vec{w} \cdot \vec{x} + b = -1)$ 
      - Assuming $Y_{i} \cdot (\vec{w} \cdot \vec{x_{i}} + b) \ge 1$ 
      - Now, distance between $x_1$ on $\pi_{+}$ and $x_2$ on $\pi_{-}$ is $(\vec{x_2} - \vec{x_1}) \cdot \frac{\vec{w}}{|w|}$, simplifies to $\frac{2}{|w|}$

#### Soft Margin SVM 

$$\min_{(w,b)} \frac{|w|}{2} + C \cdot \sum_{i=1}^{n} \zeta_{i}$$  

$$\zeta_{i} = Max(0, d(x_i , Support Plane))$$

- First term is `margin error` or regularisation loss and second term is `classification error` or hinge loss
- `C` is a hyperparameter

#### Kernel Trick
<img width="266" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/87cfe001-0736-4064-ad6d-8c7c0a307fc9"> 

- Implementation
- New Kernel : Gaussian

#### Code
- Read for scratch implementation : [Kaggle](https://www.kaggle.com/code/fareselmenshawii/svm-from-scratch)

---
# K-Nearest Neighbours
> Supervised
- Use K nearest neighbours in plot (example : PCA), assign category with most votes.
- [Scratch Implementation](https://blog.devgenius.io/implementing-k-nearest-neighbors-from-scratch-in-python-d5eaaf558d49)
- [Elaborate Scratch Kaggle](https://www.kaggle.com/code/fareselmenshawii/knn-from-scratch)
---
# Unsupervised Learning
> They group things together, not good at predictions
- Pattern recognition
### Clustering  

- Exclusive(hard clustering) : K-Means
    - Image segmentation, market segmentation, image compression 
- Overlapping(soft clustering) : Fuzzy K-Means
- Hierarchial : Make clusters from clusteres
    - Agglomerative
    - Divisive
- Probabilistic : determine which probability distribution a point belongs to
    - Gaussian Mixture Models
    - Expectiation maximisation

### Association Rules
- Looks for relationship between variables
- Eg. market basket analysis: which all items are bought together often?

### Dimensionality Reduction
- PCA : Linear transformation on data
    - PC1: Maximises variance of the dataset
    - PC2: Maximises variance, orthogonal to PC1 
- SVD : Matrix Factorisation
- Autoencoders

### Applications
- Google News categorisation
- Computer Vision: Object detection
- Medical imaging: Image segmentation, detection, classification
- Anomaly detection ?
- Create customer personas ?
- Recommender systems

### Challenges
- High complexity algorithms, longer training times
- Lack of transparency on clustering criteria
---
# K-Means Clustering
> Unsupervised
- Basic Algorithm
    - Select K
    2. Randomly select K data points
    3. Cluster all other points based on nearest cluster
    - Change all centroids to means of clusters
    - Repeat step iii. till clustering stablises
    - Repeat ii, and choose clustering with best variance (WCSS)
        - Euclidean Distance
        - Manhattan Distance 
- Choosing K
    - Plot _Reduction in Variation_ vs _K_, and choose elbow point
- [Scratch implementation](https://www.kaggle.com/code/fareselmenshawii/kmeans-from-scratch)
 
