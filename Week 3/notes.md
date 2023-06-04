## Support Vector Machines
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
---
[Playlist](https://www.youtube.com/playlist?list=PLKnIA16_RmvbOIFee-ra7U6jR2oIbCZBL)
- SVM is an improved version of Logistic Regression
    - Classify as _widely_ as possible
    - Maximise margin d
    - Touch points are called support vectors
  <p align = "center">
  <img width="182" alt="image" src="https://github.com/atul2602/IITG.ai-DSML/assets/61497490/6f570b0e-23d4-40b3-b07f-45dc14217d71">
  </p>
