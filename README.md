# Credit_Risk_Analysis
## Overview
The purpose of this analysis is to evaluate credit card risk data, and determine which algorithm utilized is the best fit for the dataset. This was done by using resampling methods, as well as classifiers. The dataset was provided by LendingClub.

## Resources
* Data Source: LoanStats_2019Q1.csv
* Software: Jupyter Notebook 6.4.8

## Results
To conduct this analysis, we utilized six machine learning models:
* Random Oversampling
* SMOTE Oversampling
* Cluster Centroids Undersampling
* SMOTEENN Combination Sampling
* Balanced Random Forest Classifier
* Easy Ensemble AdaBoost Classifier

To determine how efficient these models were, we pulled the balanced accuracy score, as well as the precision and recall.
* The balanced accuracy score tells us how good the classifier is for the dataset. The closer the number is to 1 the better.
* Precision is a measure of how reliable a positive classification is.
* Recall, or sensitivity, is the ability of the classifier to find all of the positive samples.

In assessing the results, our majority class is the low_risk, whereas our minority class is the high_risk.

### Random Oversampling
We used the RandomOverSampler from the imblearn library on the dataset. In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

![1_balanced_accuracy](https://user-images.githubusercontent.com/106129195/194686472-e70fd987-bf00-4adf-a1ae-c3280a4c0405.png)

* The balanced accuracy score is 0.65.

![1_classification_report](https://user-images.githubusercontent.com/106129195/194686670-1928d8ae-2a4c-4c45-8270-737c6eb353ab.png)

* Precision for the minority class is very low (0.01), and very high for the majority class (1.00).
* Recall for the minority class is high (0.71), and a little lower for the majority class (0.58).

### SMOTE Oversampling
We used SMOTE (Synthetic Minority Oversampling Technique) for our second algorithm. In using SMOTE, the size of the minority class is increased. However, the new instances are interpolated. For an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, the new values are created. The results were as follows:

![2_balanced_accuracy](https://user-images.githubusercontent.com/106129195/194687048-7987e32e-b1d5-4d25-8fe0-85e70f919b00.png)

* The balanced accuracy score is 0.66.

![2_classification_report](https://user-images.githubusercontent.com/106129195/194687072-63ecd631-1711-4998-a939-83b016fbe236.png)

* Precision for the minority class is very low (0.01), and very high for the majority class (1.00).
* Recall for the minority class is high (0.63), and a little higher for the majority class (0.68).

### Cluster Centroids Undersampling
We then turned to an undersampling model, Cluster Centroids, and assessed how well the model worked with this dataset. This algorithm identifies clusters of the majority class, then generates synthetic data points (centroids) that are representative of the clusters. Then the majority class is undersampled down to the size of the minority class.

![3_balanced_accuracy](https://user-images.githubusercontent.com/106129195/194687198-274cd3cc-a80f-423b-93ad-a46b1a053a78.png)

* The balanced accuracy score is 0.54.

![3_classification_report](https://user-images.githubusercontent.com/106129195/194687226-81e97557-3011-4600-96cc-20e3f86a7b2e.png)

* Precision for the minority class is very low (0.01), and very high for the majority class (1.00).
* Recall for the minority class is high (0.69), and lower for the majority class (0.40).

### SMOTEENN Combination Sampling
We also used the SMOTEENN model (Synthetic Minority Oversampling Technique and Edited Nearest Neighbors) and assessed its proficiency with the dataset. SMOTEENN oversamples the minority class using SMOTE, then cleans the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, then that data point is dropped.

![4_balanced_accuracy](https://user-images.githubusercontent.com/106129195/194687468-df7491fe-c57b-4dc5-871d-e46e878151ab.png)

* The balanced accuracy score is 0.67.

![4_classification_report](https://user-images.githubusercontent.com/106129195/194687489-e870256d-6afe-4e97-b487-62b97914ef60.png)

* Precision for the minority class is very low (0.01), and very high for the majority class (1.00).
* Recall for the minority class is high (0.73), and lower for the majority class (0.60).

### Balanced Random Forest Classifier
The Balanced Random Forest Classifier randomly under-samples each bootstrap sample to balance it. Just as with the previous models, we assessed how well this model works with the dataset.

![5_balanced_accuracy](https://user-images.githubusercontent.com/106129195/194687777-7c9168fe-f7e4-4086-b948-0ae8a0d95532.png)

* The balanced accuracy score is 0.79.

![5_classification_report](https://user-images.githubusercontent.com/106129195/194687801-199e89d2-b36a-4e7c-8c3f-d2d7866f88d6.png)

* Precision for the minority class is very low (0.03), and very high for the majority class (1.00).
* Recall for the minority class is high (0.70), and a little higher for the majority class (0.87).

### Easy Ensemble AdaBoost Classifier
The last model we utilized was the Easy Ensemble AdaBoost Classifier. The classifier is an ensemble of AdaBoost learners that trained on different balanced bootstrap samples. The balancing is achieved by random under-sampling.

![6_balanced_accuracy](https://user-images.githubusercontent.com/106129195/194687882-61878716-8c4e-4fd6-9b5c-7e15181e826d.png)

* The balanced accuracy score is 0.93.

![6_classification_report](https://user-images.githubusercontent.com/106129195/194687920-60b57855-1443-4324-91bb-df52365b9486.png)

* Precision for the minority class is low (0.09), and very high for the majority class (1.00).
* Recall for the minority class is high (0.92), and also high for the majority class (0.94).

## Summary
Based on the results above, we can see that the resampling models (Random Oversampling, SMOTE Oversampling, Cluster Centroids Undersampling, and SMOTEENN Combination Sampling) performed differently than the classifiers (Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier). As such, the recommended model to use for this dataset would be the SMOTEENN model. The difference in the recall between the two classes isn't large, and the balanced accuracy score is close to 0.70. As we would like a balanced model, the SMOTEENN model would be the best fit.
