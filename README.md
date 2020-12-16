# sklearn-neural-network-tryout
Simple tryout with sklearn to train a neural network model with UCI Adult dataset


### Training process:
 * Load data
 * Correlation analysis
 * Drop the columns according to the correlation results ['fnlwgt', 'native-country', 'workclass', 'education', 'occupation']
 * Data clean
 * Map features to numerical value
 * Split the training set and test set(20% test set)
 * Subsample the imbalanced data through under-sampling(both training and test, could only do with the training set)
 * Normalization
 * Train the model using MLPClassifier, 9-feature neural network
 * Evaluation
 
 ### Sample output
----------------------------------------
* Train dataset: (26048, 9)(26048,)
* Test dataset: (6513, 9)(6513,)
* Balanced Train dataset: (12406, 9)(12406,)
* Balanced Test dataset: (3276, 9)(3276,)
----------------------------------------
* Accuracy score:
0.8235653235653235
----------------------------------------
* Confusion Matrix:
[[1395  243]
 [ 335 1303]]
----------------------------------------
* Classification Matrix:
```
                precision    recall  f1-score   support

           0       0.81      0.85      0.83      1638
           1       0.84      0.80      0.82      1638

    accuracy                           0.82      3276
   macro avg       0.82      0.82      0.82      3276
weighted avg       0.82      0.82      0.82      3276
```
