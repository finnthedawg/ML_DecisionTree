# ML_DecisionTree
**1. Implementation of classification tree using numpy**

**2. Implementation of regression tree using numpy**

**3. Predicting flower types using petal lengths**

**2. Classifying spam and non-spam emails**

**2. Regressing housing prices in Boston suburbs**

---

Build instructions:

```
git clone
jupyter notebook DecisionTree.ipynb
```

## Classification task: Iris flower dataset (iris.csv)

The Iris flower dataset is a dataset with 4 input features that classifies the flower into either `Iris-virginica`, ` Iris-setosa` or `Iris-versicolor`. This dataset was loaded using my `loadData()` function and the first four features were converted to floats.

The classification tree's splits are determined by maximum entropy gain of each split. For categorical values, we test the information gain for splitting on each category and for continuous variables, for performance reasons, we test a split over the mean. The values of leaf nodes are set as the modal class of it's members.


We create the classification tree with `n_min = 5` which is a minimum of `5%` of the dataset for each leaf node and train it on the data.

```
tree = classificationTree(n_min = 5) #n_min = minimum percentage of data as leafnode size.
tree.fit(data,label)
```

To predict a single value:
```
tree.predict([6.4, 2.8, 5.7, 2.2])
>>> 'Iris-virginica'
```
### Accuracy evaluation of min node size using K-fold

In order to determine the best leaf size to use, we train the data on 10 k-folds at various different n_min values `[5,10,15,20]`. The results were as following:

```
#Std is of the 10 errors in each fold.
Avg accuracy over 10 folds for n_min 5 : 93.33333333333333
Avg std5 : 7.888106377466154
Avg accuracy over 10 folds for n_min 10 : 90.66666666666667
Avg std10 : 8.537498983243799
Avg accuracy over 10 folds for n_min 15 : 92.66666666666666
Avg std15 : 7.571877794400365
Avg accuracy over 10 folds for n_min 20 : 92.66666666666666
Avg std20 : 7.571877794400365
```

The best performing model was the lowest n_min percentage of 5% (Since the tree can be more granular) and it did not appear to overfit too much to the training fold and accurately predicted the test.

### Confusion matrix

Based on the best `n_min value` (5), I created a class confusion matrix using ten-fold cross-validation utilizing concatenation of the matricies on the test portions.

```
[[46  0  4]
 [ 0 50  0]
 [ 1  6 43]]
```

This confusion matrix can be understood as:

For Iris-virginica my model predicted 46 as Iris-virginica, 0 as Iris-setosa and 4 as Iris-versicolor.

For Iris-setosa my model predicted 0 as Iris-virginica, 50 as Iris-setosa and 0 as Iris-versicolor.

For Iris-versicolor my model predicted 1 as Iris-virginica, 6 as Iris-setosa and 43 as Iris-versicolor


## Classification task: Spam email database (Spambase.csv)

The spam dataset is a dataset with 4600 emails that contain many input features and a binary classification of either 1 or 0. Similar to the iris dataset I used my classification tree with a n_min of 5 initially.

Similar to the iris task, I created the classification tree with `n_min = 5`:

```
tree = classificationTree(n_min = 5) #n_min = minimum percentage of data as leafnode size.
tree.fit(data,label)
```

### Accuracy evaluation of min node size using K-fold

Similar to the iris dataset, to determine the best leaf size to train the model through 10 k-folds testing at various different n_min values `[5,10,15,20,25]`. The results were as following:

```
% Avg accuracy for n_min 5 : 89.93633877204564
% Avg std of accuracy for n_min 5 : 1.8548571262148763
% Avg accuracy for n_min 10 : 89.41469395454116
% Avg std of accuracy for n_min 10 : 1.7743518319557736
% Avg accuracy for n_min 15 : 85.80708290106574
% Avg std of accuracy for n_min 15 : 1.892617138682749
% Avg accuracy for n_min 20 : 85.80708290106574
% Avg std of accuracy for n_min 20 : 1.892617138682749
% Avg accuracy for n_min 25 : 85.26384042252192
% Avg std of accuracy for n_min 25 : 1.9161643747826416
```

Our best model was a n_min size of `5%` with a standard deviation of accuracy over the 10 folds to be 1.85

##  Regression task: predicting the value of housing prices in Boston suburbs (housing.csv)

The housing dataset is a dataset with 13 features (e.g NoX concentration, number of parks) and a value attached to each value (in thousands) of the value of a house. As the task is to predict the value of a house, I implemented and used a regression tree.

For the regression tree, the splits are decided by maximum variance rather than entropy and the value of a node is set as the mean value of it's constituent members.

We create the classification tree with `n_min = 5` which is a minimum of `5%` of the dataset for each leaf node and train it on the data.

```
tree = regressionTree(n_min = 5) #n_min = minimum percentage of data as leafnode size.
tree.fit(data,label) #Train the tree with our data and labels of the correct values
```

To predict a single value:

```
tree.predict([0.02985, 0, 2.18, 0, 0.458, 6.43, 58.7, 6.0622, 3, 222, 18.7, 394.12, 5.21])
>>> 25.550000000000004 #In thousands.
```

### Accuracy evaluation of min node size using K-fold

To evaluate the model over various different `n_min` values `[5, 10, 15, 20]`, we run the regression tree over 10 K-means. The error is the mean squared error `MSE`.

Results:

```
Avg mean squared error for n_min 5 : 37.3578768627451
Avg std for this MSE for n_min5 : 14.970789548411656
Avg mean squared error for n_min 10 : 35.27211254901962
Avg std for this MSE for n_min10 : 9.300242901179509
Avg mean squared error for n_min 15 : 37.48329921568628
Avg std for this MSE for n_min15 : 8.966479016797244
Avg mean squared error for n_min 20 : 38.86960627450981
Avg std for this MSE for n_min20 : 8.09540401947604
Avg mean squared error for n_min 25 : 95.98639058823528
Avg std for this MSE for n_min25 : 31.308222840450757
```

It appears that MSE increases slightly as n_min increases. However, past a certain n_min, the MSE dramatically increases and the algorithm beings to perform poorly.

This is because when the n_min is low, the leaf sizes are small and numerous and the model overfits the data leading to a lower accuracy over the kfolds. When the n_min is too high, the leaf sizes are too big and the model underfits the data and also performs poorly.
