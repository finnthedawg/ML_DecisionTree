# ML_DecisionTree
**1. Implementation of classification tree using numpy**

**2. Implementation of regression tree using numpy**

**3. Predicting flower types using petal lengths**

**2. Classifying spam and non-spam emails**

**2. Regressing housing prices in Boston suburbs**
## Iris flower dataset.

The Iris flower dataset is a dataset with 4 input features that classifies the flower into either `Iris-virginica`, ` Iris-setosa` or `Iris-versicolor`. This dataset was loaded using my `loadData()` function and the first four features were converted to floats.

Then we create the classification tree with `n_min = 5` which is a minimum of `5%` of the dataset for each leaf node and train it on the data.

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

In order to determine the best leaf size to use, we train the data on 10 k-folds at various different n_min values. The results were as following:

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

 
