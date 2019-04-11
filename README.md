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
