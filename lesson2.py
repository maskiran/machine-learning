"""
Continuation of lesson1 to visualize the decision tree
https://www.youtube.com/watch?v=tNa99PG8hR8&index=2&list=PLT6elRN3Aer7ncFlaCz8Zz-4B5cnsrOMt
"""

# as discussed there are many kinds of classifiers. DecisionTree
# is the easiest to visualize and interpret and understand the
# the if-else condition that the classifier came up with (learnt)
# we can literally see the tree.

# we will use some real data thats available online on wikipedia
# (which is available on scikit) - the iris flower data
# Give a sepal length and width, petal length and width (4 features)
# figure out the kind of iris (setosa, versicolor, virginica)
# there are 150 examples with features and labels provided

import sklearn.datasets
iris = sklearn.datasets.load_iris()

# iris.feature_names => gives all the feature names of the data
# iris.target_names => give all the ENUM target_values
# iris.data => all the example data
# iris.target => all the known labels from the examples (these are enum
# values that are mapped in iris.target_names)
print("First data example")
print(iris.feature_names)
print(iris.data[0]) # prints the first data example
print("First target value and its enum name")
print(iris.target[0], iris.target_names[iris.target[0]]) # first known label

# once the data is obtained, lets take a subset of this to train the model,
# then use the model to predict the other subset and see how accurate the
# prediction is. we already know the real label and can compare with
# with the predicted value. this increases our confidence of the model
# prediction.
# In the example data, its ordered by the kind of iris flower
# so the first 50 are setosa, 51-100 are versicolor and 101-150 virginica
# lets remove the 0, 50 and 100 (the first of each kind) and put them
# as test data.
test_idx = [0, 50, 100]

# numpy makes is very easy to deal with rows of data
import numpy

train_data = numpy.delete(iris.data, test_idx, axis=0)
train_target = numpy.delete(iris.target, test_idx, axis=0)

test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

# lets use the decisiontree classifier like earlier
print("Training the model")
import sklearn.tree
clf = sklearn.tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
print("Training done")

# training is done
# we will now use the trained classifier on our test data and see how accurate it is
# lets print the know test target and then print the predicted target and compare
print('The known test target')
print(test_target)
print('Predicting')
print(clf.predict(test_data))

# for this simple data set, mostly the prediction and test would succeed
# However for complex models, the prediction need not match the test data
# Dont try to get the model re-trained to get 100% of test data to match
# This is called overfitting and the goal is not get all the known data
# correct. Even if it gets the know examples correctly, it might fail
# in the future once unknown data shows up.

# Lets visualize the decision tree
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
sklearn.tree.export_graphviz(clf,
                             out_file=dot_data,
                             feature_names=iris.feature_names,
                             class_names=iris.target_names,
                             filled=True,
                             rounded=True,
                             impurity=False
                            )
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf('iris-decision-tree.pdf')