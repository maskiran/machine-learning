"""
When training a classifier, its always a challenge to figure out if the
classifier can predict correctly for the new data. To make this easier
partition the available training data into 2 sets - train and test.
https://www.youtube.com/watch?v=84gqSbLcBFE&list=PLT6elRN3Aer7ncFlaCz8Zz-4B5cnsrOMt&index=5
"""

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

# this is similar to f(x) = y, given x (features) predict y

# now that we got this data, we will partition this to train and test
# from sklearn.cross_validation import train_test_split (deprecated)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# will train using X_train and y_train and test using X_test, y_test
# test_size = 0.5 implies, we want 0.5 times the data for test

# Lets train 2 classifiers and see how it works.

# Start with decision tree
from sklearn import tree
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
predictions = tree_clf.predict(X_test)

# we already know the real y labels for the X_test, so lets check how
# accurate the predictions are.

from sklearn.metrics import accuracy_score
print("The accuracy score with Decision Tree is ", end="")
print(accuracy_score(y_test, predictions))

# we can use a different classifier and see how it predicts
# Lets use KNN - K-Nearest Neighbour
# This measures the distance of the value with the trained values
# Whatever is the closest, the label of that example is predicted
# as the label of this new data.
# It can also use the K nearest neighbours instead of 1. So the label
# of its K closest neighbours is the label of the new data.

from sklearn import neighbors
knn_clf = neighbors.KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
predictions = knn_clf.predict(X_test)
print("The accuracy score with KNN is ", end="")
print(accuracy_score(y_test, predictions))

# There are many other classifiers than these 2. However the code is
# same. Create an instance of a classifier, fit and predict

# what is learning in machine learning
# A model is assumed - a straing line, curve, sine-curve, polynomial equation
# etc.
# Once the model is assumed, the parameters for the model need to be found out
# If the model is a straight line y = mx + c
# then the parameters m and c must be figured out as part of learning
# Take some random values for m and c. Take the a data point from training set
# and predict value. If the value is good, our initial guess for the
# parameters shows good for one example. Continue with the next example.
# If the predicted value is wrong during the course, then re-adjust the
# parameters (read about gradient descent, on how params can be adjusted)
# Can be played at tensorflow playground playground.tensorflow.org
