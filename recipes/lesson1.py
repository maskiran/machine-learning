"""
Machine Learning recipes, Hello-World
https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLT6elRN3Aer7ncFlaCz8Zz-4B5cnsrOMt
"""

# sklearn (scikit-learn) is used. Its imported later 
# as the lesson progresses.

# data for fruit features and the fruit label
# weight texture fruit-name
# 140 smooth apple
# 130 smooth apple
# 150 bumpy orange
# 170 bumpy orange

# classifier will learn from the above data.
# the more data we have, the better the classifier can learn

# features contains the first 2 columns (as tuple)
features = [(140, "smooth"), (130, "smooth"), (150, "bumpy"), (170, "bumpy")]
#labels is the last column (string)
labels = ["apple", "apple", "orange", "orange"]

# instead of strings, its easier to deal with integers
# replace smooth with 1 and bumpy with 0
SMOOTH = 1
BUMPY = 0
features = [(140, SMOOTH), (130, SMOOTH), (150, BUMPY), (170, BUMPY)]

# replace labels with integers 0 for apple, 1 for orange
APPLE = 0
ORANGE = 1
labels = [APPLE, APPLE, ORANGE, ORANGE]

# there are many classifiers that can be used. DecisionTree is one
# of them.
# It follows a tree asking questions
# is weight < 150 ? follow left child, else follow right child
# left child: is texture smooth ? apple else orange
# its like writing if-else condition yourself but in a learning way
# so you dont have to write it
# DecisionTreeClassifier is at sklearn.tree

import sklearn.tree
clf = sklearn.tree.DecisionTreeClassifier()

# once this is defined, clf would learn the rules (the if-else conditions)
# by looking at the knows features (inputs) and the labels (outputs)
# fit is like "find patterns in data"
clf = clf.fit(features, labels) 

# at this point we got a classifier thats been trained

# to predict labels for new data, we use "predict" function
# that takes new example data. fit([(weight1, texture1), (weight2, texture2)])
# and return labels
# lets predict for weight 160 and BUMPY (0)
# since its a heavier fruit and bumpy texture, my guess is ORANGE
print(clf.predict([(160, BUMPY)]))
# should print [1] => 1 is ORANGE