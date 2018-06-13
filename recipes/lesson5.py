"""
Writing my own classifier instead of using the sklearn.tree or sklearn.neighbors
https://www.youtube.com/watch?v=AoeEHqVSNOw&list=PLT6elRN3Aer7ncFlaCz8Zz-4B5cnsrOMt&index=6

We will implement a very basic (scrappy) version of K-nearest neighbors
"""

# All the code here is from lesson4, except for the classifier that was used
# which we commented here, to replace that with our own classifier

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

from scipy.spatial import distance

class ScrappyKNNClassifier(object):
    # need to implement, fit and predict
    def fit(self, X_train, y_train):
        """
        Fit takes in the features and labels data and trains
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, k=1):
        """
        Takes in the feature data and returns the labels
        """
        # for the knn algo, we find the closest neighbor and use
        # that label as the prediction.
        # closes is counted as the linear distance.
        # scipy.spatial.distance.euc(a, b) gives the distance
        predictions = []
        for row in X_test:
            if k == 1:
                label = self.closest(row)
            elif k > 1:
                label = self.kClosest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = distance.euclidean(row, self.X_train[0])
        best_idx = 0
        for t_idx, t_row in enumerate(self.X_train[1:]):
            t_dist = distance.euclidean(row, t_row)
            if t_dist < best_dist:
                best_dist = t_dist
                best_idx = t_idx
        return self.y_train[best_idx]

    def kClosest(self, row):
        best_indices = []
        for i in range(3):
            best_dist = 4**32
            best_idx = None
            for t_idx, t_row in enumerate(self.X_train):
                if t_idx in best_indices:
                    continue
                t_dist = distance.euclidean(row, t_row)
                if t_dist < best_dist:
                    best_dist = t_dist
                    best_idx = t_idx
            best_indices.append(best_idx)
        best_labels = {}
        for idx in best_indices:
            label = self.y_train[idx]
            if label not in best_labels:
                best_labels[label] = 0
            best_labels[label] += 1
        # find the index with the max count
        max_label = None
        max_count = 0
        for label, count in best_labels.items():
            if count > max_count:
                max_label = label
        return max_label


# lets also write a random classifier where for the predict
# it just returns a random value from the training example
class RandomClassifier(object):
    # need to implement, fit and predict
    def fit(self, X_train, y_train):
        """
        Fit takes in the features and labels data and trains
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Takes in the feature data and returns the labels
        """
        import random
        predictions = []
        for _ in X_test:
            predictions.append(random.choices(self.y_train))
        return predictions

from sklearn import neighbors
knn_clf = neighbors.KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
predictions = knn_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print("The accuracy score with KSNN Classifier is ", end="")
print(accuracy_score(y_test, predictions))


random_clf = RandomClassifier()
random_clf.fit(X_train, y_train)
predictions = random_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print("The accuracy score with Random Classifier is ", end="")
print(accuracy_score(y_test, predictions))


sknn_clf = ScrappyKNNClassifier()
sknn_clf.fit(X_train, y_train)
predictions = sknn_clf.predict(X_test)
predictions_k = sknn_clf.predict(X_test, 5)

from sklearn.metrics import accuracy_score
print("The accuracy score with Scrappy KNN Classifier is ", end="")
print(accuracy_score(y_test, predictions))
print("The accuracy score with Scrappy 5 KNN Classifier is ", end="")
print(accuracy_score(y_test, predictions_k))