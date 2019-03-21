from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

from sklearn import tree
nordan_tree = tree.DecisionTreeClassifier()

nordan_tree.fit(X_train, y_train)

predictions = nordan_tree.predict(X_test)

print('Accuracy')
print(accuracy_score(predictions,y_test))
print('Matriz confusao')
print(confusion_matrix(y_test, predictions))
print('Precisao')
print(classification_report(y_test, predictions))
print('MCC')
print(matthews_corrcoef(y_test, predictions))
print('Recall')
print(recall_score(y_test, predictions,average = 'macro'))