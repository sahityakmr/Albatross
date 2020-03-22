import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dataset = pd.read_csv('iris.data', sep=',', header=None)
data = dataset.iloc[:, :]
print("Sum of NULL values in each column : ")
print(data.isnull().sum())

x = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)

print("Confusion Matrix : ")
print(confusion_matrix(y_test, predicted))
print("Accuracy Score : ")
print(accuracy_score(y_test, predicted))
print("Classification Report : ")
print(classification_report(y_test, predicted))
