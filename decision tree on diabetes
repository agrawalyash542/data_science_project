import pandas as pd
df = pd.read_csv("/content/diabetes.csv")
# print(df.head)

x = df[["Pregnancies","Glucose", "BloodPressure","SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = df["Outcome"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print(y_pred)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("accuracy = ",accuracy)

from sklearn.metrics import precision_score
precision = precision_score(y_test,y_pred, average='macro')
print("precision = ",precision)

from sklearn.metrics import recall_score
recall = recall_score(y_test,y_pred, average='macro')
print("recall = ",recall)

from sklearn.metrics import f1_score
f1 = f1_score(y_test,y_pred, average='macro')
print("f1 = ",f1)

# decision tree ploting
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True)
plt.show()
