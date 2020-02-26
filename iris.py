#path ='E:\python files\ML\K-NN_nearst_neighbor\iris'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# we need to assign column names to the dataset as follows

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Now, we need to read dataset to pandas dataframe as follows

dataset = pd.read_csv('iris.csv', names = headernames)
print dataset.head()

# Data Preprocessing will be done with the help of following script lines

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print len(y)  # 150
# Next, we will divide the data into train and test split.
    #Following code will split the dataset into 60% training data and 40%
        #of testing data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.40)

#print x_train[:6]
'''
[[5.6 2.8 4.9 2. ]
 [4.9 2.4 3.3 1. ]
 [5.1 3.8 1.5 0.3]
 [4.8 3.1 1.6 0.2]
 [5.  3.4 1.6 0.4]
 [6.4 3.2 5.3 2.3]]'''
#print x_test[:4]
'''[[5.6 2.7 4.2 1.3]
 [4.9 2.5 4.5 1.7]
 [6.2 3.4 5.4 2.3]
 [5.5 2.3 4.  1.3]]'''
#print y_train[:6]
'''
['Iris-virginica' 'Iris-versicolor' 'Iris-setosa' 'Iris-setosa'
 'Iris-setosa' 'Iris-virginica']'''
# Next, data scaling will be done as follows

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print x_train[:6]
'''
[[-0.33271411 -0.53489081  0.63542814  1.03775407]
 [-1.19530625 -1.47421126 -0.28696755 -0.28329072]
 [-0.94885135  1.81341031 -1.3246627  -1.20802207]
 [-1.31853369  0.16959953 -1.26701296 -1.34012655]
 [-1.0720788   0.87408986 -1.26701296 -1.07591759]
 [ 0.65310547  0.40442964  0.86602706  1.43406751]]'''
#print x_test[:6]
'''
[[-1.31853369 -0.06523059 -1.38231243 -1.20802207]
 [-0.57916901  2.04824042 -1.43996216 -1.07591759]
 [ 1.26924272  0.16959953  0.7507276   1.43406751]
 [ 0.65310547  0.40442964  0.40482922  0.37723168]
 [ 0.03696823  0.40442964  0.57777841  0.77354512]
 [ 1.14601527  0.40442964  1.21192544  1.43406751]]'''
# Next, train the model with the help of KNeighborsClassifier class
   # of sklearn as follows

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(x_train, y_train)

# At last we need to make prediction. It can be done with the help
  # of following script
y_pred = classifier.predict(x_test)

# Next, print the results as follows

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print "Confusion Matrix:"
print result
result1 = classification_report(y_test, y_pred)
print "Classification Report:",
print result1
result2 = accuracy_score(y_test,y_pred)
print "Accuracy:",result2

plt.figure(figsize=(9,9))
sns.heatmap(result,annot=True, fmt =".3f",linewidths=.5,square=True,cmap='Blues_r');
 plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title='Accuracy score : {0}'.format(result2*100)
plt.title(all_sample_title,size=15);
#plt.savefig('confussion _matrix')
#plt.show()
# Output
'''
Confusion Matrix:
[[22  0  0]
 [ 0 17  2]
 [ 0  4 15]]
Classification Report:                  precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        22
Iris-versicolor       0.81      0.89      0.85        19
 Iris-virginica       0.88      0.79      0.83        19

      micro avg       0.90      0.90      0.90        60
      macro avg       0.90      0.89      0.89        60
   weighted avg       0.90      0.90      0.90        60

Accuracy: 0.9
'''
