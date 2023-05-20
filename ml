
//ml lab


exp - 1 (Impute missing values in data inputs):
import pandas as pd

df = pd.read_csv('/content/ml/StudentsPerformance.csv')
print(df)
op = df.isna()
print(op)
op = df['gender'].isna()
print(op)
op = df['gender'].fillna("male",inplace=True)
# print(op)
print(df)
df = pd.read_csv('/content/StudentsPerformance.csv')
mean = df['reading score'].mean()
print(mean)
# print(df['reading score'])
df['reading score'].fillna(mean,inplace=True)
print(df)
# print(df['writing score'].isna())
mode = df['writing score'].mode()
print(mode)
df['writing score'].fillna(mode,inplace=True)
print(df)
median = df['writing score'].median()
df['writing score'].fillna(median,inplace=True)
print(df)

data = pd.read_csv('/content/StudentsPerformance.csv')
print(data.isna())
d=data.dropna(inplace=True)
print(data)
df.replace('some college','KEC',inplace=True)
print(df)
df.to_csv('Final.csv')


exp - 2 (Use feature selection/extraction method to perform dimensionality reduction):

import numpy as np
x=np.array([4,8,13,7])
y=np.array([11,4,5,14])
xm=np.mean(x)
ym=np.mean(y)
print(xm,ym)
covxy=np.cov(x,y)
print(covxy)
w,v=np.linalg.eig(covxy)
print(w)
print(v)
vt=v.transpose()
print(vt)
e1,e2=np.hsplit(vt,2)
print(e1)
print(e2)
x=x-xm
y=y-ym
data=np.stack((x.T,y.T),axis=0)
print(data)
p1=e1*data
print(p1)
p2=e2*data
print(p2)

exp - 3(feed forward neural networks):

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X, y = make_classification(n_samples=100, random_state=1)
np.shape(X)
np.shape(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
np.shape(X_train)
clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(X_train, y_train)
pred=clf.predict(X_test[:, :])
pred
clf.score(X_test, y_test)
report = classification_report(y_test,pred)
print(report)

exp - 4(naive bayers):

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
Y = iris.target 
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred = gnb.predict(X_test)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy (in %)",metrics.accuracy_score(Y_test, Y_pred)*100)

exp - 5( decision tree) :
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from google.colab import files
 
 
uploaded = files.upload()
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("pima-indians-diabetes (1).csv", header=None, names=col_names)
pima.head()
#Feature Selection
#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Building Decision Tree Model
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

#Optimizing Decision Tree Performance
#Maximum depth of the tree can be used as a control variable for pre-pruning.
#you can plot a decision tree on the same data with max_depth=3.
#You can also try other attribute selection measure such as entropy.
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


exp - 6(svm):

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

n_samples_1 = 1000
n_samples_2 = 100

centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(
    n_samples = [n_samples_1, n_samples_2],
    centers=centers,
    cluster_std=clusters_std,
    random_state=0,
    shuffle=False,
)

print(X)
print(y)

clf = svm.SVC(kernel="linear", C=1.0)
clf.fit(X,y)
wclf = svm.SVC(kernel="linear", class_weight={1:10})
wclf.fit(X,y)

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, edgecolors="k")

pred = clf.predict([[2,4]])
print(pred)

pred = clf.predict([[2,-2]])
print(pred)


exp - 7(Multivariate):

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded=files.upload()

test=pd.read_csv("california_housing_test.csv")
plt.figure()

train.head()

sns.heatmap(train.corr(),cmap='coolwarm')
plt.show();

sns.lmplot(x='median_income',y='median_house_value',data=train)

sns.lmplot(x='housing_median_age',y='median_house_value',data=train)

data=train

data=data[['total_rooms','total_bedrooms','housing_median_age','median_income','population','households']]

data.info()

data['total_rooms']=data['total_rooms'].fillna(data['total_rooms'].mean())

data['total_bedrooms']=data['total_bedrooms'].fillna(data['total_bedrooms'].fillna(data['total_bedrooms'].mean()))

train.head()

from sklearn.model_selection import train_test_split
y=train.iloc[:,8]

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=0)
print(y.name)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

print(regressor.intercept_)
print(regressor.coef_)

Predictions=regressor.predict(X_test)

Predictions=Predictions.reshape(-1,1)
print(Predictions)

from sklearn.metrics import mean_squared_error
print('MSE:',mean_squared_error(y_test,Predictions))

print('RMSE:',np.sqrt(mean_squared_error(y_test,Predictions)))

import numpy as np
arr=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr=arr.reshape(-1,1)
print(newarr)

exp - 8 (k-means cluster):
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("birthplace-2018-census-csv.csv")
df

x = df.iloc[:,[2,3]].values
print(x)

kmeans2 = KMeans(n_clusters=2)
kmeansy = kmeans2.fit_predict(x)

print("Cluster Centers Are:")
print(kmeans2.cluster_centers_)

plt.scatter(x[:,0],x[:,1],c=kmeansy,cmap="viridis")
plt.show()


exp - 9(Cross_Validation):

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, y_train.shape

X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

Y_predict = clf.predict(X_test)
print(classification_report(Y_predict,y_test))

from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel='linear',C=1,random_state=42)

scores = cross_val_score(clf,X,y,cv=5)

scores
