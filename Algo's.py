import pandas as pd                #pd is  an Alias
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
nb=MultinomialNB()
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5, 2),random_state=0)


data = pd.read_csv("iris.data")

x = data.drop('Species', axis=1)
y = data['Species']


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3 )

rf.fit(x_train, y_train)
lr.fit(x_train, y_train)
nb.fit(x_train, y_train)
gbm.fit(x_train, y_train)
dt.fit(x_train, y_train)
sv.fit(x_train, y_train)
nn.fit(x_train, y_train)

rf_predict = rf.predict(x_test)
lr_predict = lr.predict(x_test)
nb_predict = nb.predict(x_test)
gbm_predict = gbm.predict(x_test)
dt_predict = dt.predict(x_test)
sv_predict = sv.predict(x_test)
nn_predict = nn.predict(x_test)

print('RandomForest', accuracy_score(y_test, rf_predict))
print('Logistic', accuracy_score(y_test, lr_predict))
print('NaiveBayes', accuracy_score(y_test, nb_predict))
print( 'GradientBoostingClassifier', accuracy_score(y_test, gbm_predict))
print('DecisionTreeClassifier', accuracy_score(y_test, dt_predict))
print('svm' ,accuracy_score(y_test, sv_predict))
print('NeuralNetwork' ,accuracy_score(y_test, nn_predict))


#RandomForest 0.9777777777777777
#Logistic 0.9777777777777777
#NaiveBayes 0.6
#GradientBoostingClassifier 0.9777777777777777
#DecisionTreeClassifier 0.9777777777777777
#svm 0.9777777777777777
#NeuralNetwork 0.24444444444444444

