# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:06:29 2021

@author: Ilke Aksoy
"""
#Kütüphane atamaları
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
#Verinin okunması
veriler = pd.read_csv('ReplicatedAcousticFeatures-ParkinsonDatabase (2).csv')

#Verinin bağımsız değişken ve bağımlı olarak bölünmesi
x=veriler.iloc[:,3:48]#Bağımsız Değişkenler DataFrame Pandas
y=veriler.iloc[:,2]#Bağımlı Değişkenler DataFrame Pandas

Y=y.values #Numpy array formunda
X=x.values


print(veriler.corr()) #Yakınlık matrisi oluşturmak için kullanılır

#Veri setinin bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0) #Genel olarak en yüksek doğruluk oranı olan 33 67 olarak seçilmiştir



#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,Y)

model=sm.OLS(regressor.predict(X),X)
print(model.fit().summary())
print("Linear R2 Değeri")
print(r2_score(Y,regressor.predict(X)))

  
#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)

print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#tahminler

print('poly OLS')

model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
sc1=StandardScaler() 
x_olcekli = sc1.fit_transform(X) 
sc2=StandardScaler() 
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


#svr regression
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)


print('SVR OLS')
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print('Decision Tree OLS')
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))



#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())



print('Random Forest OLS')
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))


#Sınıflandırma Algoritmaları

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Linear")
print(cm)


#En yakın komşu algoritması(1 neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("K-NN-1")
print(cm)


#En yakın komşu algoritması(3 neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("K-NN-3")
print(cm)


#En yakın komşu algoritması(5 neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("K-NN-5")
print(cm)


#Support Vector Machine sigmoid
from sklearn.svm import SVC
svc = SVC(kernel='sigmoid')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC-Sigmoid')
print(cm)


#Support Vector Machine rbf
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC-rbf')
print(cm)


#Support Vector Machine linear
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC-linear')
print(cm)


#Support Vector Machine linear
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC-poly')
print(cm)


#Naive Bayes GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)


#Naive Bayes BernoulliNB
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('BNB')
print(cm)


#Karar Ağacı entropy
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC entropy')
print(cm)


#Karar Ağacı gini
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'gini')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC gini')
print(cm)


#Random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')#n_estimators=10 ağaç sayısı
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)