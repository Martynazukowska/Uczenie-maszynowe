# Import required libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap


# Import necessary modules
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix

def print_tablica_pomylek(Tablica_pomylek):

    S0=str(Tablica_pomylek[0][0])
    if Tablica_pomylek[0][0]<10:
        S0="0"+S0
    S1=str(Tablica_pomylek[0][1])
    if Tablica_pomylek[0][1]<10:
        S1="0"+S1
    S2=str(Tablica_pomylek[0][2])
    if Tablica_pomylek[0][2]<10:
        S2="0"+S2

    Ve0=str(Tablica_pomylek[1][0])
    if Tablica_pomylek[1][0]<10:
        Ve0="0"+Ve0
    Ve1=str(Tablica_pomylek[1][1])
    if Tablica_pomylek[1][1]<10:
        Ve1="0"+Ve1
    Ve2=str(Tablica_pomylek[1][2])
    if Tablica_pomylek[1][2]<10:
        Ve2="0"+Ve2

    Vi0=str(Tablica_pomylek[2][0])
    if Tablica_pomylek[2][0]<10:
        Vi0="0"+Vi0
    Vi1=str(Tablica_pomylek[2][1])
    if Tablica_pomylek[2][1]<10:
        Vi1="0"+Vi1
    Vi2=str(Tablica_pomylek[2][2])
    if Tablica_pomylek[2][2]<10:
        Vi2="0"+Vi2

    print(bcolors.BOLD+"                 |"+bcolors.BOLD+bcolors.OKGREEN+"   Iris-setosa   "+bcolors.BOLD+bcolors.ENDC+"| "+bcolors.BOLD+bcolors.OKBLUE+"Iris-versicolor "+bcolors.ENDC+"| "+bcolors.FAIL+" Iris-virginica "+bcolors.ENDC+"|")
    print(bcolors.BOLD+bcolors.OKGREEN+"   Iris-setosa   "+bcolors.ENDC+"|       "+bcolors.OKGREEN+S0+bcolors.ENDC+"        |       "+S1+"        |       "+S2+"        |")
    print(bcolors.BOLD+bcolors.OKBLUE+" Iris-versicolor "+bcolors.ENDC+"|       "+Ve0+"        |       "+bcolors.OKBLUE+Ve1+bcolors.ENDC+"        |       "+Ve2+"        |")
    print(bcolors.BOLD+bcolors.FAIL+"  Iris-virginica "+bcolors.ENDC+"|       "+Vi0+"        |       "+Vi1+"        |       "+bcolors.FAIL+Vi2+bcolors.ENDC+"        |")

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])

test = pd.read_csv('test.csv', header=None, on_bad_lines='skip')
trening = pd.read_csv('trening.csv', header=None)
zbior_test = test.values
zbior_trening = trening.values

x_test = zbior_test[:, :len(zbior_test[0])-1]
y_test = zbior_test[:, -1]

X_test=x_test.astype(float)

x_train = zbior_trening[:, :len(zbior_trening[0])-1]
y_train = zbior_trening[:, -1]

X_train=x_train.astype(float)

i=0
plt.title("Dane testowe")
plt.subplot(1, 2, 1)
while i<len(y_test):
    if y_test[i]=="Iris-setosa":
        plt.plot(x_test[i][0], x_test[i][1], 'o', c='b')
    elif y_test[i]=="Iris-versicolor":
        plt.plot(x_test[i][0], x_test[i][1], 'o', c='r')
    elif y_test[i]=="Iris-virginica":
        plt.plot(x_test[i][0], x_test[i][1], 'o', c='g')
    else:
        print(y_test[i])
    i=i+1
i=0
plt.legend(['Iris-virginica','Iris-versicolor','Iris-setosa'])
plt.xlabel('Sepal length') 
plt.ylabel('Sepal width') 
plt.subplot(1, 2, 2)
while i<len(y_test):
    if y_test[i]=="Iris-setosa":
        plt.plot(x_test[i][2], x_test[i][3], 'o', c='b')
    elif y_test[i]=="Iris-versicolor":
        plt.plot(x_test[i][2], x_test[i][3], 'o', c='r')
    elif y_test[i]=="Iris-virginica":
        plt.plot(x_test[i][2], x_test[i][3], 'o', c='g')
    else:
        print(y_test[i])
    i=i+1
plt.xlabel('Petal length') 
plt.ylabel('Petal width') 
plt.legend(['Iris-virginica','Iris-versicolor','Iris-setosa'])
plt.show()
print("model Multi-Layer Perceptron")

print("\n")

# czemu relu ?

# 1 ukryta warstwa o 100 neuronach 
# funkcja aktywacji ReLU (Rectified Linear Unit),
# optymalizator adam
# maksymalną liczbę iteracji, które algorytm może wykonać podczas uczenia sieci.
mlp = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, activation='relu', solver='adam', max_iter=1000)


mlp.fit(X_train,y_train)

predict_test = mlp.predict(X_test)


print_tablica_pomylek(confusion_matrix(y_test,predict_test))
#print(confusion_matrix(y_train,predict_train))
print("\n")

print(classification_report(y_test,predict_test))

# Testowanie modelu
accuracy = mlp.score(X_test, y_test)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy*100))

# mlp2 = MLPClassifier(hidden_layer_sizes=(8,8,8), learning_rate_init=0.01,activation='relu', solver='adam', max_iter=1000)


# mlp2.fit(X_train,y_train)

# predict_test = mlp2.predict(X_test)


# print_tablica_pomylek(confusion_matrix(y_test,predict_test))
# #print(confusion_matrix(y_train,predict_train))
# print("\n")

# print(classification_report(y_test,predict_test))

# # Testowanie modelu
# accuracy = mlp2.score(X_test, y_test)
# print("Dokładność klasyfikacji: {:.2f}%".format(accuracy*100))


print("\n")

print("model DecisionTreeClassifier")

print("\n")

#odrzucamy
# Tworzenie modelu DecisionTreeClassifier
Dtree = DecisionTreeClassifier(random_state=42)

# Trenowanie modelu
Dtree.fit(X_train, y_train)

predict_test = Dtree.predict(X_test)

print_tablica_pomylek(confusion_matrix(y_test,predict_test))
#print(confusion_matrix(y_train,predict_train))
print("\n")

print(classification_report(y_test,predict_test))

# Testowanie modelu
accuracy = Dtree.score(X_test, y_test)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy*100))



print("\n")

print("model KNeighborsClassifier")

print("\n")

# Tworzenie modelu KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors=3)

# Trenowanie modelu
Knn.fit(X_train, y_train)

predict_test2 = Knn.predict(X_test)

print_tablica_pomylek(confusion_matrix(y_test,predict_test2))
#print(confusion_matrix(y_train,predict_train))
print("\n")

print(classification_report(y_test,predict_test2))

# Testowanie modelu
accuracy = Knn.score(X_test, y_test)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy*100))


print("\n")

print("model RandomForestClassifier")

print("\n")

#mieszanie=bootstraping+agregacja=baging

# Tworzenie modelu RandomForestClassifier
RTree = RandomForestClassifier(random_state=42)

# Trenowanie modelu
RTree.fit(X_train, y_train)

predict_test3 = RTree.predict(X_test)

print_tablica_pomylek(confusion_matrix(y_test,predict_test3))
#print(confusion_matrix(y_train,predict_train))
print("\n")

print(classification_report(y_test,predict_test3))

# Testowanie modelu
accuracy = RTree.score(X_test, y_test)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy*100))


print("\n")

print("model SVM")

print("\n")

# Train the SVM model
clf = SVC(kernel='linear', C=1, random_state=42)
clf.fit(X_train, y_train)

predict_test4 = clf.predict(X_test)

print_tablica_pomylek(confusion_matrix(y_test,predict_test4))
print("\n")
print(classification_report(y_test,predict_test4))

# Evaluate the performance of the model
accuracy = clf.score(X_test, y_test)

print("\n")
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy*100))









