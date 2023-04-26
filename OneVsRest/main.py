import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from klasy.Preceptron import Preceptron

cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])

file_data = pd.read_csv('data_banknote_authentication.txt', sep="\t", header=None)

train, test = train_test_split(file_data, test_size= 0.4,shuffle= True, random_state= 0)
train1, test1 = train_test_split(test, test_size= 0.5,shuffle= True, random_state= 0)

Train= np.asarray(train)
Test= np.asarray(train1)
Walidacja= np.asarray(test1)

#f = open("test.txt", "w")
#f1=open("trening.txt", "w")
#f2=open("walidacja.txt", "w")
pom1=0

#while pom1!=len(Test):
#    f.write(Test[pom1][0])
#    pom1=pom1+1
#    f.write("\n")

pom1=0

#while pom1!=len(Train):
#    f1.write(Train[pom1][0])
#    pom1=pom1+1
#    f1.write("\n")

pom1=0

#while pom1!=len(Walidacja):
#    f2.write(Walidacja[pom1][0])
#    pom1=pom1+1
#    f2.write("\n")

#f.close()
#f1.close()
#f2.close()

test = pd.read_csv('walidacja.csv', header=None)
trening = pd.read_csv('trening.csv', header=None)
zbior_test = test.values
zbior_trening = trening.values

x_test = zbior_test[:, :len(zbior_test[0])-1]
y_test = zbior_test[:, -1]
y_test[y_test==0] = -1

x_train = zbior_trening[:, :len(zbior_trening[0])-1]
y_train = zbior_trening[:, -1]
y_train[y_train==0] = -1


p=Preceptron(learning_rate=0.01,n_iters=1000)
p.fit(x_train,y_train)
szukane=p.predict(x_test)

acc=np.sum(y_test==szukane)/len(y_test)

print("\n")
print(p.get_tablica_pomylek(szukane,y_test)[0])
print(p.get_tablica_pomylek(szukane,y_test)[1])
print("\n")

print(acc)