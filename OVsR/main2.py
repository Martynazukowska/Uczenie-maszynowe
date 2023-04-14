import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

from klasy.Preceptron import Preceptron

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

def get_tablica_pomylek(Tablica_pomylek, do_przewidzenia, zbior):
    a=0
    for item in do_przewidzenia:
        if do_przewidzenia[a]=="Iris-setosa":
            if "Iris-setosa"==zbior[a]:
                Tablica_pomylek[0][0]=Tablica_pomylek[0][0]+1
            if "Iris-versicolor"==zbior[a]:
                Tablica_pomylek[0][1]=Tablica_pomylek[0][1]+1
            if "Iris-virginica"==zbior[a]:
                Tablica_pomylek[0][2]=Tablica_pomylek[0][2]+1
        elif do_przewidzenia[a]=="Iris-versicolor":
            if "Iris-setosa"==zbior[a]:
                Tablica_pomylek[1][0]=Tablica_pomylek[1][0]+1
            if "Iris-versicolor"==zbior[a]:
                Tablica_pomylek[1][1]=Tablica_pomylek[1][1]+1
            if "Iris-virginica"==zbior[a]:
                Tablica_pomylek[1][2]=Tablica_pomylek[1][2]+1
        elif do_przewidzenia[a]=="Iris-virginica":
            if "Iris-setosa"==zbior[a]:
                Tablica_pomylek[2][0]=Tablica_pomylek[2][0]+1
            if "Iris-versicolor"==zbior[a]:
                Tablica_pomylek[2][1]=Tablica_pomylek[2][1]+1
            if "Iris-virginica"==zbior[a]:
                Tablica_pomylek[2][2]=Tablica_pomylek[2][2]+1
        a=a+1
    return Tablica_pomylek

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



cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])

test = pd.read_csv('walidacja.csv', header=None, on_bad_lines='skip')
trening = pd.read_csv('trening.csv', header=None)
zbior_test = test.values
zbior_trening = trening.values

x_test = zbior_test[:, :len(zbior_test[0])-1]
y_test = zbior_test[:, -1]

x_test=x_test.astype(float)

#zeby nie działać na orginale
y_test_setosa=y_test.copy()
y_test_versi=y_test.copy()
y_test_virgi=y_test.copy()

y_test_setosa[y_test_setosa=="Iris-setosa"] = 1
y_test_setosa[y_test_setosa=="Iris-versicolor"] = -1
y_test_setosa[y_test_setosa=="Iris-virginica"] = -1

y_test_virgi[y_test_virgi=="Iris-setosa"] = -1
y_test_virgi[y_test_virgi=="Iris-versicolor"] = -1
y_test_virgi[y_test_virgi=="Iris-virginica"] = 1

y_test_versi[y_test_versi=="Iris-setosa"] = -1
y_test_versi[y_test_versi=="Iris-versicolor"] = 1
y_test_versi[y_test_versi=="Iris-virginica"] = -1

x_train = zbior_trening[:, :len(zbior_trening[0])-1]
y_train = zbior_trening[:, -1]

x_train=x_train.astype(float)

y_train_setosa=y_train.copy()
y_train_versi=y_train.copy()
y_train_virgi=y_train.copy()

y_train_setosa[y_train_setosa=="Iris-setosa"] = 1
y_train_setosa[y_train_setosa=="Iris-versicolor"] = -1
y_train_setosa[y_train_setosa=="Iris-virginica"] = -1

y_train_virgi[y_train_virgi=="Iris-setosa"] = -1
y_train_virgi[y_train_virgi=="Iris-versicolor"] = -1
y_train_virgi[y_train_virgi=="Iris-virginica"] = 1

y_train_versi[y_train_versi=="Iris-setosa"] = -1
y_train_versi[y_train_versi=="Iris-versicolor"] = 1
y_train_versi[y_train_versi=="Iris-virginica"] = -1

#######################################################
#zapytać się o versicolor

print("S\n")
Psetosa=Preceptron(learning_rate=0.01,n_iters=1000)
Psetosa.fit(x_train,y_train_setosa)
szukaneS=Psetosa.predict(x_test)

print("V\n")

Pversi=Preceptron(learning_rate=0.01,n_iters=1000)
Pversi.fit(x_train,y_train_versi)
szukaneV2=Pversi.predict(x_test)

print("V\n")

Pvirgi=Preceptron(learning_rate=0.01,n_iters=1000)
Pvirgi.fit(x_train,y_train_virgi)
szukaneV=Pvirgi.predict(x_test)

#############################################################
#acc=np.sum(y_test_setosa==szukaneS)/len(y_test_setosa)

#acc=np.sum(y_test_virgi==szukaneV)/len(y_test_virgi)

Tablica_pomylek=[[0,0,0],[0,0,0],[0,0,0]]

szukane=[]
################################################################
#zapytać się o to, czemu musi być pom a nie  może być in szukaneS
pom=len(szukaneS)

print("\n")
print(szukaneS)
print("\n")
print(szukaneV2)
print("\n")
print(szukaneV)



for index in range(pom):
    if szukaneV[index]==1 and szukaneS[index]==1 and szukaneV2[index]==1:
        if Psetosa.get_przed_sig(index)>Pversi.get_przed_sig(index):
            if Psetosa.get_przed_sig(index)>Pvirgi.get_przed_sig(index):
                szukane.append("Iris-setosa")
            elif Pvirgi.get_przed_sig(index)>Pversi.get_przed_sig(index):
                szukane,append("Iris-virginica")
            else:
                szukane.append("Iris-versicolor")
    elif szukaneV[index]==-1 and szukaneS[index]==1 and szukaneV2[index]==1:
        if Psetosa.get_przed_sig(index)>Pversi.get_przed_sig(index):
            szukane.append("Iris-setosa")
        else:
            szukane.append("Iris-versicolor")
    elif szukaneV[index]==1 and szukaneS[index]==-1 and szukaneV2[index]==1:
        if Pvirgi.get_przed_sig(index)>Pversi.get_przed_sig(index):
            szukane.append("Iris-virginica")
        else:
            szukane.append("Iris-versicolor")
    elif szukaneV[index]==1 and szukaneS[index]==1 and szukaneV2[index]==-1:
        if Psetosa.get_przed_sig(index)>Pversi.get_przed_sig(index):
            szukane.append("Iris-setosa")
        else:
            szukane.append("Iris-versicolor")
    elif szukaneV[index]==1 and szukaneS[index]==-1 and szukaneV2[index]==-1:
        szukane.append("Iris-virginica")
    elif szukaneS[index]==1 and szukaneV[index]==-1 and szukaneV2[index]==-1:
        szukane.append("Iris-setosa")
    elif szukaneV2[index]==1 and szukaneS[index]==-1 and szukaneV[index]==-1:
        szukane.append("Iris-versicolor")
    else:
        szukane.append("blad")

print("\n")
print(szukane)
acc=np.sum(y_test==szukane)/len(szukane)

Tablica_pomylek=get_tablica_pomylek(Tablica_pomylek,y_test,szukane)

print("\n")
print_tablica_pomylek(Tablica_pomylek)

print("\n")
print("Dokładność: ",acc)
print("\n")


