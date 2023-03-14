import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from klasy.dzialania2 import KNN
import numpy as np


def column(matrix, i):
    return [row[i] for row in matrix]

cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])

file_irys_trening=open("./IRYS/Trening.data","r+")
file_irys_test=open("./IRYS/Test.data","r+")
file_irys_walidacja=open("./IRYS/Walidacyjny.data","r+")

#Trening

wiersz_trening=file_irys_trening.read().split("\n")
wiersz_test=file_irys_test.read().split("\n")
wiersz_walidacja=file_irys_walidacja.read().split("\n")

a=0
Tablica_pomylek=[[0,0],[0,0]]
print(Tablica_pomylek[0][0])

#Wszystko_Trening=[]
Wszystko_Test=[]
Wszystko_Walidacja=[]


SL=[]
SW=[]
PL=[]
PW=[]
klasa=[]

for ilosc in wiersz_trening:
    dane=wiersz_trening[a].split(",")
    SL.append(dane[0])
    SW.append(dane[1])
    PL.append(dane[2])
    PW.append(dane[3])
    klasa.append(dane[4])
    a=a+1

Wszystko_Trening=[SL,SW,PL,PW,klasa]


a=0
SL=[]
SW=[]
PL=[]
PW=[]
klasa=[]

for ilosc in wiersz_test:
    dane=wiersz_test[a].split(",")
    SL.append(dane[0])
    SW.append(dane[1])
    PL.append(dane[2])
    PW.append(dane[3])
    klasa.append(dane[4])
    a=a+1

Wszystko_Test=[SL,SW,PL,PW,klasa]

a=0
SL=[]
SW=[]
PL=[]
PW=[]
klasa=[]

for ilosc in wiersz_walidacja:
    dane=wiersz_walidacja[a].split(",")
    SL.append(dane[0])
    SW.append(dane[1])
    PL.append(dane[2])
    PW.append(dane[3])
    klasa.append(dane[4])
    a=a+1

Wszystko_Walidacja=[SL,SW,PL,PW,klasa]



a=[1,1,1]

knn=KNN(3)

knn.fit(Wszystko_Trening)
Dane=[Wszystko_Test[0],Wszystko_Test[1],Wszystko_Test[2],Wszystko_Test[3]]
do_przewidzenia=knn.predict(Dane)


acc=np.sum(do_przewidzenia==Wszystko_Test[4])/len(Wszystko_Test[4])

print(acc)




plt.figure()
a=0
y=[]
for ilosc in wiersz_trening:
    if Wszystko_Trening[4][a]=="Iris-setosa":
        y.append(0)
    if Wszystko_Trening[4][a]=="Iris-versicolor":
        y.append(1)
    if Wszystko_Trening[4][a]=="Iris-virginica":
        y.append(2)
    plt.scatter(Wszystko_Trening[0][a],Wszystko_Trening[1][a],c=y[a],cmap=cmap,edgecolor='k',s=20)
    a=a+1
plt.show()

print(y)


