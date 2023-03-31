import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from klasy.KNN import KNN,bcolors
import numpy as np


def column(matrix, i):
    return [row[i] for row in matrix]

cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])

# file_irys=open("./IRYS/Irys.data","r+")
# file_irys_trening, file_irys_test, file_irys_walidacja = trening_test_walidacja_split(file_irys,trening_size=0.6,test_size=0.2,random_state=1234)

file_irys_trening=open("./IRYS/Trening.data","r+")
file_irys_test=open("./IRYS/Walidacyjny.data","r+")
file_irys_walidacja=open("./IRYS/Walidacyjny.data","r+")

#Trening

wiersz_trening=file_irys_trening.read().split("\n")
wiersz_test=file_irys_test.read().split("\n")
wiersz_walidacja=file_irys_walidacja.read().split("\n")

a=0


Wszystko_Trening=[]
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



knn=KNN(2)

knn.fit(Wszystko_Trening)
Dane=[Wszystko_Test[0],Wszystko_Test[1],Wszystko_Test[2],Wszystko_Test[3]]
do_przewidzenia=knn.predict(Dane)



knn.get_tablica_pomylek(do_przewidzenia,Wszystko_Test[4])
knn.print_tablica_pomylek()

print("\n")
knn.Reszta_tablica_pomylek(do_przewidzenia,Wszystko_Test[4])
print("\n")

#acc=np.sum(do_przewidzenia==Wszystko_Test[4])/len(Wszystko_Test[4])







