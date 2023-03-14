import numpy as np
from collections import Counter

def column(matrix, i):
    return [row[i] for row in matrix]


def rownanie_odleglosci(x1,x2,y1,y2,a1,a2,b1,b2):
    rownanie=(float(x1)-float(x2))**2+(float(y1)-float(y2))**2+(float(a1)-float(a2))**2+(float(b1)-float(b2))**2
    return np.sqrt(rownanie)

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

class KNN:
    def __init__(self,k):
        self.k=k
        #                  Iris-setosa Iris-versicolor Iris-virginica
        # Iris-setosa      
        # Iris-versicolor
        # Iris-virginica
        self.Tablica_pomylek=[[0,0,0],[0,0,0],[0,0,0]]
        #dokładność
        self.acc=0

    def fit(self, Wszystkie_trening):
        self.Dane_Kwiaty_Tr=[Wszystkie_trening[0],Wszystkie_trening[1],Wszystkie_trening[2],Wszystkie_trening[3]]
        self.Klasa_Kwiaty=Wszystkie_trening[4]

    def predict(self,Wszystkie):
        #transponse
        Wszystkie=np.array(Wszystkie).T

        przewidywane_klasy=[self._predict(jeden) for jeden in Wszystkie]
        
        return np.array(przewidywane_klasy)
    
    def _predict(self,jeden):
        #obliczanie odległości

        pom=0
        POM=[]
        for cos in self.Dane_Kwiaty_Tr[0]:
            POM.append(pom)
            pom=pom+1
        
        odleglosc=[rownanie_odleglosci(jeden[0], self.Dane_Kwiaty_Tr[0][a],jeden[1], self.Dane_Kwiaty_Tr[1][a],jeden[2], self.Dane_Kwiaty_Tr[2][a],jeden[3], self.Dane_Kwiaty_Tr[3][a]) for a in POM]

        sasiedz=np.argsort(odleglosc)[:self.k]
        nowa_nazwa=[self.Klasa_Kwiaty[i] for i in sasiedz]
    
        # 2 elementowa lista z wartościa i iloscia wystepowania    
        najczesciej_wystepujaca=Counter(nowa_nazwa).most_common(1)
        return najczesciej_wystepujaca[0][0]

    def get_tablica_pomylek(self, do_przewidzenia, zbior):
        a=0
        for item in do_przewidzenia:
            if do_przewidzenia[a]=="Iris-setosa":
                if "Iris-setosa"==zbior[a]:
                    self.Tablica_pomylek[0][0]=self.Tablica_pomylek[0][0]+1
                if "Iris-versicolor"==zbior[a]:
                    self.Tablica_pomylek[0][1]=self.Tablica_pomylek[0][1]+1
                if "Iris-virginica"==zbior[a]:
                    self.Tablica_pomylek[0][2]=self.Tablica_pomylek[0][2]+1
            elif do_przewidzenia[a]=="Iris-versicolor":
                if "Iris-setosa"==zbior[a]:
                    self.Tablica_pomylek[1][0]=self.Tablica_pomylek[1][0]+1
                if "Iris-versicolor"==zbior[a]:
                    self.Tablica_pomylek[1][1]=self.Tablica_pomylek[1][1]+1
                if "Iris-virginica"==zbior[a]:
                    self.Tablica_pomylek[1][2]=self.Tablica_pomylek[1][2]+1
            elif do_przewidzenia[a]=="Iris-virginica":
                if "Iris-setosa"==zbior[a]:
                    self.Tablica_pomylek[2][0]=self.Tablica_pomylek[2][0]+1
                if "Iris-versicolor"==zbior[a]:
                    self.Tablica_pomylek[2][1]=self.Tablica_pomylek[2][1]+1
                if "Iris-virginica"==zbior[a]:
                    self.Tablica_pomylek[2][2]=self.Tablica_pomylek[2][2]+1
            a=a+1

        return self.Tablica_pomylek

    def print_tablica_pomylek(self):

        S0=str(self.Tablica_pomylek[0][0])
        if self.Tablica_pomylek[0][0]<10:
            S0="0"+S0
        S1=str(self.Tablica_pomylek[0][1])
        if self.Tablica_pomylek[0][1]<10:
            S1="0"+S1
        S2=str(self.Tablica_pomylek[0][2])
        if self.Tablica_pomylek[0][2]<10:
            S2="0"+S2

        Ve0=str(self.Tablica_pomylek[1][0])
        if self.Tablica_pomylek[1][0]<10:
            Ve0="0"+Ve0
        Ve1=str(self.Tablica_pomylek[1][1])
        if self.Tablica_pomylek[1][1]<10:
            Ve1="0"+Ve1
        Ve2=str(self.Tablica_pomylek[1][2])
        if self.Tablica_pomylek[1][2]<10:
            Ve2="0"+Ve2

        Vi0=str(self.Tablica_pomylek[2][0])
        if self.Tablica_pomylek[2][0]<10:
            Vi0="0"+Vi0
        Vi1=str(self.Tablica_pomylek[2][1])
        if self.Tablica_pomylek[2][1]<10:
            Vi1="0"+Vi1
        Vi2=str(self.Tablica_pomylek[2][2])
        if self.Tablica_pomylek[2][2]<10:
            Vi2="0"+Vi2

        print(bcolors.BOLD+"                 |"+bcolors.BOLD+bcolors.OKGREEN+"   Iris-setosa   "+bcolors.BOLD+bcolors.ENDC+"| "+bcolors.BOLD+bcolors.OKBLUE+"Iris-versicolor "+bcolors.ENDC+"| "+bcolors.FAIL+" Iris-virginica "+bcolors.ENDC+"|")
        print(bcolors.BOLD+bcolors.OKGREEN+"   Iris-setosa   "+bcolors.ENDC+"|       "+bcolors.OKGREEN+S0+bcolors.ENDC+"        |       "+S1+"        |       "+S2+"        |")
        print(bcolors.BOLD+bcolors.OKBLUE+" Iris-versicolor "+bcolors.ENDC+"|       "+Ve0+"        |       "+bcolors.OKBLUE+Ve1+bcolors.ENDC+"        |       "+Ve2+"        |")
        print(bcolors.BOLD+bcolors.FAIL+"  Iris-virginica "+bcolors.ENDC+"|       "+Vi0+"        |       "+Vi1+"        |       "+bcolors.FAIL+Vi2+bcolors.ENDC+"        |")

    def Reszta_tablica_pomylek(self,do_przewidzenia, zbior):
            self.acc=np.sum(do_przewidzenia==zbior)/len(zbior)

