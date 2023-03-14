import numpy as np
from collections import Counter
from klasy.kwiat import Kwiat

def column(matrix, i):
    return [row[i] for row in matrix]


def rownanie_odleglosci(x1,x2,y1,y2,a1,a2,b1,b2):
    rownanie=(float(x1)-float(x2))**2+(float(y1)-float(y2))**2+(float(a1)-float(a2))**2+(float(b1)-float(b2))**2
    return np.sqrt(rownanie)

class KNN:
    def __init__(self,k):
        self.k=k

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
