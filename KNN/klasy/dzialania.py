import numpy as np
from collections import Counter

def column(matrix, i):
    return [row[i] for row in matrix]


def rownanie_odleglosci(x1,x2):
    rownanie=np.sum((float(x1)-float(x2))**2)
    return np.sqrt(rownanie)

class KNN:
    def __init__(self,k):
        self.k=k

    def fit(self, Wszystkie_trening):
        self.Dane_Kwiaty_Tr=[Wszystkie_trening[0],Wszystkie_trening[1],Wszystkie_trening[2],Wszystkie_trening[3]]
        self.Klasa_Kwiaty=Wszystkie_trening[4]

    def predict(self,Wszystkie):
        przewidywane_klasy=[self._predict(jeden) for jeden in Wszystkie[0]]
        return np.array(przewidywane_klasy)
    
    def _predict(self,jeden):
        #obliczanie odległości
        odleglosc=[rownanie_odleglosci(jeden, dane_do_trenowania) for dane_do_trenowania in self.Dane_Kwiaty_Tr[0]]

        sasiedz=np.argsort(odleglosc)[:self.k]
        nowa_nazwa=[self.Klasa_Kwiaty[i] for i in sasiedz]
    
        # 2 elementowa lista z wartościa i iloscia wystepowania    
        najczesciej_wystepujaca=Counter(nowa_nazwa).most_common(1)
        return najczesciej_wystepujaca[0][0]


