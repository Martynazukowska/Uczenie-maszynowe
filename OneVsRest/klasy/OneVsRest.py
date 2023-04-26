

# class OvsR:
#     def __init__(self):
#         self.Tablica_pomylek=[[0,0],[0,0]]
#         #czulosc
#         self.tpr=0
#         #swoistosc
#         self.tnr=0
#         #precyzja
#         self.precyzjaS=0
#         self.precyzjaVe=0
#         self.precyzjaVi=0
#         #dokładność
#         self.acc=0

#     def fit(self, Wszystkie_trening):
#         self.Dane=[Wszystkie_trening[0],Wszystkie_trening[1],Wszystkie_trening[2],Wszystkie_trening[3]]
#         self.Klasa=Wszystkie_trening[4]

#     def predict(self,Wszystkie):

    
#     def _predict(self,jeden):
#         #obliczanie odległości


#     def get_tablica_pomylek(self, do_przewidzenia, zbior):
#         a=0
#         for item in do_przewidzenia:
#             if do_przewidzenia[a]=="Iris-setosa":
#                 if "Iris-setosa"==zbior[a]:
#                     self.Tablica_pomylek[0][0]=self.Tablica_pomylek[0][0]+1
#                 if "Iris-versicolor"==zbior[a]:
#                     self.Tablica_pomylek[0][1]=self.Tablica_pomylek[0][1]+1
#                 if "Iris-virginica"==zbior[a]:
#                     self.Tablica_pomylek[0][2]=self.Tablica_pomylek[0][2]+1
#             elif do_przewidzenia[a]=="Iris-versicolor":
#                 if "Iris-setosa"==zbior[a]:
#                     self.Tablica_pomylek[1][0]=self.Tablica_pomylek[1][0]+1
#                 if "Iris-versicolor"==zbior[a]:
#                     self.Tablica_pomylek[1][1]=self.Tablica_pomylek[1][1]+1
#                 if "Iris-virginica"==zbior[a]:
#                     self.Tablica_pomylek[1][2]=self.Tablica_pomylek[1][2]+1
#             elif do_przewidzenia[a]=="Iris-virginica":
#                 if "Iris-setosa"==zbior[a]:
#                     self.Tablica_pomylek[2][0]=self.Tablica_pomylek[2][0]+1
#                 if "Iris-versicolor"==zbior[a]:
#                     self.Tablica_pomylek[2][1]=self.Tablica_pomylek[2][1]+1
#                 if "Iris-virginica"==zbior[a]:
#                     self.Tablica_pomylek[2][2]=self.Tablica_pomylek[2][2]+1
#             a=a+1

#         return self.Tablica_pomylek

#     def Reszta_tablica_pomylek(self,do_przewidzenia, zbior):


