import numpy as np


class Preceptron:

    def __init__(self, learning_rate=0.01,n_iters=1000):
        self.lr=learning_rate
        self.n=n_iters
        self.wagi=None
        self.cechy=None
        self.Dane=None
        self.Klasa=None
        self.Tablica_pomylek=[[0,0],[0,0]]
        

    def _unit_step_fun(self,x):
        #return np.sign(x)
        #return np.where(x>=0,1,0)
        return np.where(x>=0,1,-1)

    def fit(self, X,y):

        #samples=len(self.Dane[0])
        #features=len(self.Dane)

        self.samples,self.features=X.shape

        #z samyc zer
        self.wagi=np.zeros(self.features)
        self.cechy=0
        
        #zapytaj się o funkcję
        #_y=np.array([1 if i>0 else 0 for i in y])
        _y=np.array([1 if i>0 else -1 for i in y])

        #naucz się wag
        for _ in range(self.n):
            for index,x_i in enumerate(X):
                linear_output=np.dot(x_i,self.wagi)+self.cechy
                y_predicted=np.sign(linear_output)
                #y_predicted=self._unit_step_fun(linear_output)
                # czy jest to potrzebne? 
                if(y_predicted!=_y[index]):
                    update=self.lr*(_y[index]-y_predicted)
                    self.wagi+=update*x_i
                    self.cechy+=update


    def predict(self,X):
        #iloczyn skalarny
        linear_output=np.dot(X,self.wagi)+self.cechy
        y_predicted=self._unit_step_fun(linear_output)
        return y_predicted

    def get_tablica_pomylek(self, do_przewidzenia,zbior):
        a=0
        for item in do_przewidzenia:
            if do_przewidzenia[a]==-1:
                if -1==zbior[a]:
                    self.Tablica_pomylek[0][0]=self.Tablica_pomylek[0][0]+1
                elif 1==zbior[a]:
                    self.Tablica_pomylek[0][1]=self.Tablica_pomylek[0][1]+1
                else:
                    print("UWAGA")
            elif do_przewidzenia[a]==1:
                if -1==zbior[a]:
                    self.Tablica_pomylek[1][0]=self.Tablica_pomylek[1][0]+1
                elif 1==zbior[a]:
                    self.Tablica_pomylek[1][1]=self.Tablica_pomylek[1][1]+1
                else:
                    print("UWAGA")
            a=a+1

        return self.Tablica_pomylek




