
class Kwiat:
    def __init__(self,sepal_length,sepal_width,petal_length,petal_width,klasa):
        self.sepal_length=sepal_length
        self.sepal_width=sepal_width
        self.petal_length=petal_length
        self.petal_width=petal_width
        self.klasa=klasa

    def get_sepal_length(self):
        return self.sepal_length

    def get_sepal_width(self):
        return self.sepal_width

    def get_petal_length(self):
        return self.petal_length

    def get_petal_width(self):
        return self.petal_width

    def get_klasa(self):
        return self.klasa
    
    def print(self):
        return self.sepal_length, self.sepal_width, self.petal_length, self.petal_width, self.klasa
