from klasy.kwiat import Kwiat 

file_irys=open("./IRYS/iris.data","r+")

#print(file_irys.read())

wiersz=file_irys.read().split("\n")

a=0

Wszystko=[]

while a<150:
    for ilosc in wiersz:
        dane=wiersz[a].split(",")
        Wszystko.append(Kwiat(dane[0],dane[1],dane[2],dane[3],dane[4]))
        a=a+1
        if a==150:
            break

a=0

while a<150:
    for ilosc in wiersz:
        print(Wszystko[a].print())
        a=a+1
        if a==150:
            break