
import pandas as pd
from CustomKnot import *
from KnotStar import *
from IPython.display import clear_output
from copy import copy,deepcopy
from time import time

dict_knot_numbers = {
    0 : 1,
    3 : 1,
    4 : 1,
    5 : 2,
    6 : 3,
    7 : 7,
    8 : 21,
    9 : 49,
    10: 165,
}

def knotNames(n: int) -> dict[int:list[str]]:
    auxDict = {numberCrosses: [] for numberCrosses in dict_knot_numbers.keys() if numberCrosses<= n}
    for numberCrosses in auxDict.keys():
        for number in range(dict_knot_numbers[numberCrosses]):
            name = str(numberCrosses)+"_"+str(number+1)
            auxDict[numberCrosses].append(name)
    return auxDict

def singleDatabase(maxCrosses: int):
    auxDict = {"name":[],"crosses":[]}
    names = knotNames(maxCrosses)
    for numberCrosses in names.keys():
        for name in names[numberCrosses]:
            auxDict["name"].append(name)
            knot = knotFromPyknotid(name)
            auxDict["crosses"].append(knot.crosses)

    return pd.DataFrame(auxDict)

def knotDatabase(maxCrosses: int, numberOfKnots: int, numberOfRandomMov: int,maxStrands: int = None,debug:bool = False):

    df = singleDatabase(maxCrosses)
    df = df.reset_index()  # make sure indexes pair with number of rows
    auxDict = {"name":[],"crosses":[]}
    for index, row in df.iterrows():
        knotSet: set[CustomKnot] = set()
        i = 0
        while i < numberOfKnots:
        #for i in range(numberOfKnots):
            #clear_output(wait=True)
            print("name", row["name"])
            print("copy", i, ", percentage: ",(((i+1)/numberOfKnots)*100))
            knot = CustomKnot(deepcopy(row["crosses"]))
            knot.randomMovN(numberOfRandomMov,maxStrands,percentage=True,debug=False)
            #Quitar cruces innecesarios y comprobar que no este en la base de datos.
            knot.reduceUnnecessaryMov()
            if not knot in knotSet:
                knotSet.add(knot)
                auxDict["name"].append(deepcopy(row["name"]))
                auxDict["crosses"].append(knot.crosses)
                i+=1

    return pd.DataFrame(auxDict)
