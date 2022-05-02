
import pandas as pd
from CustomKnot import *
from IPython.display import clear_output
from copy import copy,deepcopy

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
    auxDict = {"name":[],"numberOfCrosses":[],"crosses":[],"pd":[]}
    names = knotNames(maxCrosses)
    for numberCrosses in names.keys():
        for name in names[numberCrosses]:
            auxDict["name"].append(name)
            auxDict["numberOfCrosses"].append(numberCrosses)
            knot = knotFromPyknotid(name)
            auxDict["crosses"].append(knot.crosses)
            auxDict["pd"].append(knot.planarDiagrams())

    return pd.DataFrame(auxDict)

def dataBase(maxCrosses: int, numberOfKnots: int, numberOfRandomMov: int,maxCrossesInRandomMov: int):
    df = singleDatabase(maxCrosses)
    df = df.reset_index()  # make sure indexes pair with number of rows
    auxDict = {"name":[],"numberOfCrosses":[],"crosses":[],"pd":[]}
    for index, row in df.iterrows():
        for i in range(numberOfKnots):
            clear_output(wait=True)
            print("name", row["name"])
            print("copy", i)
            knot = CustomKnot(deepcopy(row["crosses"]))
            knot.pd = row["pd"]
            randomMovN(knot,numberOfRandomMov,maxCrossesInRandomMov,percentage=True,debug=True)
            auxDict["name"].append(deepcopy(row["name"]))
            auxDict["numberOfCrosses"].append(copy(row["numberOfCrosses"]))
            auxDict["crosses"].append(knot.crosses)
            auxDict["pd"].append(repr(knot.planarDiagrams()))
    return pd.DataFrame(auxDict)

