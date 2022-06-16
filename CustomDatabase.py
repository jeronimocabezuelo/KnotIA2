
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
    auxDict = {"name":[],"numberOfCrosses":[],"crosses":[]}
    names = knotNames(maxCrosses)
    for numberCrosses in names.keys():
        for name in names[numberCrosses]:
            auxDict["name"].append(name)
            auxDict["numberOfCrosses"].append(numberCrosses)
            knot = knotFromPyknotid(name)
            auxDict["crosses"].append(knot.crosses)

    return pd.DataFrame(auxDict)

def dataBase(maxCrosses: int, numberOfKnots: int, numberOfRandomMov: int,maxCrossesInRandomMov: int,debug:bool = False):
    df = singleDatabase(maxCrosses)
    df = df.reset_index()  # make sure indexes pair with number of rows
    auxDict = {"name":[],"numberOfCrosses":[],"crosses":[]}
    for index, row in df.iterrows():
        for i in range(numberOfKnots):
            #clear_output(wait=True)
            print("name", row["name"])
            print("copy", i)
            knot = CustomKnot(deepcopy(row["crosses"]))
            knot.randomMovN(numberOfRandomMov,maxCrossesInRandomMov,percentage=True,debug=debug)
            auxDict["name"].append(deepcopy(row["name"]))
            auxDict["numberOfCrosses"].append(copy(row["numberOfCrosses"]))
            auxDict["crosses"].append(knot.crosses)
    return pd.DataFrame(auxDict)

def randomMov(knot: CustomKnot, maxCrosses: int, visited: list[CustomKnot]):
    node = NodeKnot(knot)
    types = [1,2,3]
    shuffle(types)
    for type in types:
        if type == 1:
            createOrUndoList = [0,1]
            shuffle(createOrUndoList)
            for createOrUndo in createOrUndoList:
                if createOrUndo:
                    successors = list(node.successorsICreate(maxCrosses))
                    shuffle(successors)
                    for successor in successors:
                        if not successor in visited:
                            return successor.knot
                else:
                    successors = list(node.successorsIUndo())
                    shuffle(successors)
                    for successor in successors:
                        if not successor in visited:
                            return successor.knot
        elif type == 2:
            createOrUndoList = [0,1]
            shuffle(createOrUndoList)
            for createOrUndo in createOrUndoList:
                if createOrUndo:
                    successors = list(node.successorsIICreate(maxCrosses))
                    shuffle(successors)
                    for successor in successors:
                        if not successor in visited:
                            return successor.knot
                else:
                    successors = list(node.successorsIIUndo())
                    shuffle(successors)
                    for successor in successors:
                        if not successor in visited:
                            return successor.knot
        else:
            successors = list(node.successorsIII())
            shuffle(successors)
            for successor in successors:
                if not successor in visited:
                    return successor.knot

def randomMovN(knot: CustomKnot, n: int, maxCrosses: int,percentage = False):
    visited:list[CustomKnot] = [knot]
    i=1
    if percentage: startTime = time()
    while i <= n:
        if percentage:
                elapsedTime = time()-startTime
                percent = (((i+1)/n)*100)
                totalTime = elapsedTime*100/percent
                remainTime = totalTime-elapsedTime
                print("percentage randomMovN: {:3.2f}, time remaining: {}                                 ".format(percent,remainingTimeString(remainTime)),end="\r")
        newKnot = randomMov(visited[-1],maxCrosses,visited)
        if newKnot != None:
            visited.append(newKnot)
            i+=1
        else:
            visited.pop()
            i-=1
    return visited[-1]
