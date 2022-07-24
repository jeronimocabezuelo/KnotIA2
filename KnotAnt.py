from __future__ import annotations
from cmath import inf
from CustomKnot import *
from random import *
from enum import Enum
import numpy as np
from KnotStar import Similarity
class ReidemeisterType(Enum):
    CreateALoop = 0
    UndoALoop = 1
    CreateII = 2
    UndoII = 3
    III = 4


class KnotAnt:
    def __init__(self,house: CustomKnot, objetive:CustomKnot):
        self.house = house
        self.objetive = objetive
        self.actual = deepcopy(house)
        self.movs:list[str] = []
    
    def randomMov(self,maxCrosses=100):
        b,mov = self.actual.randomMov(self.objetive.numberOfStrands+4)
        if b:
            self.movs.append(mov)

    def check(self) -> bool:
        return self.actual == self.objetive

class Colony:
    def __init__(self,house: CustomKnot, objetive:CustomKnot,numberOfAnts = 100):
        self.ants:list[KnotAnt] = []
        for _ in range(numberOfAnts):
            self.ants.append(KnotAnt(house,objetive))
        self.similarities = Similarity(objetive)
    
    def randomMov(self, maxCrosses=100):
        for ant in self.ants:
            ant.randomMov(maxCrosses)
    
    def check(self) -> KnotAnt | None:
        maxSimilarity = 0
        minSimilarity = inf
        for ant in self.ants:
            s = self.similarities[ant.actual]
            maxSimilarity = max(maxSimilarity,s)
            minSimilarity = min(minSimilarity,s)
            #print("crosses of ant", len(ant.actual.crosses))
            if ant.check():
                return ant
        print("maxSimilarity: ", maxSimilarity)
        print("minSimilarity: ", minSimilarity)
        return None

def areSameKnotAnt(knot1: CustomKnot,knot2: CustomKnot,numberOfAnts=100,maxCrosses= 100,limit=1000) -> Tuple[bool,list[str]]:
    colony = Colony(knot1,knot2,numberOfAnts)
    c=0
    while c<limit:
        print(c)
        c+=1
        ant = colony.check()
        if ant != None:
            return True, ant.movs
        colony.randomMov(maxCrosses)
    return False, []
        