from __future__ import annotations
from cmath import inf
from CustomKnot import *
from random import *
from enum import Enum
import numpy as np
from KnotStar import DifferenceSimpleCache
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
        self.movs:List[str] = []
    
    def randomMov(self,maxCrosses=100):
        b,mov = self.actual.randomMov(self.objetive.numberOfStrands+4)
        if b:
            self.movs.append(mov)

    def check(self) -> bool:
        return self.actual == self.objetive

class Colony:
    def __init__(self,house: CustomKnot, objetive:CustomKnot,numberOfAnts = 100):
        self.ants:List[KnotAnt] = []
        for _ in range(numberOfAnts):
            self.ants.append(KnotAnt(house,objetive))
        self.differences = DifferenceSimpleCache(objetive)
    
    def randomMov(self, maxCrosses=100):
        for ant in self.ants:
            ant.randomMov(maxCrosses)
    
    def check(self) -> KnotAnt | None:
        maxDifference = 0
        minDifference = inf
        for ant in self.ants:
            s = self.differences[ant.actual]
            maxDifference = max(maxDifference,s)
            minDifference = min(minDifference,s)
            #print("crosses of ant", len(ant.actual.crosses))
            if ant.check():
                return ant
        print("maxDifference: ", maxDifference)
        print("minDifference: ", minDifference)
        return None

def areSameKnotAnt(knot1: CustomKnot,knot2: CustomKnot,numberOfAnts=100,maxCrosses= 100,limit=1000) -> Tuple[bool,List[str]]:
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
        