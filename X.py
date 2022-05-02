import numpy as np
from AuxiliarFunctions import *

class X:
    def __init__(self,x1,x2,x3,x4):
        if x1<=x3:
            self.strands = [x1,x2,x3,x4]
        else:
            self.strands = [x3,x4,x1,x2]
    def indexForStrand(self,l):
        for i in range(len(self.strands)):
            if l == self.strands[i]:
                return i
        return None
    def __repr__(self):
        return "X("+str(self.strands[0])+","+str(self.strands[1])+","+str(self.strands[2])+","+str(self.strands[3])+")"
    def __eq__(self,obj):
        if type(self)!=type(obj):
            return False
        c1 = self.strands[0] == obj.strands[0] and self.strands[1] == obj.strands[1] and self.strands[2] == obj.strands[2] and self.strands[3] == obj.strands[3]
        c2 = self.strands[0] == obj.strands[2] and self.strands[1] == obj.strands[3] and self.strands[2] == obj.strands[0] and self.strands[3] == obj.strands[1]
        return c1 or c2
    def __contains__(self,key):
        return key in self.strands  
    def isStrandAbove(self,strand):
        for i in range(len(self.strands)):
            if self.strands[i] == strand:
                return i%2 == 1
    def __hash__(self):
        return hash(self.__repr__())
    def __lt__(self,other):
        if type(self) != type(other):
            return False
        return self.strands[0] < other.strands[0]
    def __getitem__(self, item):
        return self.strands[item%4]

def crossWithStrand(cross,strand):
    for i in range(len(cross)):
        if strand in cross[i].strands:
            return i
    return None

def zonesConnectCross(cross,planarDiagram,pZones):
    ind = findCrossInPd(cross,planarDiagram)
    if ind==None:
        print("Este cruce no esta en este diagrama plano")
        return None
    return [pZones[up(left(ind))],pZones[up(right(ind))],pZones[down(left(ind))],pZones[down(right(ind))]]

def commonZone(xs,planarDiagram,planarZones):
    zs = [zonesConnectCross(x,planarDiagram,planarZones) for x in xs]
    return list(set(zs[0]).intersection(*zs[1:]).difference([1]))

def crossesWithZone(knot,planarDiagram,planarZones,zone):
    return [cross for cross in knot.crosses if zone in zonesConnectCross(cross,planarDiagram,planarZones)]

def crossesWithStrand(knot,strand):
    return [cross for cross in knot.crosses if strand in cross]











