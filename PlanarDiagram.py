from __future__ import annotations
from ast import Str
import numpy as np
from copy import copy,deepcopy
from AuxiliarFunctions import *
from Path import *
from X import *

def directionOfStrandWhenAdd(matrix:PlanarDiagram,ind:Position):
    for i in range(4):
        if exists(uDLF(i,ind),matrix) and matrix[uDLF(i,ind)]<0:
            return (i+2)%4

def addThreeRowOrColumn(matrix:PlanarDiagram,d:int,ind:Position)->tuple[PlanarDiagram,Position]:
    if d%4==0:
        return np.append(np.zeros((3,matrix.shape[1]),dtype=int),matrix,axis=0),(ind[0]+3,ind[1])
    elif d%4==1:
        return np.append(matrix,np.zeros((matrix.shape[0],3),dtype=int),axis=1), ind
    elif d%4==2:
        return np.append(matrix,np.zeros((3,matrix.shape[1]),dtype=int),axis=0), ind
    else:
        return np.append(np.zeros((matrix.shape[0],3),dtype=int),matrix,axis=1),(ind[0],ind[1]+3)

def indicesNeedFree(ind:Position,d:int):
    aux = [uDLF(d,ind),uDLF(d,uDLF(d,ind)),uDLF(d,uDLF(d,uDLF(d,ind)))]
    aux1 = [uDLF(d+1,indAux) for indAux in aux]
    aux2 = [uDLF(d-1,indAux) for indAux in aux]
    return aux+aux1+aux2

def insertThreeRowOrColumn(matrix:PlanarDiagram,d:int,ind:Position):
    unconnectedStrand = set([matrix[auxInd] for auxInd in indicesNeedFree(ind,d) if matrix[auxInd]>0])
    if d==0:
        matrix = insert(matrix,ind[0],0,3,axis=0)
        ind = (ind[0]+3,ind[1])
    elif d==1:
        matrix = insert(matrix,ind[1]+1,0,3,axis=1)
    elif d==2:
        matrix = insert(matrix,ind[0]+1,0,3,axis=0)
    elif d==3:
        matrix = insert(matrix,ind[1],0,3,axis=1)
        ind = (ind[0],ind[1]+3)

    for i in range(matrix.shape[(d+1)%2]):
        baseInd = (ind[0],i) if d%2 == 0 else (i,ind[1])
        base1 = uDLF(d,baseInd)
        base2 = uDLF(d,base1)
        base3 = uDLF(d,base2)
        base4 = uDLF(d,base3)
        if matrix[baseInd] == matrix[base4] and matrix[base4]>0:
            matrix[base1] = matrix[baseInd]
            matrix[base2] = matrix[baseInd]
            matrix[base3] = matrix[baseInd]
        # if (matrix[baseInd]>0 and matrix[base4]<0) or (matrix[baseInd]<0 and matrix[base4]>0):
        #     matrix[base1] = max(matrix[baseInd],matrix[base4])
        #     matrix[base2] = max(matrix[baseInd],matrix[base4])
        #     matrix[base3] = max(matrix[baseInd],matrix[base4])
        if matrix[baseInd]<0:
            matrix[base1] = matrix[base4]
            unconnectedStrand.add(matrix[base1])
        elif matrix[base4]<0:
            matrix[base3] = matrix[baseInd] 
            unconnectedStrand.add(matrix[base3])

    return matrix,ind,unconnectedStrand      

def indicesOfCross(ind:Position,d:int):
    return  [uDLF(d,ind),
             uDLF(d,uDLF(d+1,uDLF(d,ind))),
             uDLF(d-1,uDLF(d,uDLF(d,uDLF(d+1,uDLF(d,ind))))),
             uDLF(d-2,uDLF(d-1,uDLF(d-1,uDLF(d,uDLF(d,uDLF(d+1,uDLF(d,ind)))))))]

def indicesFreeBeforeAdd(ind:Position,d:int):
    return [uDLF(d+1,uDLF(d,ind)),
            uDLF(d-1,uDLF(d,ind)),
            uDLF(d+1,uDLF(d,uDLF(d,uDLF(d,ind)))),
            uDLF(d-1,uDLF(d,uDLF(d,uDLF(d,ind))))]

def addCrossToMatrix(matrix:PlanarDiagram,cross:X,ind:Position):
    #print("     addCrossToMatrix:",cross)
    partsUnconnected = set()
    d = directionOfStrandWhenAdd(matrix,ind)
    if not exists(uDLF(d,uDLF(d,(uDLF(d,ind)))),matrix):
        matrix,ind = addThreeRowOrColumn(matrix,d,ind)
    if any([matrix[auxInd]!=0 for auxInd in indicesNeedFree(ind,d)]):
        matrix,ind,partsUnconnected = insertThreeRowOrColumn(matrix,d,ind)
    s = cross.indexForStrand(matrix[ind])[0]
    isBelow = s%2 == 0
    isUpDown = d%2 == 0
    matrix[uDLF(d,uDLF(d,ind))] = (-1 if isUpDown else -2) if isBelow else (-2 if isUpDown else -1)
    indicesOfC = indicesOfCross(ind,d)
    for i in range(4):
        matrix[indicesOfC[i]] = cross[s+i]
    for i in indicesFreeBeforeAdd(ind,d):
        if matrix[i] != 0:
            l = matrix[i]
            partsUnconnected.add(l)
    for l in partsUnconnected:
        for j in indicesOfNumberInMatrix(matrix,l):
            if isCapeOfCross(matrix,j) == -1:
                matrix[j] = 0
    return matrix,list(partsUnconnected)

class NodePD:
    def __init__(self,pd:PlanarDiagram,unconnectedStrands:list[Strand],remainCross:list[X],lengths:dict[Strand,int]):
        self.pd:PlanarDiagram = borderByZeros(pd)
        self.unconnectedStrands = unconnectedStrands
        self.lengths = lengths
        self.remainCross = remainCross
    def successors(self,debug=False)->list[NodePD]:
        if debug:
            print("-------")
            print("Padre:")
            print(self.unconnectedStrands)
            print(self.pd)
        successorsArray  = []
        for strand in self.unconnectedStrands:
            indices = indicesOfNumberInMatrix(self.pd,strand)
            if len(indices) == 0:
                continue
            elif len(indices) == 1:
                crossCopy = deepcopy(self.remainCross)
                cross = crossCopy.pop(crossWithStrand(crossCopy,strand))
                ind = indices[0]
                newPd = deepcopy(self.pd)
                newDict = dict(self.lengths)
                newPd,partDisconnected = addCrossToMatrix(newPd,cross,ind)
                if debug:
                    print("Hijo: ",strand)
                    print(newPd)
                newUnconnected = [s for s in self.unconnectedStrands if s!=strand]
                for l in partDisconnected:
                    newUnconnected.append(l)
                    if l in newDict.keys():
                        newDict.pop(l)
                successorsArray.append(NodePD(newPd,newUnconnected,crossCopy,newDict))
            else:
                gapsCreated = False
                length,matrixConnected,_,_ = connect(self.pd,strand,strand)
                if type(matrixConnected) == type(None):
                    gapsCreated = True
                    length,matrixConnected,_,_  = connect(createGaps(self.pd),strand,strand)
                if type(matrixConnected) == type(None):
                    if debug: print("Descartamos padre")
                    return []
                newDict = dict(self.lengths)
                newDict[strand] = length
                if debug:
                    print("Hijo: ",strand)
                    print(matrixConnected)
                if gapsCreated: matrixConnected,newDict = compactPlanarDiagram(matrixConnected,newDict)
                successorsArray .append(NodePD(matrixConnected,[s for s in self.unconnectedStrands if s != strand],deepcopy(self.remainCross),newDict))
        successorsArray .sort(key=lambda x: x.length())
        return successorsArray 
    def priority(self):
        return len(self.unconnectedStrands)+len(self.remainCross)
    def length(self):
        return sum(self.lengths.values())

def reconnect(pd:PlanarDiagram,strand:Strand):
    """Delete the path from the strand, and reconnect it."""
    indices = indicesOfNumberInMatrix(pd,strand)
    for ind in indices:
        if not any([pd[uDLF(i,ind)]<0 for i in [0,1,2,3] if exists(uDLF(i,ind),pd)]):
            pd[ind] = 0
    return connectMatrixLength(pd,strand,strand)

def compactPlanarDiagram(pd:PlanarDiagram,lengths:dict[Strand,int],debug=False):
    if debug:
        print("compactPlanarDiagram")
        print(lengths)
        print(pd)
    previousShape = pd.shape
    c=0
    while True and c<100:
        c+=1
        lengthsSorted = dict(sorted(lengths.items(), key=lambda item: item[1])).keys()
        pd = removeUnnecessaryRow(pd)
        pd = removeUnnecessaryColumn(pd)
        pd = borderByZeros(pd)
        for strand in lengthsSorted:
            if debug:
                print("Antes de reconectar",strand)
                print(pd)
            pd,l = reconnect(pd,strand)
            lengths[strand] = l
            if debug:
                print("después de reconectar")
                print(pd)
        pd = borderByZeros(pd)
        if pd.shape == previousShape:
            break
        previousShape = pd.shape
    return pd,lengths