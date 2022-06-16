from __future__ import annotations
import numpy as np
from AuxiliarFunctions import *
from copy import copy
from typing import Tuple, TypeVar,Generic

from X import *

T = TypeVar('T')
class PriorityQueue(Generic[T]):
    """A queue with priorities, items can be inserted with .put(priority,item) and extracted with .get()"""
    def __init__(self):
        self.queue = {}
    def priorities(self)->list[int]:
        """Returns a list with the priorities."""
        return [priority for priority in self.queue.values()]
    #    return [element[0] for element in self.queue]
    def items(self)->list[T]:
        """Returns a list with the elements."""
        return [element for element in self.queue.keys()]#[element[1] for element in self.queue]
    def put(self,priority:int,item:T):
        """Inserts an element with its priority"""
        if not item in self.queue.keys() or self.queue[item]>priority:
            self.queue[item]=priority
        #if item in self.items():
        #    i = self.items().index(item)
        #    if self.priorities()[i]>priority:
        #        self.queue[i] = (priority,item)
        #else:
        #    self.queue.append((priority,item))
    def get(self)->T:
        """Gets an item.."""
        if len(self.queue) == 0:
            raise Exception("The queue has no elements.")
        mi = min(self.priorities())
        i = self.priorities().index(mi)
        element = self.items()[i]
        self.queue.pop(element)
        return element
        #if not self.queue:
        #    raise Exception("The queue has no elements.")
        #mi = min(self.priorities())
        #i = self.priorities().index(mi)
        #return self.queue.pop(i)[1]
    @property
    def isEmpty(self):
        """It tells us if the queue is empty."""
        return len(self.queue)==0
    @property
    def len(self):
        """It tells us the length of the queue."""
        return len(self.queue)

def distance(index1:Position,index2:Position)->int:
    """Calculates the distance between two indexes. Manhattan distance. l0"""
    return abs(index2[0]-index1[0])+abs(index2[1]-index1[1])

def isCornerOfPath(matrix:np.ndarray,ind:Position):
    """It tells us if an index is a corner of a path."""
    number = matrix[ind]
    for i in range(4):
        ind1 = uDLF(i,ind)
        ind2 = uDLF((i+1)%4,ind)
        if matrix[ind1] == number and matrix[ind2] == number:
            return True
    return False

class Node:
    def __init__(self,matrix:np.ndarray,origin:Position,destiny:Position,length:int,previousDirection:int|None = None,directionsChanges = 0):
        self.matrix = matrix
        self.origin = origin
        self.destiny = destiny
        self.length = length
        self.previousDirection = previousDirection
        self.directionsChanges = directionsChanges
    def successors(self,numberToFill:int,numberFree=0)->list[Node]:
        suc = []
        for i in range(4):
            newOrigin = uDLF(i,self.origin)
            newDirectionChanges = self.directionsChanges + (1 if self.previousDirection != i else 0)
            if exists(newOrigin,self.matrix) and self.matrix[newOrigin] == numberFree:
                newMatrix = copy(self.matrix)
                newMatrix[newOrigin] = numberToFill
                suc.append(Node(newMatrix,newOrigin,self.destiny,self.length+1,i,newDirectionChanges))
            if newOrigin == self.destiny:
                suc.append(Node(self.matrix,newOrigin,self.destiny,self.length,i,newDirectionChanges))
        return suc

#TODO: Mejorar la eficiencia haciendo de forma asíncrona la conexión desde origen a destino y desde destino a origen
def connect(matrix:np.ndarray,numberO:int,numberD:int,caminoNumber:int|None=None,corner:bool=False,numberFree:int = 0,oneOriginDestiny:bool = False,debug:bool=False):
    """Returns an array connecting numberO and numberD, the length of the path, and the indices of the origin and destination."""
    for origin in indicesOfNumberInMatrix(matrix,numberO):
        if corner:
            if isCornerOfPath(matrix,origin): 
                continue
        for destiny in [indice for indice in indicesOfNumberInMatrix(matrix,numberD) if indice != origin]:
            if corner:
                if isCornerOfPath(matrix,destiny):
                    continue
            l,m = connectOrigDest(matrix,origin,destiny,caminoNumber if caminoNumber != None else numberD,numberFree,debug=debug)
            if type(l)!= type(None):
                return l,m,origin,destiny
            elif oneOriginDestiny:
                return None,None,None,None
    return None,None,None,None

def connectOrigDest(matrix:np.ndarray,origin:Position,destiny:Position,caminoNumber,numberFree=0,debug=False)->tuple[int,np.ndarray]|tuple[None,None]:
    """Returns an array connecting origin and destiny and the length of the path."""
    if origin == destiny:
        return 0,matrix
    visited = []
    queue = PriorityQueue[Node]()
    cape = isCapeOfCross(matrix,origin)
    queue.put(distance(origin,destiny),Node(matrix,origin,destiny,0, None if cape==-1 else cape,0))
    while not queue.isEmpty:
        node = queue.get()
        if debug:
            print("-------")
            print(queue.len)
            print("newPadre")
            print(node.matrix)
            print("Sucesores:")
        for sucesor in node.successors(caminoNumber, numberFree = numberFree):
            if debug:
                print("visited:",visited)
                print("sucesor:")
                print(sucesor.origin)
                print(sucesor.matrix)
                print(sucesor.origin in visited)
            if not sucesor.origin in visited:
                d = distance(sucesor.origin,sucesor.destiny)
                if d == 0:
                    return sucesor.length,sucesor.matrix
                else:
                    visited.append(sucesor.origin)
                    queue.put(d+sucesor.length+sucesor.directionsChanges*2 ,sucesor)
    return None,None

def lengthPath(matrix:np.ndarray,numberO:int,numberD:int):
    """Returns the length of the path."""
    long,_,_,_ = connect(matrix,numberO,numberD)
    return long

def connectMatrix(matrix:np.ndarray,numberO:int,numberD:int):
    """Returns an array connecting numberO and numberD."""
    _,mat,_,_ = connect(matrix,numberO,numberD)
    return mat

def connectMatrixLength(matrix:np.ndarray,numberO:int,numberD:int):
    """Returns an array connecting numberO and numberD. And the length"""
    long,mat,_,_ = connect(matrix,numberO,numberD)
    return mat,long

def connectable(matrix:np.ndarray,numberO:int,numberD:int)->bool:
    """It tells us if it is possible to connect numberO and numberD."""
    m = connectMatrix(matrix,numberO,numberD)
    return type(m)!=type(None)

def connectableOrigDest(matrix:np.ndarray,origin:Position,destiny:Position,numberFree=0)->tuple[bool,int|None]:
    """It tells us if it is possible to connect origin and destiny."""
    l,m = connectOrigDest(matrix,origin,destiny,matrix[origin],numberFree)
    return type(m)!=type(None),l

def connected(pd:PlanarDiagram,strand:Strand):
    capes = [ind for ind in indicesOfNumberInMatrix(pd,strand) if isCapeOfCross(pd,ind) != -1]
    if len(capes)!=2:
        return False, None
    return connectableOrigDest(pd,capes[0],capes[1],numberFree=strand)