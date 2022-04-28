import numpy as np
from AuxiliarFunctions import *
from copy import copy

class PriorityQueue:
    """A queue with priorities, items can be inserted with .put(priority,item) and extracted with .get()"""
    def __init__(self):
        self.queue = []
    def priorities(self):
        """Returns a list with the priorities."""
        return [element[0] for element in self.queue]
    def items(self):
        """Returns a list with the elements."""
        return [element[1] for element in self.queue]
    def put(self,priority,item):
        """Inserts an element with its priority"""
        if item in self.items():
            i = self.items().index(item)
            if self.priorities()[i]>priority:
                self.queue[i] = (priority,item)
        else:
            self.queue.append((priority,item))
    def get(self):
        """Gets an item.."""
        if not self.queue:
            raise Exception("The queue has no elements.")
        mi = min(self.priorities())
        i = self.priorities().index(mi)
        return self.queue.pop(i)[1]
    @property
    def empty(self):
        """It tells us if the queue is empty."""
        return not self.queue
    @property
    def len(self):
        """It tells us the length of the queue."""
        return len(self.queue)

def distance(index1,index2):
    """Calculates the distance between two indexes. Manhattan distance. l0"""
    return abs(index2[0]-index1[0])+abs(index2[1]-index1[1])

def isCornerOfPath(matrix,ind):
    """It tells us if an index is a corner of a path."""
    number = matrix[ind]
    for i in range(4):
        ind1 = uDLF(i,ind)
        ind2 = uDLF((i+1)%4,ind)
        if matrix[ind1] == number and matrix[ind2] == number:
            return True
    return False

class Node:
    def __init__(self,matrix,origin,destin,length,previousDirection,directionsChanges):
        self.matrix = matrix
        self.origin = origin
        self.destin = destin
        self.length = length
        self.previousDirection = previousDirection
        self.directionsChanges = directionsChanges
    def successors(self,numberToFill,numberFree=0):
        suc = []
        for i in range(4):
            newOrigin = uDLF(i,self.origin)
            newDirectionChanges = self.directionsChanges + (1 if self.previousDirection != i else 0)
            if exists(newOrigin,self.matrix) and self.matrix[newOrigin] == numberFree:
                newMatrix = copy(self.matrix)
                newMatrix[newOrigin] = numberToFill
                suc.append(Node(newMatrix,newOrigin,self.destin,self.length+1,i,newDirectionChanges))
            if newOrigin == self.destin:
                suc.append(Node(self.matrix,newOrigin,self.destin,self.length,i,newDirectionChanges))
        return suc

#TODO: Mejorar la eficiencia haciendo de forma sincrona la conexion desde origen a destino y desde destino a origen
#TODO: Mejorar el camino añadiendo una penalizacion por cambios de sentido
def connect(matrix,numberO,numberD,caminoNumber=None,corner=False,numberFree = 0,oneOriginDestin = False,debug=False):
    """Returns an array connecting numberO and numberD, the length of the path, and the indices of the origin and destination."""
    for origin in indicesOfNumberInMatrix(matrix,numberO):
        if corner:
            if isCornerOfPath(matrix,origin): 
                continue
        for destin in [indice for indice in indicesOfNumberInMatrix(matrix,numberD) if indice != origin]:
            if corner:
                if isCornerOfPath(matrix,destin):
                    continue
            l,m = connectOrigDest(matrix,origin,destin,caminoNumber if caminoNumber != None else numberD,numberFree,debug=debug)
            if type(l)!= type(None):
                return l,m,origin,destin
            elif oneOriginDestin:
                return None,None,None,None
    return None,None,None,None

def connectOrigDest(matrix,origin,destin,caminoNumber,numberFree=0,debug=False):
    """Returns an array connecting origin and destin and the length of the path."""
    if origin == destin:
        return 0,matrix
    visited = []
    queue = PriorityQueue()
    cape = isCapeOfCross(matrix,origin)
    queue.put(distance(origin,destin),Node(matrix,origin,destin,0, None if cape==-1 else cape,0))
    while not queue.empty:
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
                d = distance(sucesor.origin,sucesor.destin)
                if d == 0:
                    return sucesor.length,sucesor.matrix
                else:
                    visited.append(sucesor.origin)
                    queue.put(d+sucesor.length+sucesor.directionsChanges*2 ,sucesor)
    return None,None

def lengthPath(matrix,numberO,numberD):
    """Returns the length of the path."""
    long,_,_,_ = connect(matrix,numberO,numberD)
    return long

def connectMatrix(matrix,numberO,numberD):
    """Returns an array connecting numberO and numberD."""
    _,mat,_,_ = connect(matrix,numberO,numberD)
    return mat

def connectMatrixLength(matrix,numberO,numberD):
    """Returns an array connecting numberO and numberD. And the length"""
    long,mat,_,_ = connect(matrix,numberO,numberD)
    return mat,long

def connectable(matrix,numberO,numberD):
    """It tells us if it is possible to connect numberO and numberD."""
    m = connectMatrix(matrix,numberO,numberD)
    return type(m)!=type(None)

def connectableOrigDest(matrix,origin,destin,numberFree=0):
    """It tells us if it is possible to connect origin and destin."""
    l,m = connectOrigDest(matrix,origin,destin,matrix[origin],numberFree)
    return type(m)!=type(None),l

def connected(pd,strand):
    capes = [ind for ind in indicesOfNumberInMatrix(pd,strand) if isCapeOfCross(pd,ind) != -1]
    if len(capes)!=2:
        return False, None
    return connectableOrigDest(pd,capes[0],capes[1],numberFree=strand)