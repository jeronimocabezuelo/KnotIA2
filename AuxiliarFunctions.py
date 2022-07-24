from ast import Str
import numpy as np
from copy import copy
from typing import TypeVar

Coordinate = int
Position = tuple[Coordinate,Coordinate]
PlanarDiagram = np.ndarray

def mod(a:int,n:int)->int:
    """Like % but between 1 and n"""
    aux = a%n
    if aux == 0:
        return n
    return aux

def remap(oldValue,oldMin,oldMax,newMin,newMax):
    oldRange = (oldMax - oldMin)  
    newRange = (newMax - newMin)  
    newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin
    return newValue

def borderByZeros(matrix:np.ndarray)->np.ndarray:
    """Borders a matrix with zeros."""
    if not np.array_equal(matrix[0,:] , np.zeros(matrix.shape[1])):
        matrix = np.insert(matrix,0,0,axis=0)
    if not np.array_equal(matrix[-1,:], np.zeros(matrix.shape[1])):
        matrix = np.insert(matrix,matrix.shape[0],0,axis=0)
    if not np.array_equal(matrix[:,0] , np.zeros(matrix.shape[0])):
        matrix = np.insert(matrix,0,0,axis=1)
    if not np.array_equal(matrix[:,-1], np.zeros(matrix.shape[0])):
        matrix = np.insert(matrix,matrix.shape[1],0,axis=1)
    return matrix

def removeBorderOfZeros(matrix:np.ndarray):
    """Removes borders of zeros from an matrix."""
    while True:
        if np.array_equal(matrix[0,:] , np.zeros(matrix.shape[1])):
            matrix = matrix[1:,:]
            continue
        if np.array_equal(matrix[-1,:], np.zeros(matrix.shape[1])):
            matrix = matrix[:-1,:]
            continue
        if np.array_equal(matrix[:,0] , np.zeros(matrix.shape[0])):
            matrix = matrix[:,1:]
            continue
        if np.array_equal(matrix[:,-1] , np.zeros(matrix.shape[0])):
            matrix = matrix[:,:-1]
            continue
        return matrix

def exists(index:Position,matrix:np.ndarray):
    """Tells us if an index is valid for a matrix."""
    return index[0]>-1 and index[1]>-1 and index[0]< matrix.shape[0] and index[1]< matrix.shape[1]

def up(ind:Position)->Position:
    return(ind[0]-1,ind[1])

def down(ind:Position)->Position:
    return(ind[0]+1,ind[1])

def left(ind:Position)->Position:
    return(ind[0],ind[1]-1)

def right(ind:Position)->Position:
    return(ind[0],ind[1]+1)

def uDLF(i:int,ind:Position)->Position:
    """Returns the index to the above, left, below, or right of another."""
    if i%4 == 0:
        return up(ind)
    elif i%4 == 1:
        return right(ind)
    elif i%4 == 2:
        return down(ind)
    else :
        return left(ind)

def indicesOfNumberInMatrix(matrix:np.ndarray,number:int)->list[Position]:
    """It gives us all the index of a matrix that has the number."""
    return [(indice[0],indice[1]) for indice in np.argwhere(matrix==number)]

def concat(matrix1:np.ndarray,matrix2:np.ndarray,axis=1)->np.ndarray:
    """Does the same as numpy's concatenate function but for two arrays and also works if one of the arrays has shape (0,0)"""
    if matrix1.shape[axis] == 0:
        return matrix2
    if matrix2.shape[axis] == 0:
        return matrix1
    return np.concatenate((matrix1,matrix2),axis=axis)

def insert(array:np.ndarray,obj,values,n=1,axis=1)->np.ndarray:
    """Like numpy's insert but for n times."""
    for _ in range(n):
        array = np.insert(array,obj,values,axis=axis)
    return array

def contains(matrix:np.ndarray,element):
    """It tells us if an element is in an array."""
    return np.any(np.isin(matrix,element))

def replace(matrix:np.ndarray,oldNumber,newNumber)->np.ndarray:
    """Replaces all occurrences of one value with another in an array."""
    return np.where(matrix==oldNumber, newNumber, matrix)

def createGaps(matrix:np.ndarray)->np.ndarray:
    """Separate the strands so that more paths enter."""
    m2 = np.zeros((matrix.shape[0]*2-1,matrix.shape[1]),dtype=int)
    for i in range(matrix.shape[0]):
        m2[i*2,:] = matrix[i,:]
        m3 = np.zeros((m2.shape[0],m2.shape[1]*2-1),dtype=int)
    for j in range(m2.shape[1]):
        m3[:,j*2] = m2[:,j]
        m3copy = copy(m3)
    for i in range(m3.shape[0]):
        for j in range(m3.shape[1]):
            if m3copy[i,j]==0:
                if i>0 and i<m3.shape[0]-1:
                    if m3[i-1,j]<0:
                        m3[i,j] = m3[i+1,j]
                        if not (m3[i+3,j] == m3[i,j] or m3[i+1,j+2] == m3[i,j] or m3[i+1,j-2] == m3[i,j]):
                            m3[i+1,j] = 0
                        continue
                    elif m3[i+1,j]<0:
                        m3[i,j] = m3[i-1,j]
                        if not (m3[i-3,j] == m3[i,j] or m3[i-1,j+2] == m3[i,j] or m3[i-1,j-2] == m3[i,j]):
                            m3[i-1,j] = 0
                        continue
                    elif m3[i-1,j] == m3[i+1,j] and m3[i+1,j]!=0:
                        m3[i,j] = m3[i-1,j]
                        continue
                if j>0 and j<m3.shape[1]-1:
                    if m3[i,j-1]<0:
                        m3[i,j] = m3[i,j+1]
                        if not (m3[i,j+3] == m3[i,j] or m3[i+2,j+1] == m3[i,j] or m3[i-2,j+1] == m3[i,j]):
                            m3[i,j+1] = 0
                        continue
                    elif m3[i,j+1]<0:
                        m3[i,j] = m3[i,j-1]
                        if not (m3[i,j-3] == m3[i,j] or m3[i+2,j-1] == m3[i,j] or m3[i-2,j-1] == m3[i,j]):
                            m3[i,j-1] = 0
                        continue
                    elif m3[i,j-1] == m3[i,j+1] and m3[i,j+1]!=0:
                        m3[i,j] = m3[i,j-1]
                        continue
    return m3


#TODO: Hacer mas corto eso con UDLF
def walkWithDirection(matrix:np.ndarray,org:Position,des:Position,dir:int):
    """Walk a path of -9 following the direction"""
    prevDirection = None
    visited:list[Position]=[]
    while org != des:
        visited.append(org)
        if not up(org) in visited:
            if exists(up(org),matrix):
                if matrix[up(org)] == -9 or up(org) == des:
                    org = up(org)
                    if prevDirection == 1:
                        dir = (dir-1)%4
                    elif prevDirection == 3:
                        dir = (dir+1)%4
                    prevDirection = 0
                    continue
        if not down(org) in visited:
            if exists(down(org),matrix):
                if matrix[down(org)] == -9 or down(org) == des:
                    org = down(org)
                    if prevDirection == 1:
                            dir = (dir+1)%4
                    elif prevDirection == 3:
                            dir = (dir-1)%4
                    prevDirection = 2
                    continue
        if not left(org) in visited:
            if exists(left(org),matrix):
                if matrix[left(org)] == -9 or left(org) == des:
                    org = left(org)
                    if prevDirection == 0:
                        dir = (dir-1)%4
                    elif prevDirection == 2:
                        dir = (dir+1)%4
                    prevDirection = 3
                    continue
        if not right(org) in visited:
            if exists(right(org),matrix):
                if matrix[right(org)] == -9 or right(org) == des:
                    org = right(org)
                    if prevDirection == 0:
                        dir = (dir+1)%4
                    elif prevDirection == 2:
                        dir = (dir-1)%4
                    prevDirection = 1
                    continue       
    return visited,dir

def findSubMatrix3x3In(matrix:np.ndarray,subMatrix3x3:np.ndarray):
    """Find a 3x3 submatrix inside another."""
    centers = indicesOfNumberInMatrix(matrix,subMatrix3x3[1][1])
    for center in centers:
        if matrix[up(center)]==subMatrix3x3[0][1] and matrix[right(center)]==subMatrix3x3[1][2] and matrix[down(center)]==subMatrix3x3[2][1] and matrix[left(center)]==subMatrix3x3[1][0]:
            return center
    return None
  


def removeUnnecessaryRow(pd:PlanarDiagram)->PlanarDiagram:
    """Delete unnecessary rows in a planar diagram."""
    for row in reversed(range(pd.shape[0])):
        toRemove = True
        if (row>0 and contains(pd[row-1],[-1,-2])) or (row<pd.shape[0]-1 and contains(pd[row+1],[-1,-2])):
            continue
        for c in range(pd.shape[1]-1):
            if pd[row][c] < 0:
                toRemove = False
            if pd[row][c] == pd[row][c+1] and pd[row][c] != 0:
                toRemove = False
            if (row>0 and contains(pd[row-1],[-1,-2])) or (row<pd.shape[0]-1 and contains(pd[row+1],[-1,-2])):
                toRemove = False
        if toRemove:
            pd = np.delete(pd,row,axis=0)
    return pd

def removeUnnecessaryColumn(pd:PlanarDiagram)->PlanarDiagram:
    """Delete unnecessary Column in a planar diagram."""
    for c in reversed(range(pd.shape[1])):
        toRemove = True
        if (c>0 and contains(pd[:,c-1],[-1,-2])) or (c<pd.shape[1]-1 and contains(pd[:,c+1],[-1,-2])):
            continue
        for row in range(pd.shape[0]-1):
            if pd[row][c] < 0:
                toRemove = False
                break
            if pd[row][c] == pd[row+1][c] and pd[row][c] != 0:
                toRemove = False
                break
            if (c>0 and contains(pd[:,c-1],[-1,-2])) or (c<pd.shape[1]-1 and contains(pd[:,c+1],[-1,-2])):
                toRemove = False
        if toRemove:
            pd = np.delete(pd,c,axis=1)
    return pd

def isCapeOfCross(pd:PlanarDiagram,ind:Position):
    for i in range(4):
        if exists(uDLF(i,ind),pd) and pd[uDLF(i,ind)]<0:
            return i
    return -1

def remainingTimeString(time:float)->str:
    if time<1:
        return "<1s"
    time = int(time)
    if time<60:
        return str(time)+"s"
    minutes = time//60
    second = time%60    
    if minutes<60:
        return str(minutes)+"min "+str(second)+"s"
    hours = minutes//60
    minutes = minutes%60
    return str(hours)+"h "+str(minutes)+"min "+str(second)+"s"

_T = TypeVar('_T')

class sset(set):
    def add(self, __element: _T) -> bool:
        prev_len = len(self)
        super().add(__element) # O(N log N) lookup
        return len(self) != prev_len