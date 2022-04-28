import numpy as np
from copy import copy

def mod(a,n):
    """Like % but between 1 and n"""
    aux = a%n
    if aux == 0:
        return n
    return aux

def borderByZeros(matrix):
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

def removeBorderOfZeros(matrix):
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

def exists(index,matrix):
    """Tells us if an index is valid for a matrix."""
    return index[0]>-1 and index[1]>-1 and index[0]< matrix.shape[0] and index[1]< matrix.shape[1]

def up(ind):
    return(ind[0]-1,ind[1])

def down(ind):
    return(ind[0]+1,ind[1])

def left(ind):
    return(ind[0],ind[1]-1)

def right(ind):
    return(ind[0],ind[1]+1)

def uDLF(i,ind):
    """Returns the index to the above, left, below, or right of another."""
    if i%4 == 0:
        return up(ind)
    elif i%4 == 1:
        return right(ind)
    elif i%4 == 2:
        return down(ind)
    else :
        return left(ind)

def indicesOfNumberInMatrix(matrix,number):
    """It gives us all the index of a matrix that has the number."""
    return [(indice[0],indice[1]) for indice in np.argwhere(matrix==number)]

def concat(matrix1,matrix2,axis=1):
    """Does the same as numpy's concatenate function but for two arrays and also works if one of the arrays has shape (0,0)"""
    if matrix1.shape[axis] == 0:
        return matrix2
    if matrix2.shape[axis] == 0:
        return matrix1
    return np.concatenate((matrix1,matrix2),axis=axis)

def insert(array,obj,values,n=1,axis=1):
    """Like numpy's insert but for n times."""
    for _ in range(n):
        array = np.insert(array,obj,values,axis=axis)
    return array

def contains(matrix,element):
    """It tells us if an element is in an array."""
    return np.any(np.isin(matrix,element))

def replace(matrix,oldNumber,newNumber):
    """Replaces all occurrences of one value with another in an array."""
    return np.where(matrix==oldNumber, newNumber, matrix)

def createGaps(matrix):
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

def direction(knot,pd,ind):
    """It will return 0, 1, 2 or 3 depending on the direction the strand is going."""
    number = pd[ind]
    if number <=0:
        return None
    directions = []
    visited = [ind]
    i = 0
    c = 0
    while True and c<5:
        if not uDLF(i,ind) in visited:
            if exists(uDLF(i,ind),pd):
                if pd[uDLF(i,ind)] == number or pd[uDLF(i,ind)] < 0:
                    ind = uDLF(i,ind)
                    visited.append(ind)
                    directions.append(i)
                    if pd[ind] < 0:          
                        break
                    else:
                        c=0
        i = (i+1)%4
        c += 1
    lastD = directions[-1]
    firstD = directions[0]
    lastInd = visited[-1]
    if number == 1:
        condition = pd[uDLF(lastD,lastInd)] == knot.numberOfStrands
    elif number == knot.numberOfStrands:
        condition = pd[uDLF(lastD,lastInd)] == knot.numberOfStrands-1
    else:
        condition = pd[uDLF(lastD,lastInd)] < number
    if condition:
        return (firstD+2)%4
    return firstD
#TODO: Hacer mas corto eso con UDLF
def walkWithDirection(matrix,org,des,dir):
    """Walk a path of -9 following the direction"""
    prevDirecion = None
    visited=[]
    while org != des:
        visited.append(org)
        if not up(org) in visited:
            if exists(up(org),matrix):
                if matrix[up(org)] == -9 or up(org) == des:
                    org = up(org)
                    if prevDirecion == 1:
                        dir = (dir-1)%4
                    elif prevDirecion == 3:
                        dir = (dir+1)%4
                    prevDirecion = 0
                    continue
        if not down(org) in visited:
            if exists(down(org),matrix):
                if matrix[down(org)] == -9 or down(org) == des:
                    org = down(org)
                    if prevDirecion == 1:
                            dir = (dir+1)%4
                    elif prevDirecion == 3:
                            dir = (dir-1)%4
                    prevDirecion = 2
                    continue
        if not left(org) in visited:
            if exists(left(org),matrix):
                if matrix[left(org)] == -9 or left(org) == des:
                    org = left(org)
                    if prevDirecion == 0:
                        dir = (dir-1)%4
                    elif prevDirecion == 2:
                        dir = (dir+1)%4
                    prevDirecion = 3
                    continue
        if not right(org) in visited:
            if exists(right(org),matrix):
                if matrix[right(org)] == -9 or right(org) == des:
                    org = right(org)
                    if prevDirecion == 0:
                        dir = (dir+1)%4
                    elif prevDirecion == 2:
                        dir = (dir-1)%4
                    prevDirecion = 1
                    continue       
    return visited,dir

def findSubMatrix3x3In(matrix,subMatrix3x3):
    """Find a 3x3 submatrix inside another."""
    centers = indicesOfNumberInMatrix(matrix,subMatrix3x3[1][1])
    for center in centers:
        if matrix[up(center)]==subMatrix3x3[0][1] and matrix[right(center)]==subMatrix3x3[1][2] and matrix[down(center)]==subMatrix3x3[2][1] and matrix[left(center)]==subMatrix3x3[1][0]:
            return center
    return None
  
def findCrossInPd(cross,planarDiagram):
    """Find a cross on a planar diagram."""
    for i in range(4):
        subMatrixX = np.array([[0,cross.strands[(i+3)%4],0],
                               [cross.strands[i],(i%2)-2,cross.strands[(i+2)%4]],
                               [0,cross.strands[(i+1)%4],0]])
        ind = findSubMatrix3x3In(planarDiagram,subMatrixX)
        if ind != None:
            return ind
    return None

def removeUnnecessaryRow(pd):
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

def removeUnnecessaryColumn(pd):
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

def isCapeOfCross(pd,ind):
    for i in range(4):
        if exists(uDLF(i,ind),pd) and pd[uDLF(i,ind)]<0:
            return i
    return -1

def removeCrossOfPD(pd,cross):
    ind = findCrossInPd(cross,pd)
    if type(ind) != type(None):
        pd[ind] = 0
        for i in range(4):
            pd[uDLF(i,ind)] = 0
    return pd

