from __future__ import annotations
from email.generator import Generator
from CustomKnot import *
from time import time
class NodeKnot:

    def __init__(self,knot:CustomKnot,mov:list[str]=[]):
        self.knot = knot
        self.mov = mov
        self.similarity: float| None = None

    def successorsRotates(self):
        #print("rotations")
        knotCopy = deepcopy(self.knot)
        for i in range(2):
            for r in range(knotCopy.numberOfStrands):
                if knotCopy.crosses != self.knot.crosses:
                    movCopy = deepcopy(self.mov)
                    movCopy.append("inverse({}).rotate({})".format(i,r))
                    yield NodeKnot(deepcopy(knotCopy),movCopy)
                knotCopy.rotate()
            knotCopy.inverse()

    def successorsICreate(self, maxCrosses: int, n : int = None):
        if n == None:
            n = self.knot.numberOfStrands
        if n<maxCrosses:
            for i in range(1,n+1):
                for orientation in range(4):
                    knotCopy = deepcopy(self.knot)
                    knotCopy.createALoop(i,orientation)
                    movCopy = deepcopy(self.mov)
                    movCopy.append("crateALoop({},{})".format(i,orientation))
                    yield NodeKnot(knotCopy,movCopy)
    
    def successorsIUndo(self, n: int = None):
        if n == None:
            n = self.knot.numberOfStrands
        for i in range(1,n+1):
            knotCopy = deepcopy(self.knot)
            movCopy = deepcopy(self.mov)
            if knotCopy.undoALoop(i):
                movCopy.append("undoALoop({})".format(i))
                yield NodeKnot(knotCopy,movCopy)

    def successorsI(self, maxCrosses:int, n: int = None):
        if n == None:
            n = self.knot.numberOfStrands
        for node in self.successorsICreate(maxCrosses,n=n):
            yield node
        for node in self.successorsIUndo(n=n):
            yield node

    def successorsIICreate(self,maxCrosses: int, n: int = None):
        if n == None:
            n = self.knot.numberOfStrands
        if n<maxCrosses:
            possibilities = [(l1,l2) for l1 in range(1,n+1) for l2 in range(l1,n+1)]
            for (l1,l2) in possibilities:
                knotCopy = deepcopy(self.knot)
                typeReidemeisterII = knotCopy.isPosibleCreateReidemeisterII(l1,l2)
                if typeReidemeisterII != 0:
                    for orientation in range(4):
                        knotCopy = deepcopy(self.knot)
                        movCopy = deepcopy(self.mov)
                        if knotCopy.createReidemeisterII(l1,l2,orientation,typeReidemeisterII):
                            movCopy.append("createReidemeisterII({},{},{},{})".format(l1,l2,orientation,typeReidemeisterII))
                            yield NodeKnot(knotCopy,movCopy)
    
    def successorsIIUndo(self,n: int = None):
        if n == None:
            n = self.knot.numberOfStrands
        possibilities = [(l1,l2) for l1 in range(1,n+1) for l2 in range(l1,n+1) if (self.knot.typeOfStrand(l1) == StrandType.ABOVE and self.knot.typeOfStrand(l2) == StrandType.BELOW) or (self.knot.typeOfStrand(l1) == StrandType.BELOW and self.knot.typeOfStrand(l2) == StrandType.ABOVE)]
        for (l1,l2) in possibilities:
            knotCopy = deepcopy(self.knot)
            movCopy = deepcopy(self.mov)
            if knotCopy.undoReidemeisterII(l1,l2):
                movCopy.append("undoReidemeisterII({},{})".format(l1,l2))
                yield NodeKnot(knotCopy,movCopy)

    def successorsII(self, maxCrosses:int, n: int = None):
        #print("II")
        if n == None:
            n = self.knot.numberOfStrands
        for node in self.successorsIICreate(maxCrosses,n=n):
            yield node
        for node in self.successorsIIUndo(n=n):
            yield node
        
    def successorsIII(self):
        #print("III")
        n = self.knot.numberOfStrands
        possibilities = {StrandType.ABOVE:[], StrandType.MIDDLE:[],StrandType.BELOW:[],None:[]}
        for l in range(1,n+1):
            t = self.knot.typeOfStrand(l)
            possibilities[t].append(l)
        possibilities = [(l1,l2,l3) for l1 in possibilities[StrandType.BELOW] for l2 in possibilities[StrandType.MIDDLE] for l3 in possibilities[StrandType.ABOVE]]
        possibilities = [(l1,l2,l3) for (l1,l2,l3) in possibilities if mod(l1+1,n)!=l2 and mod(l1-1,n)!=l2 and mod(l1+1,n)!=l3 and mod(l1-1,n)!=l3 and mod(l2+1,n)!=l3 and mod(l2-1,n)!=l3]
        for (l1,l2,l3) in possibilities:
            knotCopy = deepcopy(self.knot)
            movCopy = deepcopy(self.mov)
            if knotCopy.reidemeisterIII(l1,l2,l3,check=False):
                movCopy.append("reidemeisterIII({},{},{},check=False)".format(l1,l2,l3))
                yield NodeKnot(knotCopy,movCopy)

    def successors(self,maxCrosses: int):
        #for node in self.successorsRotates():
        #    yield node
        for node in self.successorsI(maxCrosses):
            yield node
        for node in self.successorsII(maxCrosses):
            yield node
        for node in self.successorsIII():
            yield node

    def successorsFiltered(self,maxCrosses: int):
        auxSet:set[CustomKnot] = set()
        for node in self.successors(maxCrosses):
            if not node.knot in auxSet:
                auxSet.add(node.knot)
                yield node
        
    def successorsList(self, maxCrosses: int):
        list:list[NodeKnot] = []
        for node in self.successors(maxCrosses):
            list.append(node)
        return list

    def successorsListFiltered(self, maxCrosses: int):
        list:list[NodeKnot] = []
        for node in self.successorsFiltered(maxCrosses):
            list.append(node)
        return list

    def priority(self)->float:
        if self.similarity == None:
            raise Exception("Todavía no puedes calcular la prioridad. Este nodo no tiene similarity")
        return self.similarity + len(self.mov)

def minOfArrayOfArrays(list:list[list[float]])->tuple[list[float],float,int,int]:
    actualListMin = list[0]
    actualMin = min(list[0])
    for internalList in list:
        internalMin = min(internalList)
        if internalMin<actualMin:
            actualMin = internalMin
            actualListMin = internalList
    i = list.index(actualListMin)
    j = actualListMin.index(actualMin)
    return actualListMin,actualMin,i,j
        
class PriorityQueueNodeKnot:
    """A queue of NodeKnot with priorities, Node can be inserted with .put(priority,node) and extracted with .get()"""
    def __init__(self):
        self.queue: dict[int,list[tuple[float,NodeKnot]]] = {}
    def priorities(self)->list[list[float]]:
        """Returns a list with the priorities."""
        return [[priority for priority,_ in tuples] for tuples in self.queue.values()]
    #    return [element[0] for element in self.queue]

    def hashes(self)->list[CustomKnot]:
        """Returns a list with the elements."""
        return [h for h in self.queue.keys()]

    def nodes(self)->list[list[NodeKnot]]:
        return [[node for _,node in tuples] for tuples in self.queue.values()]

    def put(self,newNode:NodeKnot,h=None):
        """Inserts an element with its priority"""
        #start = time.time()
        if h==None:
            h = hash(newNode.knot)
        newPriority = newNode.priority()
        if not h in self.queue:
            #print("insertamos")
            self.queue[h]=[(newPriority,newNode)]
        else:
            #print("no insertamos")
            for i in range(len(self.queue[h])):
                oldPriority,oldNode = self.queue[h][i]
                if oldNode.knot == newNode.knot:
                    if oldPriority>newPriority:
                        #print("es mejor")
                        self.queue[h][i] = (newPriority,newNode)
                    return
            self.queue[h].append(((newPriority,newNode)))
        #end = time.time()
        #print("Time put",end - start)
        
    def get(self)->NodeKnot:
        """Gets an NodeKnot.."""
        if len(self.queue) == 0:
            raise Exception("The queue has no elements.")
        priorities = self.priorities()
        listMi,mi,i,j = minOfArrayOfArrays(priorities)
        nodes = self.nodes()[i]
        node = deepcopy(nodes[j])
        h = self.hashes()[i]
        if len(listMi) == 1:
            self.queue.pop(h)
        else:
            self.queue[h].pop(j)
        return node
    @property
    def empty(self):
        """It tells us if the queue is empty."""
        return len(self.queue)==0
    @property
    def len(self):
        """It tells us the length of the queue."""
        return sum([len(nods) for nods in self.nodes()])

def similarityOfStrand(strand:int,knot1:CustomKnot,knot2:CustomKnot,n1:int|None = None,n2:int|None = None) ->float:
    if n1 == None:
        n1 = knot1.numberOfStrands
    if n2 == None:
        n2 = knot2.numberOfStrands
    cross1,i1 = crossWithStrandAndDirection(knot1,mod(strand,n1),True)
    cross2,i2 = crossWithStrandAndDirection(knot2,mod(strand,n2),True)
    aux = i1%2 != i2%2
    d1 = cross1[mod(i1-1,4)] == mod(cross1[mod(i1+1,4)]+1,n1)
    d2 = cross2[mod(i2-1,4)] == mod(cross2[mod(i2+1,4)]+1,n2)
    aux += d1!=d2
    s1 = cross1[mod(i1-1,4)]
    s2 = cross2[mod(i2-1,4)]
    aux += s1!=s2
    l = strand
    if cross1 == cross2:
        auxL = 1
        while True:
            l+=1
            cross1Aux,_ = crossWithStrandAndDirection(knot1,mod(l,n1),True)
            cross2Aux,_ = crossWithStrandAndDirection(knot2,mod(l,n2),True)
            if mod(l,n1) == strand or mod(l,n2)==strand or cross1Aux != cross2Aux:
                break
            auxL += 1
        aux += remap(auxL,0,min(n1,n2),1,0)
        l = strand
        auxL = 1
        while True:
            cross1Aux,_ = crossWithStrandAndDirection(knot1,mod(l,n1),False)
            cross2Aux,_ = crossWithStrandAndDirection(knot2,mod(l,n2),False)
            l-=1
            if mod(l,n1) == strand or mod(l,n2)==strand or cross1Aux != cross2Aux:
                break
            auxL += 1
        aux += remap(auxL,0,min(n1,n2),1,0)
    else:
        aux += 2
    return aux


def similarityKnot(knot1:CustomKnot,knot2:CustomKnot):
    n1 = knot1.numberOfStrands
    n2 = knot2.numberOfStrands
    similarities:list[float] = []
    for knot1Rotation in knot1.allRotation():
        aux = abs(n1-n2)
        for nStrand in range(min(n1,n2)):
            strand = mod(nStrand,min(n1,n2))
            auxSimilarity = similarityOfStrand(strand,knot1Rotation,knot2)
            aux += auxSimilarity
        aux += len(set(knot1Rotation.crosses).symmetric_difference(set(knot2.crosses)))
        similarities.append(aux)
    return min(similarities)

def similarity(knot1:CustomKnot,knot2:CustomKnot):
    return min(similarityKnot(knot1,knot2),similarityKnot(knot2,knot1))

class Similarity:
    def __init__(self,objetiveKnot:CustomKnot):
        self.objetiveKnot = objetiveKnot
        self.similarities:dict[int,list[tuple[CustomKnot,float]]] = {}
    def __getitem__(self, knot:CustomKnot):
        #start = time()
        h = hash(knot)
        if h in self.similarities.keys():
            tuples = self.similarities[h]
            for tupleKnot, tupleSimilarity in tuples:
                if tupleKnot == knot:
                    #end = time()
                    #print("Time Similarity sin calculo",end - start)  
                    return tupleSimilarity
            tupleSimilarity = similarity(knot,self.objetiveKnot)
            tuples.append((knot,tupleSimilarity))
            #end = time()
            #print("Time Similarity con calculo",end - start)          
            return tupleSimilarity
        else:
            tupleSimilarity = similarity(knot,self.objetiveKnot)
            self.similarities[h] = [(knot,tupleSimilarity)]
            #end = time()
            #print("Time Similarity con calculo",end - start)   
            return tupleSimilarity

def expandNode(node:NodeKnot,objetiveKnot:CustomKnot,maxCrosses:int,queue: PriorityQueueNodeKnot,visited:list[CustomKnot],similarities: Similarity,bestPriority=None,debug = False,times = False):
    visited.append(node.knot)
    if bestPriority == None:
        bestPriority = node.priority()
    if debug>0 : print("bestPriority: {:3.2f}, mov: {}".format(bestPriority,node.mov))
    if bestPriority == 0:
        return 0
    i=0
    if times: totalTime = 0.0
    for successor in node.successors(maxCrosses):
        if times: start = time.time()
        i+=1
        if successor.knot in visited:
            if debug>1 :print(i,"ya esta en visited")
            if times: 
                end = time.time()
                print("Time check in visited",end - start)
            continue
        if times: 
            end = time.time()
            print("Time check1",end - start)

        h = hash(successor.knot)
        successor.similarity = similarities[successor.knot]
        
        if debug>1 :print("{}: {:3.2f}, mov: {}".format(i,successor.priority(),successor.mov))
        if successor.similarity == 0:
            queue.put(successor,h=h)
            return 0
        newSimilarity = successor.similarity
        #if newSimilarity>=bestPriority*0.8:
        queue.put(successor,h=h)
        #    if times: 
        #        end = time.time()
        #        print("Time check5",end - start)
        #else:
        #    if debug>1:
        #        print("expandimos")
        #        print(successor.mov[-1])
        #    bestPriority = expandNode(successor,objetiveKnot,maxCrosses,queue,visited,similarities,newSimilarity,debug=debug,times=times)
        #    if debug>1 :print("continuamos")
        if times: 
            end = time.time()
            print("Time successor",end - start)
            totalTime += end-start
    if times: print("Time expandNode ",totalTime)
    return bestPriority 


def areSameKnots(knot1:CustomKnot,knot2:CustomKnot,maxCrosses:int, debug= False,times = False):
    initialNode = NodeKnot(deepcopy(knot1))
    initialNode.similarity = similarity(knot1,knot2)
    queue = PriorityQueueNodeKnot()
    queue.put(initialNode)
    visited:list[CustomKnot] = []
    similarities = Similarity(knot2)
    c = 0
    while not queue.empty and c<100:
        c+=1
        node = queue.get()
        if debug>0 :print("get","queue len:",queue.len,"   visited len",len(visited))
        if node.knot in visited:
            if debug>0 : print("Esta en visited")
            continue
        if node.knot == knot2:
            return True,node.mov
        expandNode(node,knot2,maxCrosses,queue,visited,similarities,debug=debug,times=times)