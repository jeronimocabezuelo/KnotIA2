from __future__ import annotations
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

    def successorsICreate(self, maxStrands: int, n : int = None):
        if n == None:
            n = self.knot.numberOfStrands
        if n<maxStrands:
            for i in range(1,n+1):
                for orientation in range(4):
                    knotCopy = deepcopy(self.knot)
                    knotCopy.createALoop(i,orientation)
                    movCopy = deepcopy(self.mov)
                    movCopy.append("createALoop({},{})".format(i,orientation))
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

    def successorsI(self, maxStrands:int, n: int = None):
        if n == None:
            n = self.knot.numberOfStrands
        for node in self.successorsICreate(maxStrands,n=n):
            yield node
        for node in self.successorsIUndo(n=n):
            yield node

    def successorsIICreate(self,maxStrands: int, n: int = None):
        #print("IICreate")
        if n == None:
            n = self.knot.numberOfStrands
        if n<maxStrands:
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
        #print("IIUndo")
        if n == None:
            n = self.knot.numberOfStrands
        possibilities = [(l1,l2) for l1 in range(1,n+1) for l2 in range(l1,n+1) if (self.knot.typeOfStrand(l1) == StrandType.ABOVE and self.knot.typeOfStrand(l2) == StrandType.BELOW) or (self.knot.typeOfStrand(l1) == StrandType.BELOW and self.knot.typeOfStrand(l2) == StrandType.ABOVE)]
        for (l1,l2) in possibilities:
            knotCopy = deepcopy(self.knot)
            movCopy = deepcopy(self.mov)
            if knotCopy.undoReidemeisterII(l1,l2):
                movCopy.append("undoReidemeisterII({},{})".format(l1,l2))
                yield NodeKnot(knotCopy,movCopy)
        
    def successorsII(self, maxStrands:int, n: int = None):
        #print("II")
        if n == None:
            n = self.knot.numberOfStrands
        for node in self.successorsIICreate(maxStrands,n=n):
            yield node
        for node in self.successorsIIUndo(n=n):
            yield node
        
    def successorsIII(self,n: int = None):
        #print("III")
        if n == None:
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

    def successors(self,maxStrands: int):
        #for node in self.successorsRotates():
        #    yield node
        n = self.knot.numberOfStrands
        for node in self.successorsI(maxStrands,n=n):
            yield node
        for node in self.successorsII(maxStrands,n=n):
            yield node
        for node in self.successorsIII(n=n):
            yield node

    def successorsFiltered(self,maxStrands: int):
        auxSet:set[CustomKnot] = set()
        for node in self.successors(maxStrands):
            if not node.knot in auxSet:
                auxSet.add(node.knot)
                yield node
        
    def successorsList(self, maxStrands: int):
        list:list[NodeKnot] = []
        for node in self.successors(maxStrands):
            list.append(node)
        return list

    def successorsListFiltered(self, maxStrands: int):
        list:list[NodeKnot] = []
        for node in self.successorsFiltered(maxStrands):
            list.append(node)
        return list

    def deepSuccessors(self,deep:int,maxStrands: int):
        currentSuccessors = list()
        currentSuccessorsKnots = sset()
        for successor in self.successors(maxStrands):
            if deep >= 1:
                if currentSuccessorsKnots.add(successor.knot):
                    currentSuccessors.append(successor)
                    yield successor
            else:
                yield successor
        if deep >= 1: 
            for successor in currentSuccessors:
                for subSuccessor in successor.deepSuccessors(deep-1,maxStrands):
                    yield subSuccessor
    
    @property
    def priority(self)->float:
        if self.similarity == None:
            raise Exception("Todavía no puedes calcular la prioridad. Este nodo no tiene similarity")
        return self.similarity + len(self.mov)#/10 

class PriorityQueueNodeKnot:
    """A queue of NodeKnot with priorities, Node can be inserted with .put(priority,node) and extracted with .get()"""
    def __init__(self):
        self.queue: dict[str,tuple[float,NodeKnot]] = {}

    def priorities(self)->list[float]:
        """Returns a list with the priorities."""
        return [priority for priority,_ in self.queue.values()]

    def nodes(self)->list[NodeKnot]:
        return [node for _,node in self.queue.values()]

    def keys(self)->list[CustomKnot]:
        """Returns a list with the elements."""
        return [key for key in self.queue.keys()]

    def put(self,newNode:NodeKnot):
        """Inserts an element with its priority"""
        #start = time()
        #print("put")
        representation = newNode.knot.representationForHash
        newPriority = newNode.priority
        if not representation in self.queue:
            #print("insertamos")
            self.queue[representation]=(newPriority,newNode)
        else:
            oldPriority,oldNode = self.queue[representation]
            #print("no insertamos")
            if oldPriority>newPriority:
                #print("es mejor")
                self.queue[representation] = (newPriority,newNode)
        #end = time()
        #print("Time put",end - start)

    def eliminateTheWorst(self,maxKnots = 10000):
        if self.len>maxKnots:
            prioritiesLitsOfList = self.priorities()
            priorities = [priority for prioritiesList in prioritiesLitsOfList for priority in prioritiesList]
            priorities.sort()
            worst = priorities[maxKnots-1]
            print("Worst",worst)
            hashes = [h for h in self.queue.keys()]
            for h in hashes:
                internalList = self.queue[h]
                for tuple in internalList:
                    p,k = tuple
                    if p>worst:
                        if len(internalList)==1:
                            self.queue.pop(h)
                        else:
                            self.queue[h].remove(tuple)
    
    def get(self)->NodeKnot:
        """Gets an NodeKnot.."""
        if len(self.queue) == 0:
            raise Exception("The queue has no elements.")
        priorities = self.priorities()
        m = min(priorities)
        i = priorities.index(m)
        node = self.nodes()[i]
        self.queue.pop(node.knot.representationForHash)
        return node

    def updateSimilarities(self, newObjetive: CustomKnot, cache: SimilarityCache = None, type: int = 1):
        if cache == None:
            for node in self.nodes():
                if type==1:
                    newSimilarity = similarity(node.knot,newObjetive)
                else:
                    newSimilarity = similarity(newObjetive,node.knot)
                node.similarity = newSimilarity
        else:
            for node in self.nodes():
                if type==1:
                    newSimilarity = cache.similarity(node.knot,newObjetive)
                else:
                    newSimilarity = cache.similarity(newObjetive,node.knot)
                node.similarity = newSimilarity

    @property
    def empty(self):
        """It tells us if the queue is empty."""
        return len(self.queue)==0

    @property
    def len(self):
        """It tells us the length of the queue."""
        return len(self.queue)
    
    @property
    def hasRepeated(self):
        s = sset()
        for node in self.nodes():
            if not s.add(node.knot):
                return True
        return False

def similarity(knot1: CustomKnot, knot2: CustomKnot):
    n1 = knot1.numberOfStrands
    n2 = knot2.numberOfStrands
    auxes:list[float] = []
    for rotation in knot1.allRotationYield():
        aux = 0
        for s in range(max(n1,n2)):
            aux += similarityOfStrand(s,rotation,knot2,n1=n1,n2=n2)
        auxes.append(aux)
    return min(auxes)+abs(n1-n2)

def similarityOfStrand(s:Strand, knot1:CustomKnot, knot2: CustomKnot, n1: int| None = None, n2=None):
    if n1 == None: n1 = knot1.numberOfStrands
    if n2 == None: n2 = knot2.numberOfStrands
    if n1 == 0 and n2 == 0:
        return 0
    if n1 == 0:
        return 3
    if n2 == 0:
        return 3

    s1 = mod(s,n1)
    s2 = mod(s,n2)
    cross1,p1 = crossWithStrandAndDirection(knot1,s1,True)
    cross2,p2 = crossWithStrandAndDirection(knot2,s2,True)
    strandAbove1 = cross1.isStrandAbove(s1)
    strandAbove2 = cross2.isStrandAbove(s2)
    aux = strandAbove1!=strandAbove2
    right1 = cross1[(p1+1)%4]
    right2 = cross2[(p2+1)%4]
    left1 = cross1[(p1-1)%4]
    left2 = cross2[(p2-1)%4]
    d1 = right1-left1
    d2 = right2-left2
    aux+= d1 != d2
    otherS1 = min(right1,left1)
    otherS2 = min(right2,left2)
    d = abs(otherS1-otherS2)
    aux += remap(d,0,max(n1,n2),0,1)
    return aux

"""
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
        #aux += len(set(knot1Rotation.crosses).symmetric_difference(set(knot2.crosses)))
        similarities.append(aux)
    return min(similarities)

def similarity(knot1:CustomKnot,knot2:CustomKnot):
    return min(similarityKnot(knot1,knot2),similarityKnot(knot2,knot1))
"""

class Similarity:
    def __init__(self,objetiveKnot:CustomKnot, times=False):
        self.times = times
        self.objetiveKnot = objetiveKnot
        self.similarities:dict[str,float] = {}
        self._calls = 0

    def __getitem__(self, knot:CustomKnot | str):
        self._calls +=1
        if self.times: start = time()
        if type(knot) == CustomKnot:
            knotReprese: str = knot.representationForHash
        elif type(knot) == str:
            knotReprese: str = knot
        else:
            raise Exception("Incorrect Type")
        if knotReprese in self.similarities.keys():
            tupleSimilarity = self.similarities[knotReprese]
            if self.times: print("Time Similarity sin calculo",time() - start)  
            return tupleSimilarity
        else:
            tupleSimilarity = similarity(knot,self.objetiveKnot)
            self.similarities[knotReprese] = tupleSimilarity
            if self.times: print("Time Similarity con calculo",time() - start)   
            return tupleSimilarity
    
    def __len__(self):
        return len(self.similarities)

class SimilarityCache:
    def __init__(self):
        self.cache: dict[str,dict[str,float]] = dict()
        self._len = 0
        self._calls = 0
    def similarity(self,knot1: CustomKnot, knot2: CustomKnot,type:int=1):
        if type == 2:
            return self.similarity(knot2,knot1)
        self._calls+=1
        representation1 = knot1.representationForHash
        representation2 = knot2.representationForHash
        if representation1 in self.cache:
            dict1 = self.cache[representation1]
            if representation2 in dict1:
                return dict1[representation2]
            else:
                s = similarity(knot1,knot2)
                dict1[representation2] = s
                self._len += 1
                return s
        else:
            s = similarity(knot1,knot2)
            self.cache[representation1] = dict()
            self.cache[representation1][representation2] = s
            self._len += 1
            return s
    def __len__(self):
        return self._len

def areSameKnotsAStar(knot1:CustomKnot,knot2:CustomKnot,maxStrands:int = None, debug= False,times = False,timeLimit:float = float('inf')):
    if knot1 == knot2:
        return True,[""]
    if maxStrands == None:
        maxStrands = knot1.numberOfStrands + knot2.numberOfStrands
    initialNode = NodeKnot(deepcopy(knot1))
    initialNode.similarity = similarity(knot1,knot2)
    queue = PriorityQueueNodeKnot()
    queue.put(initialNode)
    visited = sset()
    startTime = time()
    while not queue.empty:
        node = queue.get()
        if debug>0 :print("queue len:",queue.len,"   visited len",len(visited),"get node similarity: {:3.2f}, priority: {:3.2f}, numberOfStrands: {}".format(node.similarity,node.priority,node.knot.numberOfStrands)); print("Mov: ", node.mov)
        
        if not visited.add(node.knot.representationForHash):
            if debug>0 : print("Esta en visited")
            continue
        if node.knot == knot2:
            return True,node.mov
        i=0
        if times>0: totalTime = 0.0
        for successor in node.successors(maxStrands):
            if (time()-startTime)>timeLimit:
                return False,[]
            i+=1
            if times>0: start = time()
            if successor.knot.representationForHash in visited:
                if debug>1 :print(i,"ya esta en visited")
                if times>1: print("Time check in visited",time() - start)
                continue
            
            if times>1: print("Time check1",time() - start)

            successor.similarity = similarity(successor.knot,knot2)
            if times>1: print("Time check2",time() - start)
            if debug>1 :print("{}: {:3.2f}, mov: {}".format(i,successor.priority,successor.mov))
            if successor.similarity == 0:
                queue.put(successor)
                return True, successor.mov
            if times>1: print("Time check3",time() - start)
            queue.put(successor)
            if times>0: print("Time successor",time() - start); totalTime += time()-start
        if times>0: print("Time expandNode ",totalTime)

def areSameKnotsAStar2(knot1: CustomKnot, knot2: CustomKnot, maxStrands: int = None, debug = False,times = False,timeLimit:float = float('inf'),cachePrint = False):
    if knot1 == knot2:
        return True, [""], [""]
    if maxStrands == None:
        maxStrands = knot1.numberOfStrands + knot2.numberOfStrands
    cache = SimilarityCache()
    bestNode1 = NodeKnot(knot1)
    bestNode2 = NodeKnot(knot2)
    initSimilarity = cache.similarity(bestNode1.knot,bestNode2.knot)
    bestNode1.similarity = initSimilarity
    bestNode2.similarity = initSimilarity
    queue1 = PriorityQueueNodeKnot()
    queue1.put(bestNode1)
    queue2 = PriorityQueueNodeKnot()
    queue2.put(bestNode2)
    visited1:dict[str,NodeKnot] = dict()
    visited2:dict[str,NodeKnot] = dict()
    startTime = time()

    while not ( queue1.empty or queue2.empty):
        if debug>0:
            print("-----")
            print("SimilarityCache len {} calls {}".format(len(cache),cache._calls))
            print("len(queue1) = ", queue1.len, "len(visited1) ", len(visited1))
            print("bestNode1 priority {:3.2f}".format(bestNode1.priority), " mov", bestNode1.mov)
            
        node1 = queue1.get()

        if debug>0:
            print()
            print("node1 priority {:3.2f}".format(node1.priority), " mov", node1.mov)

        visited1[node1.knot.representationForHash] = node1
        i=0
        flag = False
        for successor in node1.successors(maxStrands):
            if (time()-startTime)>timeLimit:
                if cachePrint: print("SimilarityCache len {} calls {}".format(len(cache),cache._calls))
                return False,[],[]
            i+=1
            if successor.knot.representationForHash in visited1:
                if debug>1 :print(i,"ya esta en visited")
                continue
            successor.similarity = cache.similarity(successor.knot,bestNode2.knot)
            if debug>1 :print("{}: {:3.2f}, mov: {}".format(i,successor.priority,successor.mov))
            if successor.similarity == 0:
                if cachePrint: print("SimilarityCache len {} calls {}".format(len(cache),cache._calls))
                return True, successor.mov, bestNode2.mov
            if successor.knot.representationForHash in visited2:
                if cachePrint: print("SimilarityCache len {} calls {}".format(len(cache),cache._calls))
                return True, successor.mov, visited2[successor.knot.representationForHash].mov

            if successor.priority < bestNode1.priority:
                flag = True
                if debug>0 :print(i,"Mejora bestNode1")
                bestNode1 = successor
                bestNode2.similarity = cache.similarity(bestNode1.knot,bestNode2.knot)
            queue1.put(successor)

        if flag:
            #update queue similarities
            queue2.updateSimilarities(bestNode1.knot,cache,type=2)

        if debug>0:
            print("len(queue2) = ", queue2.len, "len(visited2) ", len(visited2))
            print("bestNode2 priority {:3.2f}".format(bestNode2.priority), " mov", bestNode2.mov)
        
        node2 = queue2.get()

        if debug>0:
            print()
            print("node2 priority {:3.2f}".format(node2.priority), " mov", node2.mov)

        visited2[node2.knot.representationForHash] = node2
        i=0
        flag = False
        for successor in node2.successors(maxStrands):
            if (time()-startTime)>timeLimit:
                if cachePrint: print("SimilarityCache len {} calls {}".format(len(cache),cache._calls))
                return False,[],[]
            i+=1
            if successor.knot.representationForHash in visited2:
                if debug>1 :print(i,"ya esta en visited")
                continue
            successor.similarity = cache.similarity(bestNode1.knot,successor.knot)
            if debug>1 :print("{}: {:3.2f}, mov: {}".format(i,successor.priority,successor.mov))
            if successor.similarity == 0:
                if cachePrint: print("SimilarityCache len {} calls {}".format(len(cache),cache._calls))
                return True, bestNode1.mov, successor.mov
            if successor.knot.representationForHash in visited1:
                if cachePrint: print("SimilarityCache len {} calls {}".format(len(cache),cache._calls))
                return True, visited1[successor.knot.representationForHash].mov, successor.mov


            if successor.priority < bestNode2.priority:
                flag = True
                if debug>0 :print(i,"Mejora bestNode2")
                bestNode2 = successor
                bestNode1.similarity = similarity(bestNode1.knot,bestNode2.knot)
            queue2.put(successor)
        
        if flag:
            queue1.updateSimilarities(bestNode2.knot,cache,type=1)

class CustomKnotStar(CustomKnot):
    def randomMov(self, maxStrands: int = 100, debug=False):
        node = NodeKnot(self)
        typeMov = randrange(1,4)
        posibilites = []
        if debug: print("-------") 
        n = self.numberOfStrands
        if debug: print("number Of Strands:",n)
        if debug: print(self)
        if debug: print("type, ", typeMov)
        if typeMov == 1:
            createOrUndo = randrange(2) if n< maxStrands  else 0
            if createOrUndo:
                if debug: print("Create")
                for successor in node.successorsICreate(maxStrands,n=n):
                    posibilites.append(successor.mov[0])
            else:
                if debug: print("undo")
                for successor in node.successorsIUndo(n=n):
                    posibilites.append(successor.mov[0])
        elif typeMov == 2:
            createOrUndo = randrange(2) if n< maxStrands  else 0
            if createOrUndo:
                if debug: print("Create")
                for successor in node.successorsIICreate(maxStrands,n=n):
                    posibilites.append(successor.mov[0])
            else:
                if debug: print("undo")
                for successor in node.successorsIIUndo(n=n):
                    posibilites.append(successor.mov[0])
        elif typeMov == 3:
            for successor in node.successorsIII(n=n):
                posibilites.append(successor.mov[0])
        if len(posibilites)==0:
            return False, ""
        mov = choice(posibilites)
        
        self.eval(".{}".format(mov))
        return True,".{}".format(mov)