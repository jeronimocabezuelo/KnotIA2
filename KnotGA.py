from __future__ import annotations
from tracemalloc import start
from CustomKnot import *
from random import *
import numpy as np
from time import time

from KnotStar import similarity, Similarity

MutationRate = 0.7
MutationRateType = 0.1
MutationRateGene = 0.3
MutationRateGenes = 0.4
MutationRateGenesNotEffective = 0.7

class Gene:
    def __init__(self,type: int, g1: int, g2: int, g3: int):
        self.gene = [type,g1,g2,g3]
        self.isEffective:bool = True
    
    def __repr__(self):
        return "Gene("+str(self.type)+","+str(self.gene[1])+","+str(self.gene[2])+","+str(self.gene[3])+")"
    
    @property
    def type(self):
        return self.gene[0]
    
    def mutate(self):
        if random()<MutationRateType:
            self.gene[0] = self.type+randint(-1, 1)
        for i in range(1,4):
            if random()<MutationRateGene:
                self.gene[i] = self.gene[i]+randint(-1,1)

    def applyToKnot(self,knot:CustomKnot)->str:
        type = mod(self.type,5)
        n = knot.numberOfStrands
        strand1 = mod(self.gene[1],n)
        strand2 = mod(self.gene[2],n)
        strand3 = mod(self.gene[3],n)
        if type == 1:
            orientation = (self.gene[2] + self.gene[3])%4
            knot.createALoop(strand1,orientation)
            self.isEffective = True
            return ".createALoop({},{})".format(strand1,orientation)
        elif type == 2:
            if knot.undoALoop(strand1):
                self.isEffective = True
                return ".undoALoop({})".format(strand1)
        elif type == 3:
            orientation = self.gene[3]%2
            if knot.createReidemeisterII(strand1,strand2,orientation):
                self.isEffective = True
                return ".createReidemeisterII({},{},{})".format(strand1,strand2,orientation)
        elif type == 4:
            if knot.undoReidemeisterII(strand1,strand2):
                self.isEffective = True
                return ".undoReidemeisterII({},{})".format(strand1,strand2)
        elif type == 5:
            if knot.reidemeisterIII(strand1,strand2,strand3):
                self.isEffective = True
                return ".reidemeisterIII({},{},{})".format(strand1,strand2,strand3)
        self.isEffective = False
        return ""
        
def randomGene()->Gene:
    return Gene(randint(0,10),randint(0,10),randint(0,10),randint(0,10))

class Genome:
    def __init__(self,genes:list[Gene] = None):
        if genes == None:
            self.genes = []
        else:
            self.genes = genes

    def __repr__(self):
        return str(self.genes)

    def __iter__(self):
        return iter(self.genes)

    def __next__(self):
        while True:
            try:
                value = next(self)
            except StopIteration:
                break
        return value

    def mutate(self):
        for gene in self.genes:
            gene.mutate()
            if not gene.isEffective:
                if random() < MutationRateGenesNotEffective:
                    self.genes.remove(gene)
        while random()<MutationRateGenes:
            addOrDelete = randrange(2)
            if addOrDelete:
                if len(self.genes)>0:
                    self.genes.remove(choice(self.genes))
            else:
                if len(self.genes)>0:
                    self.genes.insert(randrange(len(self.genes)),randomGene())
                else: self.genes.append(randomGene())
    
    def applyToKnot(self,knot:CustomKnot)->list[str]:
        moves:list[str] = []
        for gene in self.genes:
            moves.append(gene.applyToKnot(knot))
        return moves

def randomGenome(minGenes = 5 , maxGenes = 12)->Genome:
    genome = Genome()
    for _ in range(randint(minGenes,maxGenes)):
        genome.genes.append(randomGene())  
    return genome

class IndividualKnot:
    def __init__(self, knot: CustomKnot, genome: Genome = None, minGenes = 5, maxGenes = 12):
        self.knot = knot
        if genome == None:
            #self.genome = Genome()
            self.genome = randomGenome(minGenes = minGenes, maxGenes = maxGenes)
        else: self.genome = genome

    @property
    def computedKnot(self) -> CustomKnot:
        knot = deepcopy(self.knot)
        t = self.genome.applyToKnot(knot)
        return knot, t

    def mutate(self):
        self.genome.mutate()
    
def crossover(motherGenome: Genome, fatherGenome: Genome):
    motherPoint = randrange(len(motherGenome.genes)) if len(motherGenome.genes) != 0 else 0
    fatherPoint = randrange(len(fatherGenome.genes)) if len(fatherGenome.genes) != 0 else 0
    daughterGenome = Genome(deepcopy(motherGenome.genes[:motherPoint]) + deepcopy(fatherGenome.genes[fatherPoint:]))
    sonGenome = Genome(deepcopy(fatherGenome.genes[:fatherPoint]) + deepcopy(motherGenome.genes[motherPoint:]))
    return [deepcopy(motherGenome),daughterGenome,sonGenome,deepcopy(fatherGenome)]


class PopulationKnot:
    def __init__(self,knot: CustomKnot,objetiveKnot:CustomKnot ,numberOfIndividuals = 4):
        self.knot = knot
        self.objetiveKnot = objetiveKnot
        self.population: list[IndividualKnot] = []
        self.similarities = Similarity(objetiveKnot)
        self._numberOfIndividuals = numberOfIndividuals
        for _ in range(numberOfIndividuals):
            individual = IndividualKnot(knot)
            self.population.append(individual)
    
    def selection(self,times: bool = False,debug=0):
        if times: start = time()
        listSimilarity:list[float] = []
        for i in range(len(self.population)):
            if times: print("time individual init", time()-start)
            individual = self.population[i]
            percent = ((i+1)/len(self.population))*100
            if debug>0: print("  percentage selection calculate Similarity: {:3.2f}   ".format(percent),end="\r")
            computedKnot, moves = individual.computedKnot
            if times: print("time individual computedKnot", time()-start)
            similar = self.similarities[computedKnot]
            if times: print("time individual similarities", time()-start)
            if similar == 0 or computedKnot == self.objetiveKnot:
                if debug>0: print(); print(" Encontrado",moves)
                return individual
            if times: print("time individual check", time()-start)
            listSimilarity.append(similar)
            if times: print("time individual final", time()-start)
        if debug>0:
            print("                                                                 ",end="\r")
            print("  maxSimilarity: ",max(listSimilarity))
            print("  minSimilarity: ",min(listSimilarity))
            
        #listSimilarity.sort(key= lambda i : i[0])
        #
        #numberOfSelection = int((percentage/100)*len(self.population))
        #newPopulation = listSimilarity[0:numberOfSelection]
        #newPopulation = [individual for _,individual in newPopulation]

        listSimilarity = [1/(i+1) for i in listSimilarity]
        s = sum(listSimilarity)
        listSimilarity = [i/s for i in listSimilarity]
        newPopulation: list[IndividualKnot] = list(np.random.choice(self.population, self._numberOfIndividuals, p = listSimilarity, replace=False))
        
        self.population = deepcopy(newPopulation)
    
    def crossover(self,debug=0):
        shuffle(self.population)
        newPopulation: list[IndividualKnot] = []
        while len(self.population)>1:
            #start = time()
            if debug>0: print("  crossover: remaining individuals: {}   ".format(len(self.population)),end="\r")
            mother = self.population.pop()
            father = self.population.pop()
            for genome in crossover(mother.genome,father.genome):
                newPopulation.append(IndividualKnot(self.knot,genome))
            #print("Time:",time()-start)
        if debug>0: print("                                                                 ",end="\r")
        self.population = newPopulation
    
    def mutate(self,debug=0):
        for i in range(len(self.population)):
            percent = ((i+1)/len(self.population))*100
            if debug>0: print("  percentage mutate: {:3.2f}   ".format(percent),end="\r")
            individual = self.population[i]
        #for individual in self.population:
            while random() < MutationRate:
                individual.mutate()
        if debug>0: print("                                                                 ",end="\r")

    def newGeneration(self, debug=0):
        self.mutate(debug=debug)
        self.crossover(debug=debug)
        a:IndividualKnot = self.selection(debug=debug)
        if a == None:
            pass
        else:
            return a

def areSameKnotGA(knot1:CustomKnot, knot2: CustomKnot, numberOfIndividuals = 100,numberOfGenerations = 100, debug=0)->tuple[bool,IndividualKnot|None]:
    if knot1 == knot2:
        return True, IndividualKnot(knot1,Genome())
    population = PopulationKnot(knot1,knot2,numberOfIndividuals=numberOfIndividuals)
    c=0
    while c<numberOfGenerations:
        c+=1
        if debug>0: print("Generation ", c)
        a = population.newGeneration(debug=debug)
        if a != None:
            return True, a
    return False, None