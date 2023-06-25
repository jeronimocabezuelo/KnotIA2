from __future__ import annotations
from tracemalloc import start
from random import *
import numpy as np
from time import time
import matplotlib.pyplot as plt

from CustomKnot import *
from KnotStar import difference, DifferenceSimpleCache

MutationRate = 0.5
MutationRateGenes = 0.3
MutationRateGenesNotEffective = 0.7


class Gene:
    def __init__(self, type: int, s1: int, s2: int, s3: int, o: int):
        self.type = type
        self.strand1 = s1
        self.strand2 = s2
        self.strand3 = s3
        self.orientation = o
        self.isEffective: bool = True

    def __repr__(self):
        return (
            "Gene("
            + str(self.type)
            + ","
            + str(self.strand1)
            + ","
            + str(self.strand2)
            + ","
            + str(self.strand3)
            + ","
            + str(self.orientation)
            + ")"
        )

    def mutate(self, rateType=0.05, rateGene=0.1):
        if random() < rateType:
            self.type = self.type + randint(-1, 1)
        if random() < rateGene:
            self.strand1 = self.strand1 + randint(-1, 1)
        if random() < rateGene:
            self.strand2 = self.strand2 + randint(-1, 1)
        if random() < rateGene:
            self.strand3 = self.strand3 + randint(-1, 1)
        if random() < rateGene:
            self.orientation = self.orientation + randint(-1, 1)

    def applyToKnot(self, knot: CustomKnot) -> str:
        type = mod(self.type, 5)
        n = knot.numberOfStrands
        strand1 = mod(self.strand1, n)
        strand2 = mod(self.strand2, n)
        strand3 = mod(self.strand3, n)
        if type == 1:
            orientation = (self.orientation) % 4
            knot.createALoop(strand1, orientation)
            self.isEffective = True
            return ".createALoop({},{})".format(strand1, orientation)
        elif type == 2:
            if knot.undoALoop(strand1):
                self.isEffective = True
                return ".undoALoop({})".format(strand1)
        elif type == 3:
            orientation = self.orientation % 2
            if knot.createReidemeisterII(strand1, strand2, orientation):
                self.isEffective = True
                return ".createReidemeisterII({},{},{})".format(
                    strand1, strand2, orientation
                )
        elif type == 4:
            if knot.undoReidemeisterII(strand1, strand2):
                self.isEffective = True
                return ".undoReidemeisterII({},{})".format(strand1, strand2)
        elif type == 5:
            if knot.reidemeisterIII(strand1, strand2, strand3):
                self.isEffective = True
                return ".reidemeisterIII({},{},{})".format(strand1, strand2, strand3)
        self.isEffective = False
        return ""


def randomGene(maxStrand: int = 100) -> Gene:
    return Gene(
        randint(1, 6),
        randint(1, maxStrand),
        randint(1, maxStrand),
        randint(1, maxStrand),
        randint(0, 4),
    )


class Genome:
    def __init__(self, genes: List[Gene] = None):
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

    def mutate(self, rateType=0.05, rateGene=0.1, rateGenes=0.05):
        for gene in self.genes:
            gene.mutate(rateType=rateType, rateGene=rateGene)
            if not gene.isEffective:
                if random() < MutationRateGenesNotEffective:
                    self.genes.remove(gene)
        while random() < rateGenes:
            addOrDelete = randrange(2)
            if addOrDelete:
                if len(self.genes) > 0:
                    self.genes.remove(choice(self.genes))
            else:
                if len(self.genes) > 0:
                    self.genes.insert(randrange(len(self.genes)), randomGene())
                else:
                    self.genes.append(randomGene())

    def applyToKnot(self, knot: CustomKnot) -> List[str]:
        moves: List[str] = []
        for gene in self.genes:
            moves.append(gene.applyToKnot(knot))
        return moves


def randomGenome(maxStrand=100, minGenes=5, maxGenes=12) -> Genome:
    genome = Genome()
    for _ in range(randint(minGenes, maxGenes)):
        genome.genes.append(randomGene(maxStrand=maxStrand))
    return genome


class IndividualKnot:
    def __init__(
        self, knot: CustomKnot, genome: Genome = None, minGenes=5, maxGenes=30
    ):
        self.knot = knot
        if genome == None:
            # self.genome = Genome()
            self.genome = randomGenome(
                max(6, self.knot.numberOfStrands), minGenes=minGenes, maxGenes=maxGenes
            )
        else:
            self.genome = genome

    @property
    def computedKnot(self):
        knot = deepcopy(self.knot)
        t = self.genome.applyToKnot(knot)
        return knot, t

    def mutate(
        self, rateType: float = 0.05, rateGene: float = 0.1, rateGenes: float = 0.05
    ):
        k, t = self.computedKnot
        self.genome.mutate(rateType=rateType, rateGene=rateGene, rateGenes=rateGenes)


def crossover2(motherGenome: Genome, fatherGenome: Genome):
    motherPoint = (
        randrange(len(motherGenome.genes)) if len(motherGenome.genes) != 0 else 0
    )
    fatherPoint = (
        randrange(len(fatherGenome.genes)) if len(fatherGenome.genes) != 0 else 0
    )
    daughterGenome = Genome(
        deepcopy(motherGenome.genes[:motherPoint])
        + deepcopy(fatherGenome.genes[fatherPoint:])
    )
    sonGenome = Genome(
        deepcopy(fatherGenome.genes[:fatherPoint])
        + deepcopy(motherGenome.genes[motherPoint:])
    )
    return [deepcopy(motherGenome), daughterGenome, sonGenome, deepcopy(fatherGenome)]


def crossover(motherGenome: Genome, fatherGenome: Genome):
    if len(motherGenome.genes) != 0 and len(fatherGenome.genes) != 0:
        point1Aux = randrange(min(len(motherGenome.genes), len(fatherGenome.genes)))
        point2Aux = randrange(min(len(motherGenome.genes), len(fatherGenome.genes)))
    else:
        point1Aux = 0
        point2Aux = 0

    point1 = min(point1Aux, point2Aux)
    point2 = max(point1Aux, point2Aux)
    motherGenes1 = deepcopy(motherGenome.genes[:point1])
    motherGenes2 = deepcopy(motherGenome.genes[point1:point2])
    motherGenes3 = deepcopy(motherGenome.genes[point2:])
    fatherGenes1 = deepcopy(fatherGenome.genes[:point1])
    fatherGenes2 = deepcopy(fatherGenome.genes[point1:point2])
    fatherGenes3 = deepcopy(fatherGenome.genes[point2:])
    daughterGenome = Genome(motherGenes1 + fatherGenes2 + motherGenes3)
    sonGenome = Genome(fatherGenes1 + motherGenes2 + fatherGenes3)
    return [deepcopy(motherGenome), daughterGenome, sonGenome, deepcopy(fatherGenome)]


class PopulationKnot:
    def __init__(
        self,
        knot: CustomKnot,
        objetiveKnot: CustomKnot,
        numberOfIndividuals=100,
        numberOfGenerations=100,
        maxMutationRateType=0.05,
        minMutationRateType=0.0005,
        maxMutationRateGene=0.1,
        minMutationRateGene=0.001,
        maxMutationRateGenes=0.3,
        minMutationRateGenes=0.003,
        differences=None,
        generationGrowthRate=[1, 0],
    ):
        self.knot = knot
        self.objetiveKnot = objetiveKnot
        self.population: List[IndividualKnot] = []
        if differences == None:
            self.differences = DifferenceSimpleCache(objetiveKnot)
        else:
            self.differences = differences
        self._numberOfIndividuals = numberOfIndividuals
        self._numberOfGenerations = numberOfGenerations
        self._maxMutationRateType = maxMutationRateType
        self._minMutationRateType = minMutationRateType
        self._maxMutationRateGene = maxMutationRateGene
        self._minMutationRateGene = minMutationRateGene
        self._maxMutationRateGenes = maxMutationRateGenes
        self._minMutationRateGenes = minMutationRateGenes
        self.mutationRateType = maxMutationRateType
        self.mutationRateGene = maxMutationRateGene
        self.mutationRateGenes = maxMutationRateGenes
        self.generationGrowthRate = generationGrowthRate
        self._generation: int = 0

        self.maxDifferences = []
        self.meanDifferences = []
        self.minDifferences = []
        for _ in range(numberOfIndividuals):
            individual = IndividualKnot(knot)
            self.population.append(individual)

    @property
    def populationForCurrentGeneration(self):
        factor = int(self._generation / self.generationGrowthRate[0])
        return self._numberOfIndividuals + factor * self.generationGrowthRate[1]

    def selection(self, times: bool = False, debug=0):
        if times:
            start = time()
        listDifference: List[float] = []
        for i in range(len(self.population)):
            if times:
                print("time individual init", time() - start)
            individual = self.population[i]
            percent = ((i + 1) / len(self.population)) * 100
            if debug > 0:
                print(
                    "  percentage selection calculate Difference: {:3.2f}   ".format(
                        percent
                    ),
                    end="\r",
                )
            computedKnot, moves = individual.computedKnot
            if times:
                print("time individual computedKnot", time() - start)
            similar = self.differences[computedKnot]
            if times:
                print("time individual differences", time() - start)
            if similar == 0 or computedKnot == self.objetiveKnot:
                if debug > 0:
                    print()
                    print(" Encontrado", moves)
                return individual
            if times:
                print("time individual check", time() - start)
            listDifference.append(similar)
            if times:
                print("time individual final", time() - start)

        self.maxDifferences.append(max(listDifference))
        self.meanDifferences.append(sum(listDifference) / len(listDifference))
        self.minDifferences.append(min(listDifference))
        if debug > 0:
            print(
                "                                                                 ",
                end="\r",
            )
            print("  maxDifference:  ", self.maxDifferences[-1])
            print("  meanDifference: ", self.meanDifferences[-1])
            print("  minDifference:  ", self.minDifferences[-1])
            print(
                "differences len: ",
                len(self.differences),
                "differences calls: ",
                self.differences._calls,
            )

        # listDifference.sort(key= lambda i : i[0])
        #
        # numberOfSelection = int((percentage/100)*len(self.population))
        # newPopulation = listDifference[0:numberOfSelection]
        # newPopulation = [individual for _,individual in newPopulation]

        # listDifference = [1/(i+1) for i in listDifference]
        # s = sum(listDifference)
        # listDifference = [i/s for i in listDifference]

        listDifference = [i - self.minDifferences[-1] for i in listDifference]
        listDifference = [i**1 for i in listDifference]
        listDifference = [1 / (i + 1) for i in listDifference]
        s = sum(listDifference)
        listDifference = [i / s for i in listDifference]

        newPopulation: List[IndividualKnot] = list(
            np.random.choice(
                self.population,
                self.populationForCurrentGeneration,
                p=listDifference,
                replace=False,
            )
        )

        self.population = deepcopy(newPopulation)

    def crossover(self, debug=0):
        shuffle(self.population)
        newPopulation: List[IndividualKnot] = []
        while len(self.population) > 1:
            # start = time()
            if debug > 0:
                print(
                    "  crossover: remaining individuals: {}   ".format(
                        len(self.population)
                    ),
                    end="\r",
                )
            mother = self.population.pop()
            father = self.population.pop()
            sonsGenome = crossover(mother.genome, father.genome)
            for i in range(len(sonsGenome)):
                genome = sonsGenome[i]
                individual = IndividualKnot(self.knot, genome)
                # if i == 1 or i == 2:
                #    individual.mutate(rateType=self.mutationRateType,rateGene=self.mutationRateGene,rateGenes=self.mutationRateGenes)
                newPopulation.append(individual)
            # print("Time:",time()-start)
        if debug > 0:
            print(
                "                                                                 ",
                end="\r",
            )
        self.population = newPopulation

    def mutate(self, debug=0):
        for i in range(len(self.population)):
            percent = ((i + 1) / len(self.population)) * 100
            if debug > 0:
                print("  percentage mutate: {:3.2f}   ".format(percent), end="\r")
            individual = self.population[i]
            while random() < MutationRate:
                individual.mutate(
                    rateType=self.mutationRateType,
                    rateGene=self.mutationRateGene,
                    rateGenes=self.mutationRateGenes,
                )
        if debug > 0:
            print(
                "                                                                 ",
                end="\r",
            )

    def updateMutationRates(self):
        self.mutationRateType = remap(
            self._generation,
            0,
            self._numberOfGenerations,
            self._maxMutationRateType,
            self._minMutationRateType,
        )
        self.mutationRateGene = remap(
            self._generation,
            0,
            self._numberOfGenerations,
            self._maxMutationRateGene,
            self._minMutationRateGene,
        )
        self.mutationRateGenes = remap(
            self._generation,
            0,
            self._numberOfGenerations,
            self._maxMutationRateGenes,
            self._minMutationRateGenes,
        )

    def newGeneration(self, debug=0):
        self.crossover(debug=debug)
        self.mutate(debug=debug)
        self._generation += 1
        a = self.selection(debug=debug)
        self.updateMutationRates()
        if a != None:
            return a


def areSameKnotGA(
    knot1: CustomKnot,
    knot2: CustomKnot,
    numberOfIndividuals=100,
    numberOfGenerations=100,
    differences=None,
    debug=0,
    maxMutationRateType=0.05,
    minMutationRateType=0.0005,
    maxMutationRateGene=0.1,
    minMutationRateGene=0.001,
    maxMutationRateGenes=0.3,
    minMutationRateGenes=0.003,
    generationGrowthRate=[1, 0],
) -> Tuple[bool, IndividualKnot | None]:
    if knot1 == knot2:
        return True, IndividualKnot(knot1, Genome())
    population = PopulationKnot(
        knot1,
        knot2,
        numberOfIndividuals=numberOfIndividuals,
        numberOfGenerations=numberOfGenerations,
        differences=differences,
        maxMutationRateType=maxMutationRateType,
        minMutationRateType=minMutationRateType,
        maxMutationRateGene=maxMutationRateGene,
        minMutationRateGene=minMutationRateGene,
        maxMutationRateGenes=maxMutationRateGenes,
        minMutationRateGenes=minMutationRateGenes,
        generationGrowthRate=generationGrowthRate,
    )
    c = 0
    while c < numberOfGenerations:
        c += 1
        if debug > 0:
            print(
                "Generation:",
                c,
                ", Pre generation Population: ",
                len(population.population),
            )
        a = population.newGeneration(debug=debug)
        if debug > 0:
            print(
                "Generation:",
                c,
                ", Post generation Population:",
                len(population.population),
            )
        if a != None:
            return (
                True,
                a,
                [
                    population.maxDifferences,
                    population.meanDifferences,
                    population.minDifferences,
                ],
            )
    return (
        False,
        None,
        [
            population.maxDifferences,
            population.meanDifferences,
            population.minDifferences,
        ],
    )


def plotDifferences(
    differencesMMM,
):
    fig, ax = plt.subplots()
    ax.plot(
        range(1, len(differencesMMM[0]) + 1),
        differencesMMM[0],
        color="tab:blue",
        label="Max",
    )
    ax.plot(
        range(1, len(differencesMMM[1]) + 1),
        differencesMMM[1],
        color="tab:green",
        label="Mean",
    )
    ax.plot(
        range(1, len(differencesMMM[2]) + 1),
        differencesMMM[2],
        color="tab:red",
        label="Min",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Difference")
    ax.legend(loc="upper right")
    plt.show()
