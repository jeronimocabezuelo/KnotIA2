from __future__ import annotations
import matplotlib.pyplot as plt
from CustomKnot import CustomKnot
from copy import deepcopy
from enum import Enum
from numpy.random import choice
from KnotStar import DifferenceSimpleCache, difference
from KnotGA import plotDifferences
from IPython.display import clear_output
from time import time


class ReidemeisterType(Enum):
    CreateALoop = 0
    UndoALoop = 1
    CreateII = 2
    UndoII = 3
    III = 4


class KnotAnt:
    def __init__(
        self,
        house: CustomKnot,
        objetive: CustomKnot,
        actual: CustomKnot | None = None,
        differences: DifferenceSimpleCache | None = None,
    ):
        self.house = house
        self.objetive = objetive
        self.actual = deepcopy(house) if actual == None else actual
        self.differences = differences
        self.movements: list[str] = []
        self._actualDifference: float | None = None

    @property
    def actualDifference(self):
        if self._actualDifference != None:
            return self._actualDifference
        else:
            if self.differences != None:
                self._actualDifference = self.differences[self.actual]
            else:
                self._actualDifference = difference(self.actual, self.objetive)
            return self._actualDifference

    def randomMov(self, pheromones: Pheromones, maxCrosses=100):
        numberOfCopies = 4

        copies = [
            KnotAnt(self.house, self.objetive, deepcopy(self.actual), self.differences)
            for i in range(numberOfCopies)
        ]
        moves = [c.actual.randomMov(maxCrosses) for c in copies]

        movesDifferences = [c.actualDifference for c in copies]
        minMovesDifferences = min(movesDifferences)
        movesDifferences = [diff - minMovesDifferences for diff in movesDifferences]
        movesDifferences = [1 / (diff + 1) for diff in movesDifferences]
        movesDifferences = [
            movesDifferences[i] + pheromones[copies[i].actual]
            for i in range(numberOfCopies)
        ]
        s = sum(movesDifferences)
        movesDifferences = [diff / s for diff in movesDifferences]

        selectedIndex = choice([0, 1, 2, 3], p=movesDifferences)

        b, mov = moves[selectedIndex]
        if b:
            self.actual = copies[selectedIndex].actual
            self._actualDifference = copies[selectedIndex].actualDifference
            self.movements.append(mov)

    def randomMov2(self, maxCrosses=100):
        copie = KnotAnt(
            self.house, self.objetive, deepcopy(self.actual), self.differences
        )
        b, mov = copie.actual.randomMov(maxCrosses)

        if b:
            self.actual = copie.actual
            # self._actualDifference = copie.actualDifference
            self.movements.append(mov)

    def check(self) -> bool:
        return self.actual == self.objetive


class Pheromones:
    def __init__(self):
        self._internalDict: dict[str, float] = {}
        self.incrementInVisit = 0.1
        self.decrementInEvaporation = 0.04

    def knotHash(self, knot: CustomKnot | str):
        if type(knot) == CustomKnot:
            return knot.representationForHash
        elif type(knot) == str:
            return knot
        else:
            raise Exception("Incorrect Type")

    def __getitem__(self, knot: CustomKnot | str):
        knotHash: str = self.knotHash(knot)
        value = self._internalDict.get(knotHash, 0)
        return value

    def __setitem__(self, knot: CustomKnot | str, value: float):
        knotHash: str = self.knotHash(knot)
        self._internalDict[knotHash] = value

    def visit(self, knot: CustomKnot | str):
        value = self[knot]
        self[knot] = value + self.incrementInVisit

    def visitColony(self, colony: Colony):
        for ant in colony.ants:
            self.visit(ant.actual)

    def evaporation(self):
        for knot in list(self._internalDict.keys()):
            newValue = self[knot] - self.decrementInEvaporation
            if newValue > 0:
                self[knot] = newValue
            else:
                self._internalDict.pop(knot)

    def hist(self):
        plt.hist(self._internalDict.values(), bins=100)


class Colony:
    def __init__(
        self,
        house: CustomKnot,
        objetive: CustomKnot,
        numberOfAnts=100,
    ):
        self.ants: list[KnotAnt] = []
        self.differences = DifferenceSimpleCache(objetive)
        self.pheromones = Pheromones()
        for _ in range(numberOfAnts):
            self.ants.append(KnotAnt(house, objetive, differences=self.differences))

    def randomMov(self, maxCrosses=100):
        for ant in self.ants:
            ant.randomMov2(maxCrosses=maxCrosses)

    def randomMov2(self, maxCrosses=100):
        for ant in self.ants:
            ant.randomMov(self.pheromones, maxCrosses=maxCrosses)
        self.pheromones.visitColony(self)
        self.pheromones.evaporation()

    def check(self, selection=False, debug=False) -> KnotAnt | None:
        for ant in self.ants:
            if ant.check():
                return ant
        if selection:
            self.ants.sort(key=lambda ant: ant.actualDifference)
            self.ants = self.ants[0 : int(len(self.ants) / 2)]
            self.ants += deepcopy(self.ants)
            for ant in self.ants:
                ant.differences = self.differences
        if debug:
            print(
                "Cache calls:",
                self.differences._calls,
                ", Cache len: ",
                len(self.differences.differences),
            )
            print("Pheromones: ", len(self.pheromones._internalDict.keys()))
        return None


def areSameKnotAnt(
    knot1: CustomKnot,
    knot2: CustomKnot,
    numberOfAnts=100,
    maxCrosses=100,
    generationLimit=1000,
    debug=False,
    timeLimit: float = float("inf"),
) -> tuple[bool, list[str]]:
    colony = Colony(knot1, knot2, numberOfAnts)
    generation = 0
    startTime = time()
    while generation < generationLimit:
        if debug:
            print("Generation: ", generation)
            print(len(colony.ants))
        generation += 1

        ant = colony.check(debug=debug)
        if ant != None:
            return True, ant.movements
        colony.randomMov(maxCrosses)

        if (time() - startTime) > timeLimit:
            return False, []

    return False, []


def areSameKnotAnt2(
    knot1: CustomKnot,
    knot2: CustomKnot,
    numberOfAnts=100,
    maxCrosses=100,
    generationLimit=1000,
    debug=False,
    plotPheromones=False,
    timeLimit: float = float("inf"),
    selectEachGenerations=5,
) -> tuple[bool, list[str]]:
    colony = Colony(knot1, knot2, numberOfAnts)
    generation = 0
    startTime = time()
    while generation < generationLimit:
        if debug:
            print("Generation: ", generation)
            print(len(colony.ants))
        generation += 1

        if selectEachGenerations != None:
            selection = generation % selectEachGenerations == 0
        else:
            selection = False

        ant = colony.check(selection=selection, debug=debug)
        if ant != None:
            return True, ant.movements
        colony.randomMov2(maxCrosses)

        if (time() - startTime) > timeLimit:
            return False, []

        if plotPheromones:
            clear_output(wait=True)
            colony.pheromones.hist()
    return False, []


class DoubleColony:
    def __init__(self, house: CustomKnot, objetive: CustomKnot, numberOfAnts=100):
        self.colony1 = Colony(house, objetive, numberOfAnts=numberOfAnts)
        self.colony2 = Colony(objetive, house, numberOfAnts=numberOfAnts)

    def randomMov(self, maxCrosses=100):
        self.colony1.randomMov(maxCrosses=maxCrosses)
        self.colony2.randomMov(maxCrosses=maxCrosses)

    def check(self, debug=False):
        for ant1 in self.colony1.ants:
            for ant2 in self.colony2.ants:
                if ant1.actual == ant2.actual:
                    return ant1, ant2
        return None, None


def areSameKnotAnt3(
    knot1: CustomKnot,
    knot2: CustomKnot,
    numberOfAnts=100,
    maxCrosses=100,
    generationLimit=1000,
    debug=False,
    timeLimit: float = float("inf"),
) -> tuple[bool, list[str]]:
    colony = DoubleColony(knot1, knot2, numberOfAnts)
    generation = 0
    startTime = time()
    while generation < generationLimit:
        if debug:
            print("Generation: ", generation)
            print(len(colony.colony1.ants))
            print(len(colony.colony2.ants))
        generation += 1

        ant1, ant2 = colony.check(debug=debug)
        if ant1 != None and ant2 != None:
            return True, ant1.movements, ant2.movements

        colony.randomMov(maxCrosses)

        if (time() - startTime) > timeLimit:
            break
    return False, []
