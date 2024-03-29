from __future__ import annotations
from time import time
import numpy as np
from DrawingKnots import *
from AuxiliarFunctions import *
from Path import *
from X import *
from PlanarDiagram import *
from enum import Enum
from random import randrange, choice, shuffle


from pyknotid.catalogue.getdb import download_database
from pyknotid.catalogue import get_knot, from_invariants

# Uncomment the following line the first time it is imported.
# download_database()


class StrandType(Enum):
    ABOVE = 0
    MIDDLE = 1
    BELOW = 2


class CustomKnot:
    def __init__(self, xs: List[X | int] | str):
        """A knot is a mathematical object, with this class we try to model it."""
        if type(xs) == str:
            xs = eval(xs)
        if type(xs) == type([]):
            if len(xs) == 0:
                self.crosses = []
            else:
                if type(xs[0]) == type(1):
                    self._initFromExtendedGaussCode(xs)
                elif type(xs[0]) == type(X(0, 0, 0, 0)):
                    self.crosses: List[X] = sorted(xs)
        else:
            raise Exception("Incorrect type")
        self.pd = None
        self.pdz = None
        self._reprForHash = None
        self._image = None
        self.isKnotValid()

    def _initFromExtendedGaussCode(self, extendedGaussCode: List[int]):
        self.crosses = [X(0, 0, 0, 0) for i in range(len(extendedGaussCode) // 2)]
        for i in range(0, len(extendedGaussCode)):
            codeStrand = extendedGaussCode[i]
            strand = i + 1
            isNegative = codeStrand < 0
            isFirstCodeStrand = not abs(codeStrand) in [
                abs(extendedGaussCode[j])
                for j in range(0, len(extendedGaussCode))
                if j < i
            ]
            cross = self.crosses[abs(codeStrand) - 1]
            if isFirstCodeStrand:
                if isNegative:
                    cross[0] = strand
                    cross[2] = mod(strand + 1, len(extendedGaussCode))
                else:
                    cross[1] = strand
                    cross[3] = mod(strand + 1, len(extendedGaussCode))
            else:
                cross.completeWith(strand, not isNegative, len(extendedGaussCode))

    def __repr__(self):
        return (
            "Cross: "
            + self.crosses.__repr__()
            + "\nPlanar Diagrams:\n"
            + np.array2string(self.planarDiagrams(), suppress_small=True)
        )

    def forMathematica(self):
        aux = "PD["
        for cross in self.crosses:
            aux += cross.forMathematica()
            aux += ","
        aux = aux[: len(aux) - 1]
        aux += "]"

        return aux

    def __eq__(self, obj: CustomKnot):
        if type(self) != type(obj):
            return False
        if len(self.crosses) != len(obj.crosses):
            return False
        if len(self.crosses) == 0 and len(obj.crosses) == 0:
            return True
        n = self.numberOfStrands
        uno = deepcopy(self)
        for inv in range(2):
            for rot in range(n):
                if not sum([not i in uno.crosses for i in obj.crosses]):
                    # print("inv:",inv,", rot:",rot)
                    return True
                uno.rotate(n=n)
            uno.inverse(n=n)
        return False

    def __contains__(self, key: X | List[X]):
        if type(key) == type(X(0, 0, 0, 0)):
            n = self.numberOfStrands
            cross = X(
                mod(key.strands[0], n),
                mod(key.strands[1], n),
                mod(key.strands[2], n),
                mod(key.strands[3], n),
            )
            return cross in self.crosses
        if type(key) == type([X(0, 0, 0, 0)]):
            for cross in key:
                if not cross in self:
                    return False
            return True
        return False

    def eval(self, s: str):
        return eval("self{}".format(s))

    def allRotationYield(self):
        n = self.numberOfStrands
        uno = deepcopy(self)
        if n == 0:
            yield uno
        for i in range(2):
            for r in range(n):
                # print("i:",i,"r:",r)
                yield deepcopy(uno)
                uno.rotate(n=n)
            uno.inverse(n=n)

    def allRotation(self) -> List[CustomKnot]:
        aux = []
        n = self.numberOfStrands
        uno = deepcopy(self)
        for _ in range(2):
            for _ in range(n):
                aux.append(deepcopy(uno))
                uno.rotate(n=n)
            uno.inverse(n=n)
        return aux

    @property
    def representationForHash(self) -> str:
        if self._reprForHash != None:
            return deepcopy(self._reprForHash)
        s = set([k.crosses.__repr__() for k in self.allRotation()])
        a = list(s)
        a = sorted(a)
        self._reprForHash = a.__repr__()
        return deepcopy(self.representationForHash)

    def __hash__(self):
        # start = time()
        h = hash(self.representationForHash)
        # end = time()
        # print("Time hash",end - start)
        return h

    @property
    def numberOfStrands(self):
        """It returns the number of strands."""
        return len(set([strand for cross in self.crosses for strand in cross.strands]))

    @property
    def gaussCode(self):
        code: List[int] = []
        for i in range(1, self.numberOfStrands + 1):
            cross = crossWithStrandAndDirection(self, i)[0]
            codeStrand = self.crosses.index(cross) + 1
            if not cross.isStrandAbove(i):
                codeStrand *= -1
            code.append(codeStrand)
        return code

    @property
    def extendedGaussCode(self):
        extendedCode: List[int] = []
        for strand in range(1, self.numberOfStrands + 1):
            cross = crossWithStrandAndDirection(self, strand)[0]
            codeStrand = self.crosses.index(cross) + 1
            if not (codeStrand in [abs(c) for c in extendedCode]):
                if not cross.isStrandAbove(strand):
                    codeStrand *= -1
            else:
                if not cross.isPositive(self.numberOfStrands):
                    codeStrand *= -1
            extendedCode.append(codeStrand)
        return extendedCode

    def isKnotValid(self):
        """Generates an error if the knot is not correctly created."""
        for cross in self.crosses:
            cross.sort()
        self.crosses = sorted(self.crosses)
        (unique, counts) = np.unique(
            [l for cross in self.crosses for l in cross.strands], return_counts=True
        )
        for c in counts:
            if c != 2:
                print(self.crosses)
                print(unique)
                print(counts)
                raise Exception("Knot bad generated. Number of strands.")
        if not all(unique == range(1, 2 * len(self.crosses) + 1)):
            raise Exception("Knot bad generated. Strands concordance.")

    def rotate(self, i=1, n: int = None):
        """Rotate the strands."""
        if n == None:
            n = self.numberOfStrands
        for i_c in range(len(self.crosses)):
            for i_p in range(len(self.crosses[i_c].strands)):
                self.crosses[i_c].strands[i_p] = mod(
                    (self.crosses[i_c].strands[i_p] + i), n
                )
        self.isKnotValid()
        self.pd = None
        self.pdz = None

    def inverse(self, inv=True, n: int = None):
        """Reverses the order of the strands."""
        if inv:
            if n == None:
                n = self.numberOfStrands
            for i_c in range(len(self.crosses)):
                for i_p in range(len(self.crosses[i_c].strands)):
                    self.crosses[i_c].strands[i_p] = (
                        n + 1 - self.crosses[i_c].strands[i_p]
                    )
            self.isKnotValid()
            self.pd = None
            self.pdz = None

    def sortedCrossesForCodeDowkerThistlethwaite(self):
        n = self.numberOfStrands
        crossesSort: List[X] = []
        for l in range(1, n + 1):
            crossesWithS = crossesWithStrand(self, l)
            for cross in crossesWithS:
                i = cross.indexForStrand(l)[0]
                cross.strands[(i + 2) % 4]

                next = cross.strands[(i + 2) % 4]

                if mod(l + 1, n) == next:
                    crossesSort.append(cross)
        return crossesSort

    def dictCrossesForCodeDowkerThistlethwaite(self):
        n = self.numberOfStrands
        crosses = self.crosses.copy()
        # Ojo que son los mismo objetos, no debería afectar porque no os vamos a modificar
        crossesSort = self.sortedCrossesForCodeDowkerThistlethwaite()
        dictAux = {cross: (None, None) for cross in crosses}
        for i in range(1, n + 1):
            cross = crossesSort[i - 1]
            if i % 2 == 0:
                if cross.isStrandAbove(i):
                    i = -i
            if dictAux[cross][0] == None:
                dictAux[cross] = (i, None)
            else:
                dictAux[cross] = (dictAux[cross][0], i)
        return dictAux

    def codeDowkerThistlethwaite(self):
        arrayAux = [
            (l1, l2) if l1 % 2 == 1 else (l2, l1)
            for (l1, l2) in self.dictCrossesForCodeDowkerThistlethwaite().values()
        ]
        arrayAux = sorted(arrayAux, key=lambda x: x[0])
        arrayAux = [l2 for (l1, l2) in arrayAux]
        return arrayAux

    def strandsConnectedWith(
        self, strand: Strand, clockWise: bool
    ) -> List[Tuple[Strand, bool]]:
        n = self.numberOfStrands
        if strand < 1 or strand > n:
            raise Exception("Invalid strand")
        auxList: List[Tuple[Strand, bool]] = [(strand, True)]
        while True:
            previousStrand, previousDirection = auxList[-1]
            cross, i = crossWithStrandAndDirection(
                self, previousStrand, previousDirection
            )
            i = (i + (1 if clockWise else -1)) % 4
            nextStrand = cross.strands[i]
            nextDirection = cross.strands[(i + 2) % 4] == mod(nextStrand - 1, n)
            if nextStrand == strand:
                break
            auxList.append((nextStrand, nextDirection))
        return auxList

    def hasLoopInStrandsConnected(self, strandsConnected: List[Strand]) -> bool:
        n = len(strandsConnected)
        for i in range(n):
            l = strandsConnected[i]
            lNext = strandsConnected[(i + 1) % n]
            if (
                l == mod(lNext + 1, self.numberOfStrands)
                or l == mod(lNext - 1, self.numberOfStrands)
                or l == mod(lNext + 2, self.numberOfStrands)
                or l == mod(lNext - 2, self.numberOfStrands)
            ):
                return True
        return False

    def typeOfStrand(self, strand: Strand):
        crosses = crossesWithStrand(self, strand)
        if len(crosses) == 2:
            ab0 = crosses[0].isStrandAbove(strand)
            ab1 = crosses[1].isStrandAbove(strand)
            if ab0 and ab1:
                return StrandType.ABOVE
            elif not ab0 and not ab1:
                return StrandType.BELOW
            else:
                return StrandType.MIDDLE
        elif len(crosses) == 1:
            return StrandType.MIDDLE
        else:
            return None

    def planarDiagrams(self) -> np.ndarray:
        """It returns us a matrix that is the planar representation of the knot."""
        self._image = None
        crossesCopy = self.crosses.copy()
        if len(crossesCopy) == 0:  # Estamos ante un no-nudo sin cruces
            return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
        if type(self.pd) == type(None) or np.all((self.pd == 0)):
            cross = crossesCopy.pop(crossWithStrand(crossesCopy, 1))
            matrix = np.zeros((3, 3), dtype=int)
            strands = cross.strands
            # La representación del n de en medio sera -1 si pasa por arriba de izquierda a derecha y
            #                                         -2 si pasa por abajo  de izquierda a derecha
            matrix[1, 1] = -2
            matrix[1, 0] = strands[0]
            matrix[0, 1] = strands[-1 % 4]
            matrix[1, 2] = strands[-2 % 4]
            matrix[2, 1] = strands[-3 % 4]
        else:
            crossesCopy = [
                cross
                for cross in crossesCopy
                if type(findCrossInPd(cross, self.pd)) == type(None)
            ]
            matrix = self.pd
            # print(matrix)
            # print(crossesCopy)
        pd, lengths = connectPlanarDiagram(self, matrix, crossesCopy)
        pd, lengths = compactPlanarDiagram(pd, lengths)
        pd = removeBorderOfZeros(pd)
        self.pd = pd
        return copy(pd)

    def image(self):
        """Returns an image of the planar representation."""
        if type(self._image) != type(None):
            return self._image
        matrix = self.planarDiagrams()
        matrix = borderByZeros(matrix)
        image = np.zeros((0, 0), dtype=float)
        for r in range(0, matrix.shape[0]):
            imageRow = np.zeros((0, 0), dtype=float)
            for c in range(0, matrix.shape[1]):
                if matrix[r, c] == -1:
                    imageRow = concat(imageRow, cross1, axis=1)
                elif matrix[r, c] == -2:
                    imageRow = concat(imageRow, cross2, axis=1)
                elif matrix[r, c] == 0:
                    imageRow = concat(imageRow, blank, axis=1)
                else:
                    number = matrix[r, c]
                    left = number if matrix[r, c - 1] < 0 else matrix[r, c - 1]
                    up = number if matrix[r - 1, c] < 0 else matrix[r - 1, c]
                    right = number if matrix[r, c + 1] < 0 else matrix[r, c + 1]
                    down = number if matrix[r + 1, c] < 0 else matrix[r + 1, c]
                    if left == number and right == number:
                        imageRow = concat(imageRow, leftRight, axis=1)
                    elif up == number and down == number:
                        imageRow = concat(imageRow, upDown, axis=1)
                    elif up == number and right == number:
                        imageRow = concat(imageRow, upRight, axis=1)
                    elif up == number and left == number:
                        imageRow = concat(imageRow, upLeft, axis=1)
                    elif down == number and right == number:
                        imageRow = concat(imageRow, downRight, axis=1)
                    elif down == number and left == number:
                        imageRow = concat(imageRow, downLeft, axis=1)
                    else:
                        imageRow = concat(imageRow, blank, axis=1)
            image = concat(image, imageRow, axis=0)
        self._image = image
        return self.image()

    def simplePlanarDiagram(self: CustomKnot):
        matrix = self.planarDiagrams()
        matrix = borderByZeros(matrix)
        matrix = createGaps(matrix)
        matrix = removeBorderOfZeros(matrix)
        for i in range(1, self.numberOfStrands + 1):
            matrix = replace(matrix, i, 1)
        return matrix

    def createALoop(self, l, typ, recalculatePd=False):
        """Create a loop of Reidemeister's first move."""
        if len(self.crosses) == 0:
            self.pd = None
            self._image = None
            self._reprForHash = None
            self.crosses.append(X(1, 2, 2, 1))
        else:
            # Check si este strand esta en el nudo
            n = self.numberOfStrands
            if l < 1 or l > n:
                raise Exception("This strand not in the knot")
            if n == 2:
                auxCross = [
                    X(1, 4, 2, 1),
                    X(1, 4, 2, 1),
                    X(1, 4, 2, 1),
                    X(1, 4, 2, 1),
                    X(3, 3, 4, 2),
                    X(2, 2, 3, 1),
                    X(2, 1, 3, 2),
                    X(1, 2, 2, 3),
                    X(1, 3, 2, 2),
                    X(3, 4, 4, 1),
                ]
                self.crosses = [
                    auxCross[typ % 4 + (l % 2) * 5],
                    auxCross[4 + (l % 2) * 5],
                ]
                self.pd = None
                self._image = None
            else:
                for i in range(len(self.crosses)):
                    strandsCrossCopy = copy(self.crosses[i].strands)
                    for j in range(len(self.crosses[i].strands)):
                        if self.crosses[i].strands[j] > l:
                            self.crosses[i].strands[j] += 2
                        elif self.crosses[i].strands[j] == l:
                            if strandsCrossCopy[(j + 2) % 4] == mod(l + 1, n):
                                self.crosses[i].strands[j] += 2
                if recalculatePd:
                    pdCopy = deepcopy(self.pd)
                    for i in range(1, n + 1):
                        if i == l:
                            for ind in indicesOfNumberInMatrix(self.pd, l):
                                cape = isCapeOfCross(self.pd, ind)
                                if cape == -1:
                                    pdCopy[ind] = 0
                                elif self.pd[uDLF(cape, uDLF(cape, ind))] == mod(
                                    l + 1, n
                                ):
                                    pdCopy[ind] = l + 2
                        elif i > l:
                            for ind in indicesOfNumberInMatrix(self.pd, i):
                                pdCopy[ind] = i + 2
                    self.pd = pdCopy
                n = self.numberOfStrands + 2
                auxCross = [
                    X(mod(l + 1, n), l, mod(l + 2, n), mod(l + 1, n)),
                    X(l, mod(l + 1, n), mod(l + 1, n), mod(l + 2, n)),
                    X(l, mod(l + 2, n), mod(l + 1, n), mod(l + 1, n)),
                    X(mod(l + 1, n), mod(l + 1, n), mod(l + 2, n), l),
                ]
                self.crosses.append(auxCross[typ % 4])
        if recalculatePd:
            self.planarDiagrams()
        else:
            self.pd = None

        self.isKnotValid()
        self.pdz = None
        self._reprForHash = None
        self._image = None

    def isPosibleUndoALoop(self, cross: X | Strand):
        """It tells us if it is possible to unLoop Reidemeister's first move on a cross."""
        if type(cross) == type(X(0, 0, 0, 0)):
            (unique, counts) = np.unique(cross.strands, return_counts=True)
            return max(counts) > 1
        elif type(cross) == type(1):
            return len(crossesWithStrand(self, cross)) == 1
        raise Exception("Incorrect type")

    def undoALoop(self, l: Strand | X, recalculatePd=False):
        """UnLoop Reidemeister's first move in the strand labeled l"""
        if type(l) == Strand:
            for i in range(len(self.crosses)):
                if l in self.crosses[i].strands:
                    if self.crosses[i].strands.count(l) == 2:
                        break
        elif type(l) == X:
            if l not in self:
                Exception("This cross not in the knot")
            for i in range(len(self.crosses)):
                if l == self.crosses[i]:
                    break
        else:
            Exception("Type incorrect")
        if self.isPosibleUndoALoop(self.crosses[i]):
            crossWithLoop = self.crosses[i]
            (unique, counts) = np.unique(self.crosses[i].strands, return_counts=True)
            aux = np.argwhere(counts == 2)[0]
            l = unique[aux][0]
            n = self.numberOfStrands
            self.crosses.remove(crossWithLoop)
            for i in range(len(self.crosses)):
                for j in range(len(self.crosses[i].strands)):
                    if l == 1:
                        if self.crosses[i].strands[j] > 2:
                            self.crosses[i].strands[j] = self.crosses[i].strands[j] - 2
                        elif self.crosses[i].strands[j] == 2:
                            self.crosses[i].strands[j] = n - 2
                    elif l == n:
                        if self.crosses[i].strands[j] == l - 1:
                            self.crosses[i].strands[j] = 1
                    else:
                        if self.crosses[i].strands[j] > l:
                            self.crosses[i].strands[j] = self.crosses[i].strands[j] - 2
            if recalculatePd:
                pdCopy = deepcopy(self.pd)
                pdCopy = removeCrossOfPD(pdCopy, crossWithLoop)
                for i in range(1, n + 1):
                    if i == l:
                        for ind in indicesOfNumberInMatrix(self.pd, i):
                            pdCopy[ind] = 0
                    elif i == mod(l - 1, n):
                        for ind in indicesOfNumberInMatrix(self.pd, i):
                            cape = isCapeOfCross(self.pd, ind)
                            if cape == -1:
                                pdCopy[ind] = 0
                            elif self.pd[uDLF(cape, uDLF(cape, ind))] == mod(l - 2, n):
                                pdCopy[ind] = mod(l - 1, n - 2)
                    elif i == mod(l + 1, n):
                        for ind in indicesOfNumberInMatrix(self.pd, i):
                            cape = isCapeOfCross(self.pd, ind)
                            if cape == -1:
                                pdCopy[ind] = 0
                            elif self.pd[uDLF(cape, ind)] < 0 and self.pd[
                                uDLF(cape, uDLF(cape, ind))
                            ] == mod(l + 2, n):
                                pdCopy[ind] = mod(l - 1, n - 2)
                    elif i > l + 1:
                        for ind in indicesOfNumberInMatrix(self.pd, i):
                            pdCopy[ind] = mod(pdCopy[ind] - 2, n - 2)
                if self.numberOfStrands < 3:
                    self.pd = None
                else:
                    self.pd = pdCopy
                self.planarDiagrams()
            else:
                self.pd = None
            self.pdz = None
            self._reprForHash = None
            self._image = None
            self.isKnotValid()
            return True
        return False

    def isPosibleCreateReidemeisterII2(self, l1: Strand, l2: Strand, debug=False):
        """It will return 0 if it is not possible to make Reidemeister's second move,
        will return 2 or 3 if they have opposite directions, depending on the type,
        will return 4 or 5 if they have the same addresses, depending on the type"""
        if debug:
            print("isPosibleCreateReidemeisterII(", l1, ",", l2, ")")
        if l2 <= l1:
            return 0
        if len(self.crosses) == 1:
            if (l1 == 1 and l2 == 2) or (l1 == 1 and l2 == 1):
                return 1
        pd = self.planarDiagrams()
        pd = borderByZeros(pd)
        pd = createGaps(pd)
        _, pd2, org, des = connect(
            pd, l1, l2, caminoNumber=-9, corner=True, oneOriginDestiny=True, debug=debug
        )
        if type(pd2) == type(None):
            return 0
        d1 = direction(self, pd, org)
        d2 = direction(self, pd, des)
        visited, dR = walkWithDirection(pd2, org, des, d1)
        v1 = visited[1]
        u0 = up(visited[0])
        r0 = right(visited[0])
        d0 = down(visited[0])
        l0 = left(visited[0])
        if d2 != dR:
            if (
                (d1 == 0 and v1 == l0)
                or (d1 == 1 and v1 == u0)
                or (d1 == 2 and v1 == r0)
                or (d1 == 3 and v1 == d0)
            ):
                return 4
            if (
                (d1 == 0 and v1 == r0)
                or (d1 == 1 and v1 == d0)
                or (d1 == 2 and v1 == l0)
                or (d1 == 3 and v1 == u0)
            ):
                return 5
        else:
            # Aquí tenemos que comprobar los otros dos casos
            if (
                (d1 == 0 and v1 == r0)
                or (d1 == 1 and v1 == d0)
                or (d1 == 2 and v1 == l0)
                or (d1 == 3 and v1 == u0)
            ):
                return 2
            if (
                (d1 == 0 and v1 == l0)
                or (d1 == 1 and v1 == u0)
                or (d1 == 2 and v1 == r0)
                or (d1 == 3 and v1 == d0)
            ):
                return 3
        print(pd2)
        print("d1:", d1)
        print("d2:", d2)
        print("dR:", dR)
        print("v0", visited[0])
        print("v1", v1)
        print("u0", u0)
        print("r0", r0)
        print("d0", d0)
        print("l0", l0)
        raise Exception("No se debería generar este error")

    def isPosibleCreateReidemeisterII(self, l1: Strand, l2: Strand, debug=False):
        """It will return 0 if it is not possible to make Reidemeister's second move,
        will return 2 or 3 if they have opposite directions, depending on the type,
        will return 4 or 5 if they have the same addresses, depending on the type"""
        # Es una mejora del anterior sin el uso del planar Diagram
        if debug:
            print("isPosibleCreateReidemeisterII(", l1, ",", l2, ")")
        if len(self.crosses) == 1:
            if (l1 == 1 and l2 == 2) or (l1 == 1 and l2 == 1):
                return 1
        if l2 <= l1:
            return 0
        strandsConnected = self.strandsConnectedWith(l1, True)
        for l, sameDirection in strandsConnected:
            if l == l2:
                # Esta parte es solo para buscar un posible nudo interesante
                # otherClock = [l for (l,_) in self.strandsConnectedWith(l1,False)]
                # if l2 in otherClock and not self.hasLoopInStrandsConnected([l for l,_ in strandsConnected]) and not self.#hasLoopInStrandsConnected(otherClock):
                #    print("este nudo es interesante, más")
                #    print(l1,l2)
                #    print(self)
                if sameDirection:
                    return 5
                else:
                    return 2
        strandsConnected = self.strandsConnectedWith(l1, False)
        for l, sameDirection in strandsConnected:
            if l == l2:
                if sameDirection:
                    return 4
                else:
                    return 3
        return 0

    def createReidemeisterII(
        self,
        l1: Strand,
        l2: Strand,
        orientation,
        reidemeister=None,
        recalculatePd=False,
        debug=False,
    ):
        """Create a second Reidemeister move."""
        if len(self.crosses) == 1:
            if (l1 == 1 and l2 == 2) or (l1 == 1 and l2 == 1):
                if orientation:
                    self.crosses = [X(3, 1, 4, 6), X(5, 3, 6, 2), X(4, 1, 5, 2)]
                else:
                    self.crosses = [X(1, 5, 2, 4), X(2, 5, 3, 6), X(3, 1, 4, 6)]
                self.isKnotValid()
                self.pd = None
                self.pdz = None
                self._reprForHash = None
                self._image = None
                return True
            else:
                if debug:
                    print("No es posible crear un reidemeister.")
                return False
        if reidemeister == None:
            reidemeister = self.isPosibleCreateReidemeisterII(l1, l2, debug=debug)
        if reidemeister != 0:
            for c in range(len(self.crosses)):
                crossCopy = copy(self.crosses[c].strands)
                for p in range(len(self.crosses[c].strands)):
                    if self.crosses[c].strands[p] == l1:
                        if crossCopy[(p + 2) % 4] == l1 + 1:
                            self.crosses[c].strands[p] = self.crosses[c].strands[p] + 2
                    elif (
                        self.crosses[c].strands[p] > l1
                        and self.crosses[c].strands[p] < l2
                    ):
                        self.crosses[c].strands[p] = self.crosses[c].strands[p] + 2
                    elif self.crosses[c].strands[p] == l2:
                        if crossCopy[(p + 2) % 4] == l2 - 1:
                            self.crosses[c].strands[p] = self.crosses[c].strands[p] + 2
                        else:
                            self.crosses[c].strands[p] = self.crosses[c].strands[p] + 4
                    elif self.crosses[c].strands[p] > l2:
                        self.crosses[c].strands[p] = self.crosses[c].strands[p] + 4
                    if debug:
                        print(crossCopy[p], "--->", self.crosses[c].strands[p])
            n = self.numberOfStrands

            if reidemeister == 2:
                if debug:
                    print("Tienen la misma dirección, tipo 2")
                if orientation:
                    self.crosses.append(X(l2 + 2, l1 + 1, l2 + 3, l1))
                    self.crosses.append(X(l2 + 3, l1 + 1, l2 + 4, l1 + 2))
                else:
                    self.crosses.append(X(l1, l2 + 2, l1 + 1, l2 + 3))
                    self.crosses.append(X(l1 + 1, l2 + 4, l1 + 2, l2 + 3))

            elif reidemeister == 3:
                if debug:
                    print("Tienen la misma dirección, tipo 3")
                if orientation:
                    self.crosses.append(X(l1, l2 + 3, l1 + 1, l2 + 2))
                    self.crosses.append(X(l1 + 1, l2 + 3, l1 + 2, l2 + 4))
                else:
                    self.crosses.append(X(l2 + 2, l1, l2 + 3, l1 + 1))
                    self.crosses.append(X(l2 + 3, l1 + 2, l2 + 4, l1 + 1))
            elif reidemeister == 4:
                if debug:
                    print("Tienen distinta dirección, tipo 4")
                if orientation:
                    self.crosses.append(X(l2 + 2, l1 + 1, l2 + 3, l1 + 2))
                    self.crosses.append(X(l2 + 3, l1 + 1, l2 + 4, l1))
                else:
                    self.crosses.append(X(l1 + 2, l2 + 2, l1 + 1, l2 + 3))
                    self.crosses.append(X(l1, l2 + 3, l1 + 1, l2 + 4))
            elif reidemeister == 5:
                if debug:
                    print("Tienen distinta dirección, tipo 5")
                if orientation:
                    self.crosses.append(X(l2 + 4, l1 + 1, l2 + 3, l1))
                    self.crosses.append(X(l2 + 3, l1 + 1, l2 + 2, l1 + 2))
                else:
                    self.crosses.append(X(l1, l2 + 4, l1 + 1, l2 + 3))
                    self.crosses.append(X(l1 + 1, l2 + 2, l1 + 2, l2 + 3))
            self.isKnotValid()
            if recalculatePd:
                pdCopy = deepcopy(self.pd)
                for i in range(l1, n + 1):
                    for ind in indicesOfNumberInMatrix(self.pd, i):
                        if i == l1:
                            cape = isCapeOfCross(self.pd, ind)
                            if cape == -1:
                                pdCopy[ind] = 0
                            else:
                                if pdCopy[uDLF(cape, uDLF(cape, ind))] == l1 + 1:
                                    pdCopy[ind] = pdCopy[ind] + 2
                        elif i > l1 and i < l2:
                            pdCopy[ind] = pdCopy[ind] + 2
                        elif i == l2:
                            cape = isCapeOfCross(self.pd, ind)
                            if cape == -1:
                                pdCopy[ind] = 0
                            else:
                                if self.pd[uDLF(cape, uDLF(cape, ind))] == mod(
                                    l2 + 1, n
                                ):
                                    pdCopy[ind] = pdCopy[ind] + 4
                                else:
                                    pdCopy[ind] = pdCopy[ind] + 2
                        elif i > l2:
                            pdCopy[ind] = pdCopy[ind] + 4
                self.pd = pdCopy
                self.planarDiagrams()
            else:
                self.pd = None
            self.pdz = None
            self._reprForHash = None
            self._image = None
            return True
        else:
            if debug:
                print("No ha sido posible", reidemeister)
            return False

    def isPosibleUndoReidemeisterII(self, l1: Strand, l2: Strand):
        """It tells us if it is possible to undo a second Reidemeister move, it will return the two crosses involved in the move if it is possible and None if it is not possible."""
        n = self.numberOfStrands
        x1 = X(l1, l2, mod(l1 + 1, n), mod(l2 + 1, n))
        x2 = X(l1, mod(l2 - 1, n), mod(l1 - 1, n), l2)
        if x1 in self.crosses and x2 in self.crosses:
            return (x1, x2)
        x1 = X(l2, mod(l1 + 1, n), mod(l2 + 1, n), l1)
        x2 = X(mod(l2 - 1, n), mod(l1 - 1, n), l2, l1)
        if x1 in self.crosses and x2 in self.crosses:
            return (x1, x2)
        x1 = X(l1, l2, mod(l1 + 1, n), mod(l2 - 1, n))
        x2 = X(l1, mod(l2 + 1, n), mod(l1 - 1, n), l2)
        if x1 in self.crosses and x2 in self.crosses:
            return (x1, x2)
        x1 = X(l1, l2, mod(l1 - 1, n), mod(l2 + 1, n))
        x2 = X(l1, mod(l2 - 1, n), mod(l1 + 1, n), l2)
        if x1 in self.crosses and x2 in self.crosses:
            return (x1, x2)
        return None

    def undoReidemeisterII(
        self, l1: Strand, l2: Strand, recalculatePd=False, debug=False
    ):
        """Undo a second move from Reidemeister, on strands l1 and l2."""
        x1x2 = self.isPosibleUndoReidemeisterII(l1, l2)
        if type(x1x2) == type(None):
            x1x2 = self.isPosibleUndoReidemeisterII(l2, l1)
            if type(x1x2) == type(None):
                if debug:
                    print("No se puede deshacer Reidemeister II", l1, l2)
                return False
        n = self.numberOfStrands
        lm = min(l1, l2)
        lM = max(l1, l2)
        (x1, x2) = x1x2
        if debug:
            print("Los cruces que quitamos son", x1, x2)
        self.crosses.remove(x1)
        self.crosses.remove(x2)
        for c in range(len(self.crosses)):
            crossCopy = copy(self.crosses[c].strands)
            for p in range(len(self.crosses[c].strands)):
                strand = crossCopy[p]
                if strand >= lm + 1 and strand <= lM - 1:
                    self.crosses[c].strands[p] = mod(strand - 2, n - 4)
                elif strand >= lM + 1:
                    self.crosses[c].strands[p] = mod(strand - 4, n - 4)
                else:
                    self.crosses[c].strands[p] = mod(strand, n - 4)
                if debug:
                    print(strand, "--", self.crosses[c].strands[p])
        self.isKnotValid()
        if recalculatePd:
            self.pd = removeCrossOfPD(self.pd, x1)
            self.pd = removeCrossOfPD(self.pd, x2)
            pdCopy = deepcopy(self.pd)
            for i in range(max(lm - 1, 1), n + 1):
                for ind in indicesOfNumberInMatrix(self.pd, i):
                    cape = isCapeOfCross(self.pd, ind)
                    if mod(i, n) == mod(lm - 1, n):
                        if cape == -1:
                            pdCopy[ind] = 0
                        else:
                            pdCopy[ind] = mod(lm - 1, n - 4)
                    elif i == lm:
                        pdCopy[ind] = 0
                    elif i == lm + 1:
                        if cape == -1:
                            pdCopy[ind] = 0
                        else:
                            pdCopy[ind] = mod(pdCopy[ind] - 2, n - 4)
                    elif i < lM - 1:
                        pdCopy[ind] = mod(pdCopy[ind] - 2, n - 4)
                    elif i == lM - 1:
                        if cape == -1:
                            pdCopy[ind] = 0
                        else:
                            pdCopy[ind] = mod(pdCopy[ind] - 2, n - 4)
                    elif i == lM:
                        pdCopy[ind] = 0
                    elif mod(i, n) == mod(lM + 1, n):
                        if cape == -1:
                            pdCopy[ind] = 0
                        else:
                            pdCopy[ind] = mod(pdCopy[ind] - 4, n - 4)
                    else:
                        pdCopy[ind] = mod(pdCopy[ind] - 4, n - 4)
            self.pd = pdCopy
            self.planarDiagrams()
        else:
            self.pd = None
        self.pdz = None
        self._reprForHash = None
        self._image = None
        return True

    def auxIsPosibleReidemeisterIII(self, s1: Strand, s2: Strand, s3: Strand):
        n = self.numberOfStrands
        if (
            s1 == mod(s2 + 1, n)
            or s1 == mod(s2 - 1, n)
            or s1 == mod(s3 + 1, n)
            or s1 == mod(s3 - 1, n)
            or s3 == mod(s2 + 1, n)
            or s3 == mod(s2 - 1, n)
        ):
            return False, None, None, None
        t1 = self.typeOfStrand(s1)
        t2 = self.typeOfStrand(s2)
        t3 = self.typeOfStrand(s3)
        strandBelow = None  # Debajo
        strandMiddle = None  # Medio
        strandAbove = None  # Encima
        for i in range(3):
            t = [t1, t2, t3][i]
            s = [s1, s2, s3][i]
            if t == StrandType.BELOW:
                strandBelow = s
            elif t == StrandType.MIDDLE:
                strandMiddle = s
            elif t == StrandType.ABOVE:
                strandAbove = s
        if strandBelow != None and strandMiddle != None and strandAbove != None:
            return True, strandBelow, strandMiddle, strandAbove
        return False, None, None, None

    def reidemeisterIII2(
        self,
        s1: Strand,
        s2: Strand,
        s3: Strand,
        recalculatePd=False,
        debug=False,
        check=True,
    ):
        """It makes a Reidemeister move of the third type."""
        if check:
            posible, B, M, A = self.auxIsPosibleReidemeisterIII(s1, s2, s3)
            if not posible:
                if debug:
                    print("no es posible hacer este movimiento")
                return False
        else:
            B, M, A = s1, s2, s3
        n = self.numberOfStrands
        xsPrueba = {}
        i = 0
        for s1 in [-1, +1]:
            for s2 in [-1, +1]:
                for s3 in [-1, +1]:
                    i += 1
                    xsPrueba[i] = [
                        X(B, A, mod(B + s1, n), mod(A + s2, n)),
                        X(B, mod(M + s3, n), mod(B - s1, n), M),
                        X(M, mod(A - s2, n), mod(M - s3, n), A),
                    ]
                    xsPrueba[i + 10] = [
                        X(M, A, mod(M + s1, n), mod(A + s2, n)),
                        X(B, M, mod(B + s3, n), mod(M - s1, n)),
                        X(B, mod(A - s2, n), mod(B - s3, n), A),
                    ]

        for key, xs in xsPrueba.items():
            if xs in self:
                if len(set(xs)) != 3:
                    continue
                self.crosses = [cross for cross in self.crosses if cross not in xs]
                associatedKey = 9 - key if key < 10 else 29 - key
                self.crosses += xsPrueba[associatedKey]
                self.isKnotValid()
                if recalculatePd:
                    for x in xs:
                        self.pd = removeCrossOfPD(self.pd, x)
                    for l in [B, A, M]:
                        for ind in indicesOfNumberInMatrix(self.pd, l):
                            self.pd[ind] = 0
                    for l in set(
                        [
                            mod(B + 1, n),
                            mod(B - 1, n),
                            mod(M + 1, n),
                            mod(M - 1, n),
                            mod(A + 1, n),
                            mod(A - 1, n),
                        ]
                    ):
                        for ind in indicesOfNumberInMatrix(self.pd, l):
                            cape = isCapeOfCross(self.pd, ind)
                            if cape == -1:
                                self.pd[ind] = 0
                    self.planarDiagrams()
                else:
                    self.pd = None
                self.pdz = None
                self._reprForHash = None
                self._image = None
                return True
        return False

    def isPosibleReidemeisterIII(
        self,
        s1: Strand,
        s2: Strand,
        s3: Strand,
        recalculatePd=False,
        debug=False,
        check=True,
    ):
        if check:
            posible, B, M, A = self.auxIsPosibleReidemeisterIII(s1, s2, s3)
            if not posible:
                if debug:
                    print("no es posible hacer este movimiento")
                return False, None, None, None, None, None, None
        else:
            B, M, A = s1, s2, s3
        n = self.numberOfStrands
        xsPrueba = {}
        i = 0
        for s1 in [-1, +1]:
            for s2 in [-1, +1]:
                for s3 in [-1, +1]:
                    i += 1
                    xsPrueba[i] = [
                        X(B, A, mod(B + s1, n), mod(A + s2, n)),
                        X(B, mod(M + s3, n), mod(B - s1, n), M),
                        X(M, mod(A - s2, n), mod(M - s3, n), A),
                    ]
                    xsPrueba[i + 10] = [
                        X(M, A, mod(M + s1, n), mod(A + s2, n)),
                        X(B, M, mod(B + s3, n), mod(M - s1, n)),
                        X(B, mod(A - s2, n), mod(B - s3, n), A),
                    ]

        for key, xs in xsPrueba.items():
            if xs in self:
                if len(set(xs)) != 3:
                    continue
                return True, key, xs, xsPrueba, B, M, A

        return False, None, None, None, None, None, None

    def reidemeisterIII(
        self,
        s1: Strand,
        s2: Strand,
        s3: Strand,
        recalculatePd=False,
        debug=False,
        check=True,
    ):
        """It makes a Reidemeister move of the third type."""
        isPosible, key, xs, xsPrueba, B, M, A = self.isPosibleReidemeisterIII(
            s1, s2, s3, recalculatePd, debug, check
        )
        if not isPosible:
            return False
        self.crosses = [cross for cross in self.crosses if cross not in xs]
        associatedKey = 9 - key if key < 10 else 29 - key
        self.crosses += xsPrueba[associatedKey]
        self.isKnotValid()
        if recalculatePd:
            n = self.numberOfStrands
            for x in xs:
                self.pd = removeCrossOfPD(self.pd, x)
            for l in [B, A, M]:
                for ind in indicesOfNumberInMatrix(self.pd, l):
                    self.pd[ind] = 0
            for l in set(
                [
                    mod(B + 1, n),
                    mod(B - 1, n),
                    mod(M + 1, n),
                    mod(M - 1, n),
                    mod(A + 1, n),
                    mod(A - 1, n),
                ]
            ):
                for ind in indicesOfNumberInMatrix(self.pd, l):
                    cape = isCapeOfCross(self.pd, ind)
                    if cape == -1:
                        self.pd[ind] = 0
            self.planarDiagrams()
        else:
            self.pd = None
        self.pdz = None
        self._reprForHash = None
        self._image = None
        return True

    def reduceUnnecessaryMov(self, debug=False):
        while True:
            continueWhile = False
            n = self.numberOfStrands
            for i in range(1, n + 1):
                if self.undoALoop(i):
                    if debug:
                        print("undoALoop({})".format(i))
                    continueWhile = True
                    break
            if continueWhile:
                continue
            possibilities = [
                (l1, l2)
                for l1 in range(1, n + 1)
                for l2 in range(l1, n + 1)
                if (
                    self.typeOfStrand(l1) == StrandType.ABOVE
                    and self.typeOfStrand(l2) == StrandType.BELOW
                )
                or (
                    self.typeOfStrand(l1) == StrandType.BELOW
                    and self.typeOfStrand(l2) == StrandType.ABOVE
                )
            ]
            for l1, l2 in possibilities:
                if self.undoReidemeisterII(l1, l2):
                    if debug:
                        print(".undoReidemeisterII({},{})".format(l1, l2))
                    continueWhile = True
                    break
            if continueWhile:
                continue
            break

    def planarDiagramZones(self):
        """Returns a planar diagram with the zones delimited with numbers and the number of zones.
        Attention: The diagram is surrounded by zeros and Gaps are created."""
        if type(self.pdz) != type(None):
            return self.pdz
        pd = self.planarDiagrams()
        pd = borderByZeros(pd)
        pd = createGaps(pd)
        pd[pd != 0] = -1
        ceroIndices = indicesOfNumberInMatrix(pd, 0)
        i = 0
        while ceroIndices:
            i += 1
            base = ceroIndices[0]
            pd[base] = i
            indConectados = [base]
            cola = PriorityQueue()
            cola.put(
                0,
                Node(
                    pd,
                    base,
                    None,
                    0,
                ),
            )
            while not cola.isEmpty:
                nodo = cola.get()
                for sucesor in nodo.successors(i):
                    if not sucesor.origin in indConectados:
                        indConectados.append(sucesor.origin)
                        cola.put(0, sucesor)
            for ind in indConectados:
                pd[ind] = i
            ceroIndices = indicesOfNumberInMatrix(pd, 0)
        self.pdz = pd
        return self.planarDiagramZones()

    def randomMov(self, maxStrands: int = None, debug=False) -> Tuple[bool, str | None]:
        if debug > 0:
            print("-------")
        n = self.numberOfStrands
        if maxStrands == None:
            maxStrands = max(4 * n, 20)
        if debug > 0:
            print("number Of Strands:", n)
        if debug > 0:
            print(self)
        types = [1, 2, 3]
        shuffle(types)
        while len(types) > 0:
            typeMov = types.pop()
            if typeMov == 1:
                if debug > 0:
                    print(typeMov)
                createOrUndoList = [0, 1] if n < maxStrands else [0]
                shuffle(createOrUndoList)
                while len(createOrUndoList) > 0:
                    createOrUndo = createOrUndoList.pop()
                    if createOrUndo:
                        if debug > 0:
                            print("Create")
                        if n == 0:
                            strandToCreate = 1
                        else:
                            strandToCreate = randrange(1, n + 1)
                        if debug > 0:
                            print("strandToCreate", strandToCreate)
                        orientation = randrange(4)
                        self.createALoop(strandToCreate, orientation)
                        if debug > 0:
                            print("Hecho", strandToCreate, "orientation: ", orientation)
                        return True, ".createALoop({},{})".format(
                            strandToCreate, orientation
                        )
                    else:
                        if debug > 0:
                            print("undo")
                        posibleCross = [
                            cross
                            for cross in self.crosses
                            if self.isPosibleUndoALoop(cross)
                        ]
                        if len(posibleCross) > 0:
                            randomCross = choice(posibleCross)
                            if debug > 0:
                                print("randomCross", randomCross)
                            self.undoALoop(randomCross)
                            if debug > 0:
                                print("Hecho", randomCross)
                            return True, ".undoALoop({})".format(randomCross)
            elif typeMov == 2:
                if debug > 0:
                    print(typeMov)
                createOrUndoList = [0, 1] if n < maxStrands else [0]
                shuffle(createOrUndoList)
                while len(createOrUndoList) > 0:
                    createOrUndo = createOrUndoList.pop()
                    createOrUndo = randrange(2) if n < maxStrands else 0
                    if debug > 0:
                        print("create") if createOrUndo else print("undo")
                    # El segundo puede que sea desde l1?
                    possibilities = [
                        (l1, l2) for l1 in range(1, n + 1) for l2 in range(1, n + 1)
                    ]
                    if not createOrUndo:
                        possibilities = [
                            (l1, l2)
                            for (l1, l2) in possibilities
                            if (
                                self.typeOfStrand(l1) == StrandType.ABOVE
                                and self.typeOfStrand(l2) == StrandType.BELOW
                            )
                            or (
                                self.typeOfStrand(l1) == StrandType.BELOW
                                and self.typeOfStrand(l2) == StrandType.ABOVE
                            )
                        ]
                    shuffle(possibilities)
                    while possibilities:
                        (l1, l2) = possibilities.pop()
                        if createOrUndo:
                            orientation = randrange(2)
                            if debug > 0:
                                print(l1, l2, orientation)
                            if self.createReidemeisterII(l1, l2, orientation):
                                if debug > 0:
                                    print("Hecho", l1, l2, "orientation", orientation)
                                return True, ".createReidemeisterII({},{},{})".format(
                                    l1, l2, orientation
                                )
                        else:
                            if debug > 0:
                                print("intentando:", l1, l2)
                            if self.undoReidemeisterII(l1, l2):
                                if debug > 0:
                                    print("Hecho", l1, l2)
                                return True, ".undoReidemeisterII({},{})".format(l1, l2)
            elif typeMov == 3:
                if debug > 0:
                    print(typeMov)
                possibilities = {
                    StrandType.ABOVE: [],
                    StrandType.MIDDLE: [],
                    StrandType.BELOW: [],
                }

                for l in range(1, n + 1):
                    t = self.typeOfStrand(l)
                    possibilities[t].append(l)
                possibilities = [
                    (l1, l2, l3)
                    for l1 in possibilities[StrandType.BELOW]
                    for l2 in possibilities[StrandType.MIDDLE]
                    for l3 in possibilities[StrandType.ABOVE]
                ]
                possibilities = [
                    (l1, l2, l3)
                    for (l1, l2, l3) in possibilities
                    if mod(l1 + 1, n) != l2
                    and mod(l1 - 1, n) != l2
                    and mod(l1 + 1, n) != l3
                    and mod(l1 - 1, n) != l3
                    and mod(l2 + 1, n) != l3
                    and mod(l2 - 1, n) != l3
                ]
                shuffle(possibilities)
                while possibilities:
                    (l1, l2, l3) = possibilities.pop()
                    if debug > 0:
                        print(l1, l2, l3)
                    if self.reidemeisterIII(l1, l2, l3, check=False):
                        if debug > 0:
                            print("Hecho III")
                        return True, ".reidemeisterIII({},{},{},check=False)".format(
                            l1, l2, l3
                        )
                if debug > 0:
                    print("No se puede hacer")
        return False, ""

    def randomMovN(self, n: int, maxStrands: int = None, percentage=False, debug=False):
        if maxStrands == None:
            maxStrands = max(4 * self.numberOfStrands, 20)
        moves: List[str] = []
        i = 0
        c = 0
        if percentage:
            startTime = time()
        while i < n:
            c += 1
            if percentage:
                elapsedTime = time() - startTime
                percent = ((i + 1) / n) * 100
                totalTime = elapsedTime * 100 / percent
                remainTime = totalTime - elapsedTime
                print("\r", end="")
                print(
                    "percentage randomMovN: {:3.2f}, numberOfStrands: {} ,time remaining: {}                                 ".format(
                        percent, self.numberOfStrands, remainingTimeString(remainTime)
                    ),
                    end="",
                )
            b, mov = self.randomMov(maxStrands, debug)
            if b:
                i += 1
                c = 0
                moves.append(mov)
            if c == 100:
                print("Parece que no se pueden hacer movimientos")
                break
        if percentage:
            print("\n", end="")
        return moves


def knotFromPyknotid(s: str) -> CustomKnot:
    if s == "0_1":
        return CustomKnot([])
    k = get_knot(s)
    crosses = k.planar_diagram.split()
    crossesFine = []
    for cross in crosses:
        if "," in cross[2:]:
            aux = cross[2:].split(",")
            crossesFine.append(X(int(aux[0]), int(aux[1]), int(aux[2]), int(aux[3])))
        else:
            crossesFine.append(
                X(
                    int(cross[2:][0]),
                    int(cross[2:][1]),
                    int(cross[2:][2]),
                    int(cross[2:][3]),
                )
            )
    k = CustomKnot(crossesFine)
    return k


def direction(knot: CustomKnot, pd: np.ndarray, ind: Tuple[int, int]):
    """It will return 0, 1, 2 or 3 depending on the direction the strand is going."""
    number = pd[ind]
    if number <= 0:
        return None
    directions = []
    visited = [ind]
    i = 0
    c = 0
    while True and c < 5:
        if not uDLF(i, ind) in visited:
            if exists(uDLF(i, ind), pd):
                if pd[uDLF(i, ind)] == number or pd[uDLF(i, ind)] < 0:
                    ind = uDLF(i, ind)
                    visited.append(ind)
                    directions.append(i)
                    if pd[ind] < 0:
                        break
                    else:
                        c = 0
        i = (i + 1) % 4
        c += 1
    lastD = directions[-1]
    firstD = directions[0]
    lastInd = visited[-1]
    if number == 1:
        condition = pd[uDLF(lastD, lastInd)] == knot.numberOfStrands
    elif number == knot.numberOfStrands:
        condition = pd[uDLF(lastD, lastInd)] == knot.numberOfStrands - 1
    else:
        condition = pd[uDLF(lastD, lastInd)] < number
    if condition:
        return (firstD + 2) % 4
    return firstD


def crossesWithZone(knot: CustomKnot, planarDiagram, planarZones, zone):
    return [
        cross
        for cross in knot.crosses
        if zone in zonesConnectCross(cross, planarDiagram, planarZones)
    ]


def crossesWithStrand(knot: CustomKnot, strand: Strand) -> List[X]:
    return [cross for cross in knot.crosses if strand in cross]


def crossWithStrandAndDirection(
    knot: CustomKnot, strand: Strand, direction: bool = True
) -> Tuple[X, int]:
    crosses = crossesWithStrand(knot, strand)
    for cross in crosses:
        for i in cross.indexForStrand(strand):
            if cross.strands[(i + 2) % 4] == mod(
                strand + (1 if direction else -1), knot.numberOfStrands
            ):
                return cross, i


def connectPlanarDiagram(
    k: CustomKnot, matrix: np.ndarray, remainCross: List[X], debug=False
) -> Tuple[np.ndarray, Dict[Strand, int]]:
    if debug:
        print("conectaPlanarDiagram")
    queue = PriorityQueue[NodePD]()
    allStrands: List[Strand] = [i for i in range(1, 2 * len(k.crosses) + 1)]
    tuples = [(connected(matrix, strand), strand) for strand in allStrands]
    strandsDict = {strand: l for (c, l), strand in tuples if c}
    strands = [strand for strand in allStrands if strand not in strandsDict.keys()]
    if debug:
        print("unconnected", strands)
    if len(remainCross) == 0 and len(strands) == 0:
        return matrix, strandsDict
    firstNode = NodePD(matrix, strands, remainCross, strandsDict)
    queue.put(firstNode.priority(), firstNode)
    c = 0
    while not queue.isEmpty and c < 100:
        c += 1
        node = queue.get()
        for successor in node.successors(debug):
            d = successor.priority()
            if d == 0:
                return successor.pd, successor.lengths
            else:
                queue.put(d, successor)
    raise Exception("Se han acabado las opciones")
