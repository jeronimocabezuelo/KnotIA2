import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, TypeVar, Generic

from enum import Enum
from typing import Dict
from copy import deepcopy
from AuxiliarFunctions import (
    exists3D,
    borderByZeros,
    createGaps,
    replace,
    up,
    down,
    left,
    right,
    addLayerToMatrix,
)
from X import findCrossInPd
from CustomKnot import CustomKnot

np.set_printoptions(threshold=np.inf, linewidth=np.inf, nanstr="n")
S = TypeVar("S")

# TODO: Esto en realidad no se utiliza en ningún sitio no?


def unionDifference(unions: list[set[S]], differences: list[set[S]] = []):
    aux: set[S] = set()
    for u in unions:
        for e in u:
            aux.add(e)
    for d in differences:
        for e in d:
            aux.discard(e)
    return aux


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2

    def axisDirection(self, direction):
        direction = 0 if direction == -1 else direction
        return AxisDirection(self.value * 2 + direction)

    @staticmethod
    def thatAreNot(axes):
        if type(axes) == list:
            axes += [a.axis for a in axes if type(a) == AxisDirection]
            available_axes = [a for a in list(Axis) if not a in axes]
            return available_axes[0] if len(available_axes) == 1 else available_axes
        elif type(axes) == Axis:
            return [a for a in list(Axis) if a != axes]
        else:
            raise Exception("Invalid type")


class AxisDirection(Enum):
    XU = 0
    XD = 1
    YU = 2
    YD = 3
    ZU = 4
    ZD = 5

    @property
    def axis(self):
        return Axis(self.value // 2)

    @property
    def direction(self):
        return 1 if self.value % 2 == 0 else -1

    @property
    def opposite(self):
        return AxisDirection(self.value + 1 if self.value % 2 == 0 else self.value - 1)


class ThreeDimensionalPoint:
    def __init__(self, point, y=None, z=None):
        self._point: Dict[Axis, int] = {}
        if type(point) == type((0, 0, 0)):
            _z, _x, _y = point
            self._point[Axis.X] = _x
            self._point[Axis.Y] = _y
            self._point[Axis.Z] = _z
        elif type(point) == type([0, 0, 0]) and len(point) == 3:
            self._point[Axis.Z] = point[0]
            self._point[Axis.X] = point[1]
            self._point[Axis.Y] = point[2]
        elif type(point) == type(np.array([0, 0, 0])) and len(point) == 3:
            self._point[Axis.Z] = point[0]
            self._point[Axis.X] = point[1]
            self._point[Axis.Y] = point[2]
        elif (
            isinstance(point, (int, np.integer))
            and isinstance(y, (int, np.integer))
            and isinstance(z, (int, np.integer))
        ):
            self._point[Axis.X] = int(point)
            self._point[Axis.Y] = int(y)
            self._point[Axis.Z] = int(z)
        else:
            raise Exception("Type incorrect")

    # def __repr__(self):
    #     return "(x: {}, y: {}, z:{})".format(self.x, self.y, self.z)

    def __repr__(self):
        return "ThreeDimensionalPoint({},{},{})".format(self.x, self.y, self.z)

    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y and self.z == __o.z

    def __hash__(self) -> int:
        return hash("(x:{}, y:{}, z:{})".format(self.x, self.y, self.z))

    def move(self, axis: Axis | AxisDirection, number=1):
        if type(axis) == list:
            return [self.move(a) for a in axis]
        point = deepcopy(self)
        if type(axis) == Axis:
            point._point[axis] += number
        elif type(axis) == AxisDirection:
            point._point[axis.axis] += number * axis.direction
        else:
            raise Exception("Type incorrect")
        return point

    def twoPointInAxis(self, axis: Axis | AxisDirection, number=1):
        return [self.move(axis, number=number), self.move(axis, number=number * (-1))]

    def threePointsInAxis(self, axis: Axis | AxisDirection, number=1):
        return self.twoPointInAxis(axis, number) + [self]

    def eightPointsInTwoAxis(
        self, axis1: Axis | AxisDirection, axis2: Axis | AxisDirection
    ):
        aux = self.twoPointInAxis(axis1)
        aux2 = aux + [self]
        aux += sum([p.twoPointInAxis(axis2) for p in aux2], [])
        return aux

    def ninePointsInTwoAxis(
        self, axis1: Axis | AxisDirection, axis2: Axis | AxisDirection
    ):
        aux: list[ThreeDimensionalPoint] = [self] + self.eightPointsInTwoAxis(
            axis1, axis2
        )
        return aux

    def sixPointsInAxisAndDirection(self, axis: Axis, direction: AxisDirection):
        aux = self.threePointsInAxis(axis)
        return aux + [p.move(direction) for p in aux]

    def moves(self, axes: list[Axis | AxisDirection]):
        result = [deepcopy(self)]
        for axis in axes:
            result.append(result[-1].move(axis))
        return result

    @property
    def x(self):
        return self._point[Axis.X]

    @property
    def y(self):
        return self._point[Axis.Y]

    @property
    def z(self):
        return self._point[Axis.Z]

    def pointConnected(self):
        return [self.move(axis) for axis in list(AxisDirection)]


def directionFromPoint(
    fromPoint: ThreeDimensionalPoint, toPoint: ThreeDimensionalPoint
) -> AxisDirection:
    if (
        fromPoint.x == toPoint.x
        and fromPoint.y == toPoint.y
        and fromPoint.z == toPoint.z
    ):
        raise Exception("Are the same point")
    if fromPoint.x == toPoint.x and fromPoint.y == toPoint.y:
        if fromPoint.z < toPoint.z:
            return AxisDirection.ZU
        else:
            return AxisDirection.ZD
    if fromPoint.x == toPoint.x and fromPoint.z == toPoint.z:
        if fromPoint.y < toPoint.y:
            return AxisDirection.YU
        else:
            return AxisDirection.YD
    if fromPoint.y == toPoint.y and fromPoint.z == toPoint.z:
        if fromPoint.x < toPoint.x:
            return AxisDirection.XU
        else:
            return AxisDirection.XD
    raise Exception("This points aren't in the same axis")


def distance(
    p: ThreeDimensionalPoint, q: ThreeDimensionalPoint | list[ThreeDimensionalPoint]
):
    if type(q) == list:
        return min([distance(p, r) for r in q])
    elif type(q) == ThreeDimensionalPoint:
        return abs(p.x - q.x) + abs(p.y - q.y) + abs(p.z - q.z)
    else:
        raise Exception("Invalid type of q")


def meanOfPoints(points: list[ThreeDimensionalPoint]):
    sumX, sumY, sumZ = 0, 0, 0
    for point in points:
        sumX += point.x
        sumY += point.y
        sumZ += point.z
    n = len(points)
    return ThreeDimensionalPoint(sumX // n, sumY // n, sumZ // n)


numberOfPointForNearby = 10


def nearbyPointsOf(point: ThreeDimensionalPoint, inPoints: list[ThreeDimensionalPoint]):
    inPoints.sort(key=lambda p: distance(p, point))
    return inPoints[:numberOfPointForNearby]


class FlooredMatrix:
    def __init__(self, baseMatrix=np.zeros((1, 0, 0), dtype=bool)):
        self.baseMatrix = baseMatrix
        self.x_0 = 0
        self.y_0 = 0
        self.z_0 = 0

    def tupleOfPoint(self, point: ThreeDimensionalPoint):
        return (point.z + self.z_0, point.x + self.x_0, point.y + self.y_0)

    def __getitem__(self, key: ThreeDimensionalPoint | int):
        if type(key) == int:
            return self.baseMatrix[key]
        else:
            return self.baseMatrix[self.tupleOfPoint(key)]

    def checkPoint(self, point: ThreeDimensionalPoint):
        t = self.tupleOfPoint(point)
        if t[1] < 0:
            return AxisDirection.XD
        if t[2] < 0:
            return AxisDirection.YD
        if t[0] < 0:
            return AxisDirection.ZD
        if t[1] >= self.baseMatrix.shape[1]:
            return AxisDirection.XU
        if t[2] >= self.baseMatrix.shape[2]:
            return AxisDirection.YU
        if t[0] >= self.baseMatrix.shape[0]:
            return AxisDirection.ZU
        return None

    def addLayerToPoint(self, point: ThreeDimensionalPoint):
        while True:
            axis = self.checkPoint(point)
            if axis == None:
                break
            self.addLayer(axis)

    def __setitem__(
        self, key: ThreeDimensionalPoint | list[ThreeDimensionalPoint], value
    ):
        if type(key) == ThreeDimensionalPoint:
            self.addLayerToPoint(key)
            self.baseMatrix[self.tupleOfPoint(key)] = value
        elif type(key) == list[ThreeDimensionalPoint]:
            for p in key:
                self[p] = value
        else:
            raise Exception("Incorrect type")

    def isOne(self, point: ThreeDimensionalPoint):
        return exists3D(self.tupleOfPoint(point), self.baseMatrix) and self[point] == 1

    def isZero(self, point: ThreeDimensionalPoint):
        return not exists3D(self.tupleOfPoint(point), self.baseMatrix) or (
            exists3D(self.tupleOfPoint(point), self.baseMatrix) and self[point] == 0
        )

    def addLayer(self, axis: AxisDirection):
        aheadOrBehind = axis.direction == 1
        if axis.axis == Axis.Z:
            if aheadOrBehind:
                self.addUpFloor()
            else:
                self.addDownFloor()
                self.z_0 += 1
        elif axis.axis == Axis.X:
            self.baseMatrix = addLayerToMatrix(self.baseMatrix, 1, aheadOrBehind)
            if not aheadOrBehind:
                self.x_0 += 1
        else:
            self.baseMatrix = addLayerToMatrix(self.baseMatrix, 2, aheadOrBehind)
            if not aheadOrBehind:
                self.y_0 += 1

    def addUpFloor(self):
        zerosFloor = np.zeros_like(self.baseMatrix[0])
        self.baseMatrix = np.insert(
            self.baseMatrix, self.baseMatrix.shape[0], zerosFloor, 0
        )

    def addDownFloor(self):
        zerosFloor = np.zeros_like(self.baseMatrix[0])
        self.baseMatrix = np.insert(self.baseMatrix, 0, zerosFloor, 0)

    def pointsOrdered(self):
        t = np.argwhere(self.baseMatrix == 1)[0]
        t[1] -= self.x_0
        t[2] -= self.y_0
        t[0] -= self.z_0
        previosPoint = ThreeDimensionalPoint(t)
        points = [previosPoint]
        continueWhile = True
        while continueWhile:
            continueWhile = False
            for point in previosPoint.pointConnected():
                if self.isOne(point) and not point in points:
                    previosPoint = point
                    points.append(point)
                    continueWhile = True
                    break
        return points

    def plot(self, pointsAuxiliar=[], elev=None, azim=None):
        x = []
        y = []
        z = []
        points = self.pointsOrdered()
        points.append(points[0])
        for point in points:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
        fig = plt.figure(figsize=(4, 4))
        ax = Axes3D(fig)
        # ax = fig.add_subplot(111,projection='3d')
        # ax.view_init(elev=elev, azim=azim)
        ax.plot(x, y, z)
        x = []
        y = []
        z = []
        for point in pointsAuxiliar:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
        ax.plot(x, y, z, "r")

    @staticmethod
    def fromKnot(knot: CustomKnot):
        pd = knot.planarDiagrams()
        pd = borderByZeros(pd, False)
        pd = borderByZeros(pd, False)
        pd = borderByZeros(pd, False)
        pd = borderByZeros(pd, False)
        pd = borderByZeros(pd, False)
        pd = createGaps(pd)
        pd01 = deepcopy(pd)
        pd01 = replace(pd01, list(range(1, knot.numberOfStrands + 1)), 1)
        pd01 = replace(pd01, [-1, -2], 0)
        pd01 = np.array([pd01])
        fM = FlooredMatrix(pd01)
        fM.addUpFloor()
        fM.addUpFloor()
        for cross in knot.crosses:
            index = findCrossInPd(cross, pd)
            # La representación del n de en medio sera -1 si pasa por arriba de izquierda a derecha y
            #                                         -2 si pasa por arriba de derecha a izquierda
            if pd[index] == -1:
                fM[0][index] = 1
                fM[0][left(index)] = 0
                fM[0][right(index)] = 0
                fM[1][left(left(index))] = 1
                fM[1][right(right(index))] = 1
                fM[2][left(left(index))] = 1
                fM[2][right(right(index))] = 1
                fM[2][left(index)] = 1
                fM[2][right(index)] = 1
                fM[2][index] = 1

            elif pd[index] == -2:
                fM[0][index] = 1
                fM[0][up(index)] = 0
                fM[0][down(index)] = 0
                fM[1][up(up(index))] = 1
                fM[1][down(down(index))] = 1
                fM[2][up(up(index))] = 1
                fM[2][down(down(index))] = 1
                fM[2][up(index)] = 1
                fM[2][down(index)] = 1
                fM[2][index] = 1
            else:
                raise Exception("El index no es de un cruce")
        fM.addUpFloor()
        fM.addUpFloor()
        fM.addDownFloor()
        fM.addDownFloor()

        return fM


class ThreeDimensionalTransformType(Enum):
    inLine = 0
    allInLine = 1
    inCorner = 2
    allInLineDoubleDirection = 3
    allInLineDoubleDirectionLoop = 4
    switchSide = 5
    slideCorner = 6
    doubleAllInLine = 7
    inCornerForSide = 8
    allInAxis = 9


class ThreeDimensionalTransform:
    def __init__(
        self,
        point: ThreeDimensionalPoint,
        type: ThreeDimensionalTransformType,
        variety: bool | None = None,
        direction: AxisDirection | None = None,
    ):
        self.point = point
        self.type = type
        self.direction = direction
        self.variety = variety

    def __str__(self) -> str:
        return "ThreeDimensionalTransform({},{},{},{})".format(
            self.point, self.type, self.variety, self.direction
        )

    def __repr__(self) -> str:
        return "ThreeDimensionalTransform({},{},{},{})".format(
            self.point, self.type, self.variety, self.direction
        )

    @staticmethod
    def allPosibilitesForReduceLengthIn(point: ThreeDimensionalPoint):
        posibilites = [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInLine, False
            )
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInAxis, None, d
            )
            for d in list(AxisDirection)
        ]
        return posibilites

    @staticmethod
    def allPosibilitesIn(point):
        posibilites = [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInLineDoubleDirectionLoop, True
            ),
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInLineDoubleDirectionLoop, False
            ),
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInAxis, None, d
            )
            for d in list(AxisDirection)
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInLineDoubleDirection, True
            ),
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInLineDoubleDirection, False
            ),
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.inLine, True, d
            )
            for d in list(AxisDirection)
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.doubleAllInLine, True, d
            )
            for d in list(AxisDirection)
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInLine, True, d
            )
            for d in list(AxisDirection)
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.switchSide, v, d
            )
            for d in list(AxisDirection)
            for v in [False, True]
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.inCornerForSide, v, d
            )
            for d in list(AxisDirection)
            for v in [False, True]
        ]
        posibilites += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.inLine, False
            ),
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInLine, False
            ),
            ThreeDimensionalTransform(point, ThreeDimensionalTransformType.inCorner),
            ThreeDimensionalTransform(point, ThreeDimensionalTransformType.slideCorner),
        ]

        return posibilites

    @staticmethod
    def allPosibilitesNearbyOf(
        point: ThreeDimensionalPoint, inPoints: list[ThreeDimensionalPoint]
    ):
        pointsNearby = nearbyPointsOf(point, inPoints)
        return sum(
            [ThreeDimensionalTransform.allPosibilitesIn(p) for p in pointsNearby], []
        )


def move(
    point: ThreeDimensionalPoint | list[ThreeDimensionalPoint],
    axis: Axis | AxisDirection,
    number=1,
):
    if type(point) == ThreeDimensionalPoint:
        return point.move(axis, number)
    elif type(point) == list:
        return [p.move(axis, number) for p in point]
    else:
        raise Exception("Invalid type")


class ThreeDimensionalTarget:
    def __init__(
        self, point: ThreeDimensionalPoint, isFirst: bool = False, isLast: bool = False
    ):
        self.point = point
        self.isFirst = isFirst
        self.isLast = isLast


class FlooredKnot:
    def __init__(self, knot: CustomKnot):
        self.fM = FlooredMatrix.fromKnot(knot)

    def __setitem__(
        self, key: ThreeDimensionalPoint | list[ThreeDimensionalPoint], value
    ):
        if type(key) == ThreeDimensionalPoint:
            self.fM[key] = value
        elif type(key) == list:
            for p in key:
                self[p] = value
        else:
            raise Exception("Incorrect type")

    def __eq__(self, __o) -> bool:
        if type(__o) != FlooredKnot:
            return False
        other: FlooredKnot = __o
        # selfPoints = set(self.fM.pointsOrdered())
        # otherPoints = set(other.fM.pointsOrdered())
        # return selfPoints == otherPoints
        if self.fM.baseMatrix.shape != other.fM.baseMatrix.shape:
            return False

        if type(self.fM.baseMatrix == other.fM.baseMatrix) == bool:
            print(self.fM.baseMatrix)
            print(self.fM.baseMatrix.shape)
            print(other.fM.baseMatrix)
            print(other.fM.baseMatrix.shape)

        return (self.fM.baseMatrix == other.fM.baseMatrix).all()

    def plot(self, pointsAuxiliar=[], elev=None, azim=None):
        self.fM.plot(pointsAuxiliar=pointsAuxiliar, elev=elev, azim=azim)

    def addLayer(self, axis: Axis, direction: bool):
        self.fM.addLayer(axis.axisDirection(direction))

    def isValue(
        self,
        value,
        point: ThreeDimensionalPoint | list[ThreeDimensionalPoint],
        axisDirection: AxisDirection = None,
        number=1,
    ):
        if type(axisDirection) == list:
            return all(
                [
                    self.isValue(value, point, axisDirection=axis, number=number)
                    for axis in axisDirection
                ]
            )
        if type(point) == list:
            return all(
                [
                    self.isValue(value, p, axisDirection=axisDirection, number=number)
                    for p in point
                ]
            )
        if axisDirection == None:
            if value == 0:
                return self.fM.isZero(point)
            else:
                return self.fM.isOne(point)
        else:
            return self.isValue(value, point.move(axisDirection, number=number))

    def isOne(
        self,
        point: ThreeDimensionalPoint | list[ThreeDimensionalPoint],
        axisDirection: AxisDirection = None,
        number=1,
    ):
        return self.isValue(1, point, axisDirection, number)

    def isZero(
        self,
        point: ThreeDimensionalPoint | list[ThreeDimensionalPoint],
        axisDirection: AxisDirection = None,
        number=1,
    ):
        return self.isValue(0, point, axisDirection, number)

    def typeThreeInLine(self, point: ThreeDimensionalPoint):
        if self.isZero(point):
            return None

        for axis in list(Axis):
            if self.isOne(point.twoPointInAxis(axis)):
                return axis

        return None

    def allPointsInLine(self, point: ThreeDimensionalPoint, axis: Axis):
        points = [point]
        auxiliarPoint = point.move(axis)
        while True:
            if self.isOne(auxiliarPoint):
                points.append(auxiliarPoint)
                auxiliarPoint = auxiliarPoint.move(axis)
            else:
                break

        auxiliarPoint = point.move(axis, -1)

        while True:
            if self.isOne(auxiliarPoint):
                points.insert(0, auxiliarPoint)
                auxiliarPoint = auxiliarPoint.move(axis, -1)
            else:
                break
        return points

    # MARK: ThreeInLine:

    def auxiliarPointsForThreeInLine(
        self, point: ThreeDimensionalPoint, direction: AxisDirection
    ) -> tuple[
        list[ThreeDimensionalPoint],
        list[ThreeDimensionalPoint],
        list[ThreeDimensionalPoint],
    ]:
        if type(point) == list:
            z = zip(*[self.auxiliarPointsForThreeInLine(p, direction) for p in point])
            return tuple([list(set(sum(f, []))) for f in [list(t) for t in list(z)]])

        axis = self.typeThreeInLine(point)
        if type(axis) == type(None):
            return None
        # Quitamos los casos que no deberían ocurrir ej "x" 0, "x" 1, "y", 2
        if axis.value == direction.value // 2:
            # print("esto no puede ocurrir")
            return None
        # La primera Axis me la determina direction//2, la segunda será la que no sea ni esa ni axis
        firstAxis = direction.axis
        firstDirection = direction.direction
        secondAxis = Axis.thatAreNot([firstAxis, axis])
        points = [
            point.move(firstAxis, firstDirection),
            point.move(firstAxis, firstDirection * 2),
        ]

        upPoints = [point.move(secondAxis, 1) for point in points]
        dwPoints = [point.move(secondAxis, -1) for point in points]
        points = points + upPoints + dwPoints

        upPoints = [point.move(axis, 1) for point in points]
        upUpPoints = [point.move(axis, 2) for point in points]
        dwPoints = [point.move(axis, -1) for point in points]
        dwDwPoints = [point.move(axis, -2) for point in points]
        freePoints = points + upPoints + upUpPoints + dwPoints + dwDwPoints

        toCompletePointCenter = point.move(firstAxis, firstDirection)
        toCompletePointCenterUp = toCompletePointCenter.move(axis, 1)
        toCompletePointCenterDw = toCompletePointCenter.move(axis, -1)

        toCompletePoints = [
            toCompletePointCenter,
            toCompletePointCenterUp,
            toCompletePointCenterDw,
        ]
        pointsToDelete = [point]
        # self.plot(freePoints)
        return freePoints, toCompletePoints, pointsToDelete

    def auxiliarPointsForUndoThreeInLine(self, point: ThreeDimensionalPoint):
        axis = self.typeThreeInLine(point)
        if type(axis) == type(None):
            return None
        availableAxis = Axis.thatAreNot(axis)
        availableDirections = sum(
            [[axis.axisDirection(d) for d in [-1, 1]] for axis in availableAxis], []
        )
        for direction in availableDirections:
            desiredPoint = point.move(direction)
            if self.isOne(desiredPoint.twoPointInAxis(axis)):
                freePoint = desiredPoint.move(direction)
                otherAxis = Axis.thatAreNot([axis, direction.axis])
                pointsNeedFree = [freePoint] + freePoint.twoPointInAxis(otherAxis)
                pointsToDelete = [point] + point.twoPointInAxis(axis)
                pointsToComplete = [desiredPoint]
                return pointsNeedFree, pointsToComplete, pointsToDelete
        return None

    # MARK: AllInLine

    def auxiliarPointsForAllInLineTransform(
        self, point: ThreeDimensionalPoint, direction: AxisDirection
    ):
        axis = self.typeThreeInLine(point)
        if type(axis) == type(None):
            return None
        # Quitamos los casos que no deberían ocurrir ej "x" 0, "x" 1, "y", 2
        if axis.value == direction.value // 2:
            # print("esto no puede ocurrir")
            return None
        allPoints = self.allPointsInLine(point, axis)
        if len(allPoints) < 3:
            return None
        if len(allPoints) == 3:
            return self.auxiliarPointsForThreeInLine(allPoints[1], direction)
        allPoints = allPoints[1:-1]
        return self.auxiliarPointsForThreeInLine(allPoints, direction)

    def auxiliarPointsForDoubleAllInLine(self, point, direction: AxisDirection):
        copy = deepcopy(self)
        auxPoints = copy.auxiliarPointsForAllInLineTransform(point, direction)
        if type(auxPoints) == type(None):
            return None
        firstPointsNeedFree, firstPointsToComplete, firstPointsToDelete = auxPoints
        isFirstTransformApply = copy.applyTransform(None, auxPoints)
        if not isFirstTransformApply:
            return None

        point = point.move(direction)
        auxPoints = copy.auxiliarPointsForAllInLineTransform(point, direction)
        if type(auxPoints) == type(None):
            return None

        (
            secondsPointsNeedFree,
            secondsPointsToComplete,
            secondsPointsToDelete,
        ) = auxPoints

        pointsNeedFree = list(set(firstPointsNeedFree + secondsPointsNeedFree))
        pointsToComplete = list(
            set(firstPointsToComplete)
            .union(set(secondsPointsToComplete))
            .difference(set(secondsPointsToDelete))
        )
        pointsToDelete = list(
            set(firstPointsToDelete)
            .union(set(secondsPointsToDelete))
            .difference((set(secondsPointsToComplete)))
        )

        return pointsNeedFree, pointsToComplete, pointsToDelete

    def auxiliarPointsForUndoAllInLineTransform(self, point: ThreeDimensionalPoint):
        axis = self.typeThreeInLine(point)
        if type(axis) == type(None):
            return None
        allPoints = self.allPointsInLine(point, axis)
        if len(allPoints) < 3:
            return None
        if len(allPoints) == 3:
            return self.auxiliarPointsForUndoThreeInLine(allPoints[1])

        availableAxis = Axis.thatAreNot(axis)
        availableDirections = sum(
            [[axis.axisDirection(d) for d in [-1, 1]] for axis in availableAxis], []
        )
        for direction in availableDirections:
            desiredPoints = [p.move(direction) for p in allPoints[1:-1]]
            externalsPoints = [
                allPoints[0].move(direction),
                allPoints[-1].move(direction),
            ]
            if self.isOne(externalsPoints):
                pointsNeedFree = [
                    desiredPoint.move(direction) for desiredPoint in desiredPoints
                ]
                otherAxis = Axis.thatAreNot([axis, direction.axis])
                pointsNeedFree = sum(
                    [p.twoPointInAxis(otherAxis) for p in pointsNeedFree],
                    pointsNeedFree,
                )
                pointsToComplete = desiredPoints
                pointsToDelete = allPoints

                return pointsNeedFree, pointsToComplete, pointsToDelete
        return None

    # MARK: AllInLineDoubleDirection

    def auxiliarPointsForAllInLineDoubleDirectionTransform(
        self,
        point: ThreeDimensionalPoint,
        variety: bool,
        previousDirection=None,
        withDirection=False,
    ):
        axis = self.typeThreeInLine(point)
        if type(axis) == type(None):
            # print("Not axis")
            return None
        allPoints = self.allPointsInLine(point, axis)
        if len(allPoints) < 2:
            # print("len low")
            return None
        external1 = allPoints[0]
        external2 = allPoints[-1]
        direction1: AxisDirection = None
        direction2: AxisDirection = None, None
        for d in sum(
            [[a.axisDirection(1), a.axisDirection(-1)] for a in Axis.thatAreNot(axis)],
            [],
        ):
            if self.isOne(external1.move(d)):
                direction1 = d
            if self.isOne(external2.move(d)):
                direction2 = d
        if direction1 == None or direction2 == None:
            print("point", point)
            print("variety", variety)
            print("previousDirection", previousDirection)
            self.plot()
            raise Exception("Caso a estudiar")
        if direction1 == direction2:
            # print("Same direction")
            return None

        external1, external2 = [external1, external2][variety], [external1, external2][
            not variety
        ]
        direction1, direction2 = [direction1, direction2][variety], [
            direction1,
            direction2,
        ][not variety]
        if type(previousDirection) != type(None) and direction1 != previousDirection:
            return None
        axis1, _ = axis.axisDirection(variety), axis.axisDirection(not variety)

        pointsNeedFree: list[ThreeDimensionalPoint] = external2.move(
            direction1
        ).ninePointsInTwoAxis(axis, Axis.thatAreNot([axis, direction1.axis]))
        pointsNeedFree += (
            external2.move(direction1)
            .move(direction1)
            .ninePointsInTwoAxis(axis, Axis.thatAreNot([axis, direction1.axis]))
        )

        freePoints1 = allPoints[:-1] if variety else allPoints[1:]
        freePoints1 = move(freePoints1, direction1)
        freePoints1 += sum(
            [
                p.twoPointInAxis(Axis.thatAreNot([axis, direction1.axis]))
                for p in freePoints1
            ],
            [],
        )
        freePoints1 += [p.move(direction1) for p in freePoints1]

        pointsNeedFree += freePoints1
        pointsToDelete = allPoints[:-1] if not variety else allPoints[1:]
        pointsToComplete = move(move(pointsToDelete, direction1), axis1)

        if withDirection:
            return pointsNeedFree, pointsToComplete, pointsToDelete, direction1
        else:
            return pointsNeedFree, pointsToComplete, pointsToDelete

    def auxiliarPointsForAllInLineDoubleDirectionLoopTransform(
        self, point: ThreeDimensionalPoint, variety: bool
    ):
        auxiliarPoints = self.auxiliarPointsForAllInLineDoubleDirectionTransform(
            point, variety, withDirection=True
        )
        if type(auxiliarPoints) == type(None):
            return None
        (
            auxPointsNeedFree,
            auxPointsToComplete,
            auxPointsToDelete,
            direction,
        ) = auxiliarPoints
        pointsNeedFree = set(auxPointsNeedFree)
        pointsToDelete = set(auxPointsToDelete)
        pointsToComplete = set(auxPointsToComplete)
        copy = deepcopy(self)
        while True:
            if not copy.isZero(auxPointsNeedFree):
                return (
                    list(pointsNeedFree),
                    list(pointsToComplete),
                    list(pointsToDelete),
                )
            pointsNeedFree.update(auxPointsNeedFree)
            pointsToComplete.update(auxPointsToComplete)
            pointsToDelete.update(auxPointsToDelete)

            copy[auxPointsToComplete] = 1
            copy[auxPointsToDelete] = 0

            point = point.move(direction)
            auxiliarPoints = copy.auxiliarPointsForAllInLineDoubleDirectionTransform(
                point, variety, withDirection=True
            )
            if type(auxiliarPoints) == type(None):
                return (
                    list(pointsNeedFree),
                    list(pointsToComplete),
                    list(pointsToDelete),
                )
            (
                auxPointsNeedFree,
                auxPointsToComplete,
                auxPointsToDelete,
                auxDirection,
            ) = auxiliarPoints

            if direction != auxDirection:
                return (
                    list(pointsNeedFree),
                    list(pointsToComplete),
                    list(pointsToDelete),
                )

    # MARK: ThreeInCorner

    def isThreeInCorner(
        self,
        point: ThreeDimensionalPoint,
        direction1: AxisDirection,
        direction2: AxisDirection,
    ):
        return self.isOne(point, [direction1, direction2]) and self.isZero(
            point, [direction1.opposite, direction2.opposite]
        )

    def directionsForThreeInCorner(self, point: ThreeDimensionalPoint):
        if self.isZero(point):
            return None
        axes = list(Axis)
        axesTuples = [
            (axes[i], axes[j])
            for i in range(len(axes))
            for j in range(len(axes))
            if i < j
        ]
        axesDirectionsTuples = [
            (axis1.axisDirection(d1), axis2.axisDirection(d2))
            for axis1, axis2 in axesTuples
            for d1 in [-1, 1]
            for d2 in [-1, 1]
        ]

        for aD1, aD2 in axesDirectionsTuples:
            if self.isThreeInCorner(point, aD1, aD2):
                return aD1, aD2

        return None

    def fourPoints(
        self,
        point: ThreeDimensionalPoint,
        direction1: AxisDirection,
        direction2: AxisDirection,
    ):
        return [
            point,
            point.move(direction1),
            point.move(direction2),
            point.move(direction2).move(direction1),
        ]

    def auxiliarPointsForThreeInCorner(self, point: ThreeDimensionalPoint):
        t = self.directionsForThreeInCorner(point)
        if type(t) == type(None):
            return None
        aD1, aD2 = t

        if aD1.axis == aD2.axis:
            raise Exception("Esto no debería ocurrir")

        desiredPoint = point.move(aD1).move(aD2)

        otherAxis = Axis.thatAreNot([aD1, aD2])
        upPoint = desiredPoint.move(otherAxis, 1)
        dwPoint = desiredPoint.move(otherAxis, -1)

        pointsToComplete = [desiredPoint]
        pointsToDelete = [point]
        pointsNeedFree = (
            self.fourPoints(desiredPoint, aD1, aD2)
            + self.fourPoints(upPoint, aD1, aD2)
            + self.fourPoints(dwPoint, aD1, aD2)
        )

        return pointsNeedFree, pointsToComplete, pointsToDelete

    def auxiliarPointsForSwitchSideOpposite(
        self, point: ThreeDimensionalPoint, aD1: AxisDirection, aD2: AxisDirection
    ):
        if self.isZero(point.move(aD1, 2)):
            return None

        otherAxis = Axis.thatAreNot([aD1, aD2])
        downFreePoint = point.move(aD2.opposite)
        downFreePoints = downFreePoint.threePointsInAxis(otherAxis)
        moreDownFreePoints = (
            sum([p.twoPointInAxis(aD1) for p in downFreePoints], [])
            + sum([p.twoPointInAxis(aD1, 2) for p in downFreePoints], [])
            + [p.move(aD1, 3) for p in downFreePoints]
        )
        downFreePoints += moreDownFreePoints

        moreDownFreePoints = [p.move(aD2.opposite) for p in downFreePoints] + [
            p.move(aD2.opposite, 2) for p in downFreePoints
        ]
        downFreePoints += moreDownFreePoints

        pointsNeedFree = point.move(aD1.opposite).threePointsInAxis(otherAxis)
        pointsNeedFree += [p.move(aD1.opposite) for p in pointsNeedFree]
        pointsNeedFree += [p.move(aD2) for p in pointsNeedFree]

        pointsNeedFree += downFreePoints

        pointsToDelete = [point.move(aD1)]

        pointsToComplete = [
            point.move(aD1.opposite),
            point.move(aD1.opposite).move(aD2.opposite),
            point.move(aD1.opposite).move(aD2.opposite, 2),
            point.move(aD2.opposite, 2),
            point.move(aD1).move(aD2.opposite, 2),
            point.move(aD1, 2).move(aD2.opposite, 2),
            point.move(aD1, 2).move(aD2.opposite, 1),
        ]
        return pointsNeedFree, pointsToComplete, pointsToDelete

    def auxiliarPointsForSwitchSideTransversal(
        self,
        point: ThreeDimensionalPoint,
        direction: AxisDirection,
        aD1: AxisDirection,
        aD2: AxisDirection,
    ):
        if self.isZero(point.move(aD1, 2)):
            return None

        basicPoints = [
            point.move(aD1.opposite).move(direction, 2),
            point.move(aD1, 2).move(direction, 2),
        ]
        basicPoints = sum(
            [p.ninePointsInTwoAxis(direction, aD1) for p in basicPoints], []
        )

        pointsNeedFree = sum([p.threePointsInAxis(aD2) for p in basicPoints], [])

        basicPoints = [point.move(aD1.opposite), point.move(aD1.opposite)] + point.move(
            direction.opposite
        ).move(aD1.opposite).threePointsInAxis(aD1)
        pointsNeedFree += sum([p.threePointsInAxis(aD2) for p in basicPoints], [])
        pointsNeedFree += [point.move(aD2.opposite)]

        pointsToComplete = [
            point.move(aD1, -1).move(direction, 0),
            point.move(aD1, -1).move(direction, 1),
            point.move(aD1, -1).move(direction, 2),
            point.move(aD1, 0).move(direction, 2),
            point.move(aD1, 1).move(direction, 2),
            point.move(aD1, 2).move(direction, 2),
            point.move(aD1, 2).move(direction, 1),
        ]
        pointsToDelete = [point.move(aD1)]

        return pointsNeedFree, pointsToComplete, pointsToDelete

    def auxiliarPointsForSwitchSide(
        self, point: ThreeDimensionalPoint, variety: bool, direction: AxisDirection
    ):
        t = self.directionsForThreeInCorner(point)
        if t == None:
            return None
        aD1, aD2 = t
        if variety:
            aD1, aD2 = aD2, aD1
        if direction.opposite == aD2:
            return self.auxiliarPointsForSwitchSideOpposite(point, aD1, aD2)
        if direction.axis == Axis.thatAreNot([aD1, aD2]):
            return self.auxiliarPointsForSwitchSideTransversal(
                point, direction, aD1, aD2
            )
        return None

    def auxiliarPointsForSlideCorner(self, point: ThreeDimensionalPoint):
        t = self.directionsForThreeInCorner(point)
        if type(t) == type(None):
            return None
        aD1, aD2 = t

        if aD1.axis == aD2.axis:
            raise Exception("Esto no debería ocurrir")

        allInLine1 = self.allPointsInLine(point.move(aD1), aD1.axis)
        allInLine2 = self.allPointsInLine(point.move(aD2), aD2.axis)
        if allInLine1[-1] == point:
            allInLine1.reverse()
        if allInLine2[-1] == point:
            allInLine2.reverse()
        outside1 = allInLine1[-1]
        outside2 = allInLine2[-1]
        t1 = self.directionsForThreeInCorner(outside1)
        t2 = self.directionsForThreeInCorner(outside2)

        if type(t1) == type(None) or type(t2) == type(None):
            raise Exception("Esto no debería ocurrir")

        oD11, oD12 = t1
        oD21, oD22 = t2

        oDs1 = [d for d in [oD11, oD12] if d.axis != aD1.axis]
        oDs2 = [d for d in [oD21, oD22] if d.axis != aD2.axis]

        if len(oDs1) != 1 or len(oDs2) != 1:
            raise Exception("Esto no debería ocurrir")

        oD1 = oDs1[0]
        oD2 = oDs2[0]
        if oD1 != oD2:
            return None
        outsideDirection = oD1

        otherAxis1 = aD2.axis
        aux1: list[ThreeDimensionalPoint] = move(allInLine1[:-1], outsideDirection)
        if len(allInLine2) > 2:
            freePoints1 = sum([p.threePointsInAxis(otherAxis1) for p in aux1], [])
        else:
            freePoints1 = aux1 + [p.move(aD2.opposite) for p in aux1]
        freePoints1 += [p.move(outsideDirection) for p in freePoints1]

        otherAxis2 = aD1.axis
        aux2: list[ThreeDimensionalPoint] = move(allInLine2[:-1], outsideDirection)
        if len(allInLine1) > 2:
            freePoints2 = sum([p.threePointsInAxis(otherAxis2) for p in aux2], [])
        else:
            freePoints2 = aux2 + [p.move(aD1.opposite) for p in aux2]
        freePoints2 += [p.move(outsideDirection) for p in freePoints2]

        cornerPointFree = (
            point.move(aD1.opposite).move(aD2.opposite).move(outsideDirection)
        )

        pointsNeedFree = (
            freePoints1
            + freePoints2
            + [cornerPointFree, cornerPointFree.move(outsideDirection)]
        )

        pointsToDelete = allInLine1 + allInLine2

        pointsToComplete = aux1 + aux2

        return pointsNeedFree, pointsToComplete, pointsToDelete

    def auxiliarPointsForInCornerForSide(
        self, point: ThreeDimensionalPoint, variety: bool, direction: AxisDirection
    ):
        t = self.directionsForThreeInCorner(point)
        if type(t) == type(None):
            return None
        aD1, aD2 = t

        if not variety:
            aD1, aD2 = aD2, aD1

        if self.isZero(point.move(aD1, 2)):
            return None

        auxFreePoints = point.move(direction.opposite, 2).ninePointsInTwoAxis(
            direction, aD2
        )

        freePoints1 = auxFreePoints + [p.move(aD1) for p in auxFreePoints]
        freePoints1 += [p.move(aD1, 2) for p in freePoints1]

        freePoints2 = [p.move(aD1.opposite) for p in auxFreePoints]
        freePoints2 += [p.move(direction, 3) for p in freePoints2]
        freePoints2 += [p.move(aD1.opposite) for p in freePoints2] + [
            p.move(aD1.opposite, 2) for p in freePoints2
        ]

        freePoints3 = point.move(direction, 2).threePointsInAxis(aD2)
        freePoints3 += [p.move(aD1) for p in freePoints3]

        pointsNeedFree = freePoints2 + freePoints3 + freePoints1

        pointsToComplete = (
            point.move(aD1, 2)
            .move(direction.opposite)
            .moves(
                [
                    direction.opposite,
                    aD1.opposite,
                    aD1.opposite,
                    aD1.opposite,
                    aD1.opposite,
                    direction,
                    direction,
                    direction,
                    aD1,
                    aD1,
                ]
            )
        )

        pointsToDelete = [point.move(aD1)]

        # self.plot(pointsNeedFree)
        # self.plot(pointsToComplete)
        # self.plot(pointsToDelete+ [ThreeDimensionalPoint(0,0,0)])

        return pointsNeedFree, pointsToComplete, pointsToDelete

    def allPointsInAxis(self, point: ThreeDimensionalPoint, axis: AxisDirection):
        valueAxis = point._point[axis]
        allPoints = self.fM.pointsOrdered()
        points = [point]
        baseIndex = allPoints.index(point)
        for i in range(1, len(allPoints)):
            p = allPoints[(baseIndex + i) % len(allPoints)]
            if p._point[axis] == valueAxis:
                points.append(p)
            else:
                break

        for i in range(1, len(allPoints)):
            p = allPoints[(baseIndex - i) % len(allPoints)]
            if p._point[axis] == valueAxis:
                points.insert(0, p)
            else:
                break
        return points

    def auxiliarPointsForAllInAxis(
        self, point: ThreeDimensionalPoint, direction: AxisDirection
    ):
        points = self.allPointsInAxis(point, direction.axis)
        if len(points) < 4:
            return None

        direction1 = directionFromPoint(points[1], points[0])
        otherDirection1 = Axis.thatAreNot([direction1, direction])
        if self.isOne(points[0].move(direction)):
            extreme1FreePoints = (
                points[1]
                .move(direction)
                .sixPointsInAxisAndDirection(otherDirection1, direction1.opposite)
            )
            extreme1FreePoints += [p.move(direction) for p in extreme1FreePoints]

            extreme1ToComplete = []
            extreme1ToDelete = [points[0]]
        elif self.isOne(points[0].move(direction.opposite)):
            extreme1FreePoints = (
                points[0]
                .move(direction)
                .sixPointsInAxisAndDirection(otherDirection1, direction1)
            )
            extreme1FreePoints += (
                points[1]
                .move(direction)
                .sixPointsInAxisAndDirection(otherDirection1, direction1.opposite)
            )
            extreme1FreePoints += [p.move(direction) for p in extreme1FreePoints]

            extreme1ToComplete = [points[0].move(direction)]
            extreme1ToDelete = []
        else:
            raise Exception("Esto no debería ocurrir")

        direction2 = directionFromPoint(points[-2], points[-1])
        otherDirection2 = Axis.thatAreNot([direction2, direction])
        if self.isOne(points[-1].move(direction)):
            extreme2FreePoints = (
                points[-2]
                .move(direction)
                .sixPointsInAxisAndDirection(otherDirection2, direction2.opposite)
            )
            extreme2FreePoints += [p.move(direction) for p in extreme2FreePoints]

            extreme2ToComplete = []
            extreme2ToDelete = [points[-1]]
        elif self.isOne(points[-1].move(direction.opposite)):
            extreme2FreePoints = (
                points[-1]
                .move(direction)
                .sixPointsInAxisAndDirection(otherDirection2, direction2)
            )
            extreme2FreePoints += (
                points[-2]
                .move(direction)
                .sixPointsInAxisAndDirection(otherDirection2, direction2.opposite)
            )
            extreme2FreePoints += [p.move(direction) for p in extreme2FreePoints]

            extreme2ToComplete = [points[-1].move(direction)]
            extreme2ToDelete = []
        else:
            raise Exception("Esto no debería ocurrir")

        normalPoints = points[2:-2]
        otherAxes = Axis.thatAreNot(direction.axis)
        normalFreePoints: list[ThreeDimensionalPoint] = sum(
            [
                p.move(direction).ninePointsInTwoAxis(otherAxes[0], otherAxes[1])
                for p in normalPoints
            ],
            [],
        )
        normalFreePoints += [p.move(direction) for p in normalFreePoints]

        pointsNeedFree = extreme1FreePoints + normalFreePoints + extreme2FreePoints

        pointsToComplete = (
            extreme1ToComplete
            + [p.move(direction) for p in points[1:-1]]
            + extreme2ToComplete
        )

        pointsToDelete = extreme1ToDelete + points[1:-1] + extreme2ToDelete

        return pointsNeedFree, pointsToComplete, pointsToDelete

    def auxiliarPointsForTransform(
        self, transform: ThreeDimensionalTransform
    ) -> tuple[
        list[ThreeDimensionalPoint],
        list[ThreeDimensionalPoint],
        list[ThreeDimensionalPoint],
    ]:
        if transform.type == ThreeDimensionalTransformType.inLine:
            if transform.variety:
                return self.auxiliarPointsForThreeInLine(
                    transform.point, transform.direction
                )
            else:
                return self.auxiliarPointsForUndoThreeInLine(transform.point)
        elif transform.type == ThreeDimensionalTransformType.allInLine:
            if transform.variety:
                return self.auxiliarPointsForAllInLineTransform(
                    transform.point, transform.direction
                )
            else:
                return self.auxiliarPointsForUndoAllInLineTransform(transform.point)
        elif transform.type == ThreeDimensionalTransformType.doubleAllInLine:
            return self.auxiliarPointsForDoubleAllInLine(
                transform.point, transform.direction
            )
        elif transform.type == ThreeDimensionalTransformType.inCorner:
            return self.auxiliarPointsForThreeInCorner(transform.point)
        elif transform.type == ThreeDimensionalTransformType.allInLineDoubleDirection:
            return self.auxiliarPointsForAllInLineDoubleDirectionTransform(
                transform.point, transform.variety
            )
        elif (
            transform.type == ThreeDimensionalTransformType.allInLineDoubleDirectionLoop
        ):
            return self.auxiliarPointsForAllInLineDoubleDirectionLoopTransform(
                transform.point, transform.variety
            )
        elif transform.type == ThreeDimensionalTransformType.switchSide:
            return self.auxiliarPointsForSwitchSide(
                transform.point, transform.variety, transform.direction
            )
        elif transform.type == ThreeDimensionalTransformType.slideCorner:
            return self.auxiliarPointsForSlideCorner(transform.point)
        elif transform.type == ThreeDimensionalTransformType.inCornerForSide:
            return self.auxiliarPointsForInCornerForSide(
                transform.point, transform.variety, transform.direction
            )
        elif transform.type == ThreeDimensionalTransformType.allInAxis:
            return self.auxiliarPointsForAllInAxis(transform.point, transform.direction)
        else:
            raise Exception("Invalid transform")

    def applyTransform(
        self, transform: ThreeDimensionalTransform, auxiliarPoints=None, debug=False
    ):
        if type(auxiliarPoints) == type(None):
            auxiliarPoints = self.auxiliarPointsForTransform(transform)

        if type(auxiliarPoints) == type(None):
            if debug:
                print("auxiliarPoints es None")
            return False

        pointsNeedFree, pointsToComplete, pointsToDelete = auxiliarPoints
        if debug:
            print("pointsNeedFree:", pointsNeedFree)
            self.plot(pointsNeedFree)
            print("pointsToComplete:", pointsToComplete)
            self.plot(pointsToComplete)
            print("pointsToDelete:", pointsToDelete)
            self.plot(pointsToDelete)
        if self.isZero(pointsNeedFree):
            self[pointsToComplete] = 1
            self[pointsToDelete] = 0
            return True

        return False

    def transformsThatCompletePoints(
        self, point: ThreeDimensionalPoint | list[ThreeDimensionalPoint], debug: False
    ):
        if type(point) == list:
            return sum(
                [self.transformsThatCompletePoints(p, debug=debug) for p in point], []
            )
        elif type(point) == ThreeDimensionalPoint:
            transforms = ThreeDimensionalTransform.allPosibilitesNearbyOf(
                point, self.fM.pointsOrdered()
            )
            auxTransforms = []
            if debug:
                print(
                    "                     transformsThatCompletePoints: estudiamos las transformaciones"
                )
            for transform in transforms:
                if debug:
                    print("                     transformación: {}".format(transform))
                auxiliarPoints = self.auxiliarPointsForTransform(transform)
                if type(auxiliarPoints) == type(None):
                    if debug:
                        print("                     auxiliarPoints es None")
                    continue
                _, pointsToComplete, _ = auxiliarPoints
                if point in pointsToComplete:
                    if debug:
                        print("                     la añadimos")
                    auxTransforms.append(transform)
                else:
                    if debug:
                        print(
                            "                     no completa el punto {}".format(point)
                        )
            return auxTransforms
        else:
            raise Exception("Incorrect Type")

    def _reducesALength(
        self, requirements: list[ThreeDimensionalPoint] = [], j=0, debug=False
    ):
        actualLength = len(self.fM.pointsOrdered())
        validPointsForThreeInCorner: set[ThreeDimensionalPoint] = set()
        validPointsForAllInLineUndo: set[ThreeDimensionalPoint] = set()
        pointsForAllInLineThatAreStudy: set[ThreeDimensionalPoint] = set()
        validPointsForAllInAxis: set[Tuple[ThreeDimensionalPoint, Axis]] = set()
        pointsForAllInAxisThatAreStudy: set[Tuple[ThreeDimensionalPoint, Axis]] = set()
        if debug > 0:
            print("Repasamos los puntos")
        for point in self.fM.pointsOrdered():
            if debug > 1:
                print("Punto {}".format(point))
            if type(self.directionsForThreeInCorner(point)) != type(None):
                validPointsForThreeInCorner.add(point)
                if debug > 1:
                    print("Intentamos añadir en slideCorner")

            if not point in pointsForAllInLineThatAreStudy:
                axis = self.typeThreeInLine(point)
                if axis != None:
                    allPoints = self.allPointsInLine(point, axis)
                    if len(allPoints) > 2:
                        validPointsForAllInLineUndo.add(allPoints[1])
                        pointsForAllInLineThatAreStudy.update(allPoints[1:-1])
                        if debug > 1:
                            print("Intentamos añadir en allInLine")
            else:
                if debug > 1:
                    print("Este ya está estudiado")

            for a in list(Axis):
                if not (point, a) in pointsForAllInAxisThatAreStudy:
                    allPoints = self.allPointsInAxis(point, a)
                    allPointsWithAxis = [(p, a) for p in allPoints]
                    if len(allPoints) > 3:
                        validPointsForAllInAxis.add((allPoints[1], a))
                        pointsForAllInAxisThatAreStudy.update(allPointsWithAxis)
                        if debug > 1:
                            print("Intentamos añadir en allInAxis")
            else:
                if debug > 1:
                    print("Este ya está estudiado")

        possibleTransforms = [
            ThreeDimensionalTransform(point, ThreeDimensionalTransformType.slideCorner)
            for point in validPointsForThreeInCorner
        ]
        possibleTransforms += [
            ThreeDimensionalTransform(
                point, ThreeDimensionalTransformType.allInLine, variety=False
            )
            for point in validPointsForAllInLineUndo
        ]
        possibleTransforms += [
            ThreeDimensionalTransform(
                point,
                ThreeDimensionalTransformType.allInAxis,
                variety=None,
                direction=d,
            )
            for (point, axis) in validPointsForAllInAxis
            for d in [axis.axisDirection(1), axis.axisDirection(0)]
        ]

        if debug > 0:
            print(
                "Repasados los puntos, hay {} posibilidades".format(
                    len(possibleTransforms)
                )
            )
        for transform in possibleTransforms:
            if debug > 1:
                print("Intentando con transformación: {}".format(transform))
            auxiliarPoints = self.auxiliarPointsForTransform(transform)
            if type(auxiliarPoints) == type(None):
                if debug > 1:
                    print("Es none")
                continue
            _, _, pointsToDelete = auxiliarPoints
            if any([p in requirements for p in pointsToDelete]):
                if debug > 1:
                    print("No cumple requirements")
                continue
            copy = deepcopy(self)
            copy.applyTransform(transform, auxiliarPoints)
            newLength = len(copy.fM.pointsOrdered())
            if newLength < actualLength:
                if debug > 0:
                    print("Mejorado", newLength)
                self.applyTransform(transform, auxiliarPoints)
                return transform
        return None

    def reducesLength(
        self, requirements: list[ThreeDimensionalPoint] = [], debug=False
    ):
        j = 0
        transforms: list[ThreeDimensionalTransform] = []
        while True:
            transform = self._reducesALength(requirements, j, debug=debug)
            if transform == None:
                break
            transforms.append(transform)
            if debug:
                print(
                    "Estamos en el ciclo {}, y tiene longitud {}".format(
                        j, len(self.fM.pointsOrdered())
                    )
                )
                self.plot()
                # plt.pause(0.005)
            j += 1
        return transforms

    @property
    def targets(self):
        _targets = [ThreeDimensionalTarget(point) for point in self.fM.pointsOrdered()]
        _targets[0].isFirst = True
        _targets[-1].isLast = True
        return _targets


def equalizeShapes(fK1: FlooredKnot, fK2: FlooredKnot):
    direction = True
    c = 0
    while fK1.fM.baseMatrix.shape != fK2.fM.baseMatrix.shape and c < 100:
        c += 1
        for i in range(3):
            axis = Axis.Z if i == 0 else Axis.Y if i == 1 else Axis.X
            if fK1.fM.baseMatrix.shape[i] < fK2.fM.baseMatrix.shape[i]:
                fK1.addLayer(axis, direction)
                break
            elif fK1.fM.baseMatrix.shape[i] > fK2.fM.baseMatrix.shape[i]:
                fK2.addLayer(axis, direction)
                break
        direction = not direction
    return fK1, fK2
