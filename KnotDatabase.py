import pandas as pd
from CustomKnot import *
from KnotStar import *
from IPython.display import clear_output
from copy import copy, deepcopy
from time import time

dict_knot_numbers = {
    0: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 3,
    7: 7,
    8: 21,
    9: 49,
    10: 165,
}


def codeKnotToNumber(s: str) -> int:
    numberOfCross, number = int(s.split("_")[0]), int(s.split("_")[1])
    previous = 0
    i = 0
    while i < numberOfCross:
        if i in dict_knot_numbers.keys():
            previous += dict_knot_numbers[i]
        i += 1
    previous += number
    return previous - 1


def numberToCodeKnot(n: int) -> str:
    n += 1
    keys = list(dict_knot_numbers.keys())
    for key in keys:
        if n <= dict_knot_numbers[key]:
            return "{}_{}".format(key, n)
        n -= dict_knot_numbers[key]
    return ""


def knotNames(n: int) -> Dict[int, List[str]]:
    auxDict: Dict[int, List[str]] = {
        numberCrosses: []
        for numberCrosses in dict_knot_numbers.keys()
        if numberCrosses <= n
    }
    for numberCrosses in auxDict.keys():
        for number in range(dict_knot_numbers[numberCrosses]):
            name = str(numberCrosses) + "_" + str(number + 1)
            auxDict[numberCrosses].append(name)
    return auxDict


def knotNamesList(n: int) -> List[str]:
    aux: List[str] = []
    names = knotNames(n)
    for singleList in names.values():
        for name in singleList:
            aux.append(name)
    return aux


def singleDatabase(maxCrosses: int):
    auxDict = {"name": [], "crosses": []}
    names = knotNames(maxCrosses)
    for numberCrosses in names.keys():
        for name in names[numberCrosses]:
            auxDict["name"].append(name)
            knot = knotFromPyknotid(name)
            auxDict["crosses"].append(knot.crosses)

    return pd.DataFrame(auxDict)


def knotDatabase(
    maxCrosses: int,
    numberOfKnots: int,
    numberOfRandomMov: int = 100,
    maxStrands: int = None,
    debug: bool = False,
    initDB=None,
):
    names = knotNamesList(maxCrosses)
    df = singleDatabase(maxCrosses)
    df = df.reset_index()  # make sure indexes pair with number of rows
    auxDict = {"name": [], "crosses": [], "numberOfStrands": []}
    knotSetNames: Dict[str, simpleSet[CustomKnot]] = dict()
    if type(initDB) != type(None):
        for index, row in initDB.iterrows():
            name = row["name"]
            if not name in names:
                continue
            if not name in knotSetNames.keys():
                knotSetNames[name] = simpleSet()
            if name in list(initDB.name):
                crosses = row["crosses"]
                knot = CustomKnot(crosses)
                if knotSetNames[name].add(knot):
                    auxDict["name"].append(deepcopy(name))
                    auxDict["crosses"].append(deepcopy(crosses))
                    auxDict["numberOfStrands"].append(knot.numberOfStrands)

    for index, row in df.iterrows():
        name = row["name"]
        if not name in knotSetNames.keys():
            knotSetNames[name] = simpleSet()
        knotSet: simpleSet[CustomKnot] = knotSetNames[name]
        i = len([name_dict for name_dict in auxDict["name"] if name_dict == name])
        while i < numberOfKnots:
            # for i in range(numberOfKnots):
            # clear_output(wait=True)
            if debug > 0:
                print("name", name, len(knotSet))
            if debug > 0:
                print(
                    "copy",
                    i,
                    ", percentage: {:3.2f}".format(((i) / numberOfKnots) * 100),
                )
            if len([n for n in auxDict["name"] if n == name]) == 0:
                knot = CustomKnot(deepcopy(row["crosses"]))
            else:
                indexesPosible = []
                for j in range(len(auxDict["name"])):
                    if auxDict["name"][j] == name:
                        indexesPosible.append(j)
                j = choice(indexesPosible)
                knot = CustomKnot(deepcopy(auxDict["crosses"][j]))
            knot.randomMovN(
                numberOfRandomMov, maxStrands, percentage=True, debug=debug - 1
            )
            # Quitar cruces innecesarios y comprobar que no este en la base de datos.
            knot.reduceUnnecessaryMov()
            if knotSet.add(knot):
                auxDict["name"].append(deepcopy(name))
                auxDict["crosses"].append(knot.crosses)
                auxDict["numberOfStrands"].append(knot.numberOfStrands)
                i = len(
                    [name_dict for name_dict in auxDict["name"] if name_dict == name]
                )

    return pd.DataFrame(auxDict)


def combineDatabase(dbs: List[pd.DataFrame]):
    import pandas as pandas

    knotsDict: dict[str, dict[str, CustomKnot]] = dict()
    dbIndex = 0
    print("Leemos las bases de datos:")
    numberOfDbs = len(dbs)
    for db in dbs:
        dbIndex += 1
        rowIndex = 0
        numberOfRow = len(db.index)
        for _, row in db.iterrows():
            rowIndex += 1
            name = row["name"]
            print(
                "DataBase {}/{}, row {}/{}, {}           ".format(
                    dbIndex, numberOfDbs, rowIndex, numberOfRow, name
                ),
                end="\r",
            )
            crosses = row["crosses"]
            if "pd" in db.columns and row.pd != None:
                pd = eval(row["pd"])
            else:
                pd = None
            if not name in knotsDict.keys():
                knotsDict[name] = dict()
            currentKnotDict = knotsDict[name]
            knot = CustomKnot(crosses)
            knot.pd = pd
            representation = knot.representationForHash
            if not representation in currentKnotDict.keys():
                currentKnotDict[representation] = knot
            elif type(currentKnotDict[representation].pd) == type(None):
                currentKnotDict[representation].pd = pd
    print()
    print("Terminada la lectura, creamos ahora la nueva base de datos")
    auxDict = {"name": [], "crosses": [], "numberOfStrands": [], "pd": []}
    for name in knotsDict.keys():
        reprIndex = 0
        for representation in knotsDict[name].keys():
            reprIndex += 1
            print("{} {}          ".format(name, reprIndex), end="\r")
            knot = knotsDict[name][representation]
            auxDict["name"].append(deepcopy(name))
            auxDict["crosses"].append(deepcopy(knot.crosses))
            auxDict["numberOfStrands"].append(knot.numberOfStrands)
            if type(knot.pd) != type(None):
                auxDict["pd"].append("np." + repr(knot.pd))
            else:
                auxDict["pd"].append("None")
    print()
    print("Terminada la creación de la nueva base de datos")

    return pandas.DataFrame(auxDict)


def createPDsForDatabase(db: pd.DataFrame):
    if not "pd" in db.columns:
        db = db.assign(pd=None)
    numberOfRow = len(db.index)
    auxDict = {"name": [], "crosses": [], "numberOfStrands": [], "pd": []}
    start = time()
    for index, row in db.iterrows():
        # if index >3: break
        name = row["name"]
        crosses = row["crosses"]
        knot = CustomKnot(crosses)
        if row.pd != None:
            knot.pd = eval(row["pd"])
        auxDict["name"].append(deepcopy(name))
        auxDict["crosses"].append(deepcopy(crosses))
        auxDict["numberOfStrands"].append(knot.numberOfStrands)
        auxDict["pd"].append("np." + repr(knot.planarDiagrams()))
        percentage = 100 * index / numberOfRow
        end = time()
        remainingTime = 100 * (end - start) / percentage if percentage > 0 else 0
        print(
            "percentage: {:3.2f},time remaining:{}          ".format(
                percentage, remainingTimeString(remainingTime - (end - start))
            ),
            end="\r",
        )
    return pd.DataFrame(auxDict)


def createSimplePDsForDatabase(path="databases/masterPD.csv"):
    db = pd.read_csv(path)
    # db = db.assign(sPD=None)
    numberOfRows = len(db.index)
    indexRow = 0
    sPDs = []
    for index, row in db.iterrows():
        indexRow += 1
        name = row["name"]
        print("{} {}/{}     ".format(name, indexRow, numberOfRows), end="\r")
        knot = CustomKnot(row["crosses"])
        if "pd" in db.columns and row.pd != None:
            knot.pd = eval(row["pd"])
        sPDs.append("np." + repr(knot.simplePlanarDiagram()))

    db["sPD"] = sPDs
    db = db.drop("pd", axis=1)

    return db


def readMasterDatabase(
    path="databases/master.csv", maxRepresentations: int = 500, knots: list[str] = None
):
    masterDb = pd.read_csv(path)
    array: List[Tuple[str, CustomKnot]] = []
    for index, row in masterDb.iterrows():
        name = row["name"]
        if knots != None and not name in knots:
            continue
        numberRepresentationsOfCurrent = len([t for t in array if t[0] == name])
        if numberRepresentationsOfCurrent >= maxRepresentations:
            continue
        knot = CustomKnot(row["crosses"])
        if "pd" in masterDb.columns and row.pd != None:
            knot.pd = eval(row["pd"])
        array.append([name, knot])
    return array


class CompactWBImage:
    def __init__(self, initializer):
        if type(initializer) == type(np.array([[1, 1], [1, 1]])):
            self.image: np.ndarray = initializer
            self.compact = self.computeCompact()
        elif type(initializer) == list:
            self.compact: list = initializer
            self.image = self.computeImage()
        else:
            print(type(initializer))
            raise Exception("Type incorrect CompactWBImage")

    def computeCompact(self):
        shape = self.image.shape

        ones = indicesOfNumberInMatrix(self.image, 1)

        return [shape] + ones

    def computeImage(self):
        if len(self.compact) == 0:
            raise Exception("Incorrect type")
        shape = self.compact[0]

        image = np.zeros(shape, dtype=float)

        for point in self.compact[1:]:
            image[point] = 1.0

        return image


def createImagesForDatabase(
    path: str = "databases/masterPD.csv",
    limitNumberOfCrosses=7,
    limitNumberOfImages=100,
    debug=0,
):
    masterDbPD = readMasterDatabase(path=path)
    dictionary: Dict[str, List[CustomKnot]] = {}
    i = 0
    for name, knot in masterDbPD:
        numberOfCrosses = name.split("_")[0]
        if int(numberOfCrosses) > limitNumberOfCrosses:
            break
        if not name in dictionary.keys():
            dictionary[name] = []
        dictionary[name].append(knot)
    for name in dictionary.keys():
        print("Sorting", name)
        # dictionary[name] = sorted(dictionary[name],key=lambda knot: knot.numberOfStrands)
        dictionary[name] = sorted(
            dictionary[name], key=lambda knot: max(knot.pd.shape[0], knot.pd.shape[1])
        )
        print("sorted")
    # print(list(map(lambda knot: knot.numberOfStrands, dictionary["0_1"]) ))

    clear_output()

    totalOfKnots = 0

    for name in dictionary.keys():
        if debug > 0:
            print(name)
        numberOfStrands = list(map(lambda knot: knot.numberOfStrands, dictionary[name]))
        prevN = len(numberOfStrands)
        prevMax = max(numberOfStrands)
        prevMean = sum(numberOfStrands) / len(numberOfStrands)
        prevMin = min(numberOfStrands)
        dictionary[name] = dictionary[name][:limitNumberOfImages]
        totalOfKnots += len(dictionary[name])
        numberOfStrands = list(map(lambda knot: knot.numberOfStrands, dictionary[name]))
        if debug > 0:
            print("   n:    {} --> {}".format(prevN, len(numberOfStrands)))
            print("   max:  {} --> {}".format(prevMax, max(numberOfStrands)))
            print(
                "   mean: {:3.2f} --> {:3.2f}".format(
                    prevMean, sum(numberOfStrands) / len(numberOfStrands)
                )
            )
            print("   min:  {} --> {}".format(prevMin, min(numberOfStrands)))

    auxDict = {"name": [], "image": []}
    print("CalculateImages")
    for name in dictionary.keys():
        i = 0
        for knot in dictionary[name]:
            i += 1
            print(
                "knot {}, percentage {:3.2f}      ".format(
                    name, 100 * i / len(dictionary[name])
                ),
                end="\r",
            )
            auxDict["name"].append(name)
            auxDict["image"].append(knot.image())
        print()
    maxShape = max(
        max([image.shape[0] for image in auxDict["image"]]),
        max([image.shape[1] for image in auxDict["image"]]),
    )
    maxShape = [maxShape, maxShape]
    for i in range(len(auxDict["image"])):
        auxDict["image"][i] = normalizeImage(auxDict["image"][i], maxShape)
        print(
            "percentage normalize: {:3.2f}        ".format(
                100 * (i + 1) / len(auxDict["image"])
            ),
            end="\r",
        )
        pass
    print()
    for i in range(len(auxDict["image"])):
        ci = CompactWBImage(auxDict["image"][i])
        auxDict["image"][i] = ci.compact
        print(
            "percentage Compact: {:3.2f}        ".format(
                100 * (i + 1) / len(auxDict["image"])
            ),
            end="\r",
        )

    return pd.DataFrame(auxDict)
