from multiprocessing import Lock, Process, Queue, current_process, managers
import multiprocessing
from time import sleep
from KnotDatabase import *

maxCrosses = 3
numberOfKnots = 550
numberOfRandomMov = 500
maxStrands = 100
debug = 0


def do_job(mainDict, a, b):
    processName = current_process().name
    while True:
        # printLog(processName + ": Looking new task")
        number = None
        for name in mainDict["database"].keys():
            if len(mainDict["database"][name]["crosses"]) < numberOfKnots:
                number = len(mainDict["database"][name]["crosses"]) + 1
                break
        if number == None:
            printLog(processName + ": end")
            break
        task = name + " " + str(number)
        printLog(processName + ": init task " + task)

        knotSet = mainDict["knotSetNames"][name]
        basicCrosses = mainDict["basicKnot"][name]
        if number == 1:
            knot = CustomKnot(deepcopy(basicCrosses))
        else:
            posibleCrosses = deepcopy(mainDict["database"][name]["crosses"])
            crosses = choice(posibleCrosses)
            knot = CustomKnot(deepcopy(crosses))

        knot.randomMovN(
            numberOfRandomMov, maxStrands, percentage=False, debug=debug - 1
        )
        # Quitar cruces innecesarios y comprobar que no este en la base de datos.
        knot.reduceUnnecessaryMov()
        # printLog(processName + ": " + task + " finish reduceUnnecessaryMov")
        if not knot.representationForHash in knotSet:
            knotSet.append(knot.representationForHash)
            mainDict["database"][name]["crosses"].append(knot.crosses)
            mainDict["database"][name]["numberOfStrands"].append(knot.numberOfStrands)

            printLog(processName + ": finish task " + task + " successfully")
            mainDict["taskComplete"].append((processName, task))
        else:
            printLog(processName + ": finish task " + task + " failure")

            sleep(0.5)

        completes = 0
        needs = 0
        for name in mainDict["database"].keys():
            completes += min(len(mainDict["database"][name]["crosses"]), numberOfKnots)
            needs += numberOfKnots

        printLog("state:  {}/{}".format(completes, needs))

    return True


def main():
    names = knotNamesList(maxCrosses)
    df = singleDatabase(maxCrosses)
    df = df.reset_index()
    manager = multiprocessing.Manager()

    mainDict = manager.dict()

    mainDict["basicKnot"] = manager.dict()
    mainDict["database"] = manager.dict()
    mainDict["knotSetNames"] = manager.dict()
    mainDict["taskComplete"] = manager.list()

    # Añadimos todos los nudos que ya están en el master
    masterDb = pd.read_csv("databases/master.csv")
    if type(masterDb) != type(None):
        for index, row in masterDb.iterrows():
            name = row["name"]
            if not name in names:
                continue
            if not name in mainDict["knotSetNames"].keys():
                mainDict["knotSetNames"][name] = manager.list()
            if name in list(masterDb["name"]):
                crosses = row["crosses"]
                knot = CustomKnot(crosses)
                if not knot.representationForHash in mainDict["knotSetNames"][name]:
                    mainDict["knotSetNames"][name].append(knot.representationForHash)
                    if not name in mainDict["database"].keys():
                        mainDict["database"][name] = manager.dict()
                        mainDict["database"][name]["crosses"] = manager.list()
                        mainDict["database"][name]["numberOfStrands"] = manager.list()

                    mainDict["database"][name]["crosses"].append(deepcopy(crosses))
                    mainDict["database"][name]["numberOfStrands"].append(
                        knot.numberOfStrands
                    )

    number_of_processes = 8

    for index, row in df.iterrows():
        name = row["name"]
        if not name in mainDict["knotSetNames"].keys():
            mainDict["knotSetNames"][name] = manager.list()
        if not name in mainDict["basicKnot"].keys():
            mainDict["basicKnot"][name] = row["crosses"]

    processes = []
    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(mainDict, "b", "adios"))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    taskComplete = dict()
    for processName, task in mainDict["taskComplete"]:
        if not processName in taskComplete.keys():
            taskComplete[processName] = list()
        taskComplete[processName].append(task)

    for processName in taskComplete.keys():
        i = 0
        printLog(processName + " Complete tasks: ")
        for task in taskComplete[processName]:
            i += 1
            printLog("     " + task)
        printLog("     " + "Total task {}".format(i))

    newAuxDict = {"name": [], "crosses": [], "numberOfStrands": []}
    for name in mainDict["database"].keys():
        for crosses in mainDict["database"][name]["crosses"]:
            newAuxDict["name"].append(name)
            newAuxDict["crosses"].append(crosses)
        for numberOfStrands in mainDict["database"][name]["numberOfStrands"]:
            newAuxDict["numberOfStrands"].append(numberOfStrands)

    db = pd.DataFrame(newAuxDict)
    masterDb = combineDatabase([db, masterDb])
    masterDb.to_csv("databases/master.csv")
    return True


if __name__ == "__main__":
    start = time()
    main()
    end = time()
    printLog(remainingTimeString(end - start))
