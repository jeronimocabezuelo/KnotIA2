from __future__ import annotations
from CustomKnot import *
from random import *
from enum import Enum
import numpy as np
from KnotStar import Similarity
class ReidemeisterType(Enum):
    CreateALoop = 0
    UndoALoop = 1
    CreateII = 2
    UndoII = 3
    III = 4


class KnotAnt:
    def __init__(self,house: CustomKnot, ):
        pass