import numpy as np
from copy import copy
blank = np.zeros((51,51),dtype=float)
upDown = copy(blank)
upDown[:,21:30] = 1
leftRight = np.rot90(upDown)
upLeft = copy(blank)
upLeft[:30,21:30] = 1
upLeft[21:30,:30] = 1
downLeft = np.rot90(upLeft)
downRight = np.rot90(upLeft,k=2)
upRight = np.rot90(upLeft,k=3)
cross1 = copy(blank)
cross1[21:30,:] = 1
cross1[:11,21:30] = 1
cross1[39:,21:30] = 1
cross2 = np.rot90(cross1)