import numpy as np
from copy import copy

grosor = 5
grosorLineas = int(grosor/5)
medio = grosor/2
principio = int(medio-grosorLineas/2)
fin = int(medio+grosorLineas/2)
principioCross = int(grosor/5)
finCross = int(grosor/5)*4

blank = np.zeros((grosor, grosor), dtype=float)
upDown = copy(blank)
upDown[:, principio:fin] = 1
leftRight = np.rot90(upDown)
upLeft = copy(blank)
upLeft[:fin, principio:fin] = 1
upLeft[principio:fin, :fin] = 1
downLeft = np.rot90(upLeft)
downRight = np.rot90(upLeft, k=2)
upRight = np.rot90(upLeft, k=3)
cross1 = copy(blank)
cross1[principio:fin, :] = 1
cross1[:principioCross, principio:fin] = 1
cross1[finCross:, principio:fin] = 1
cross2 = np.rot90(cross1)
