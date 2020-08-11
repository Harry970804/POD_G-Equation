import numpy as np
from numpy import linalg as LA
np.set_printoptions(threshold=np.nan)
import scipy
import os.path
import math
import collections
import time
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameter setting
d = 0.02
A = 4.0
S_L = 1.0
P = np.array([1,0])
N_col = 40
N_row = 40
time_step = 1600
time_step_POD = 1600
dt = 1.0 / time_step
dt_POD = 1.0 / time_step_POD
dx = 1.0 / N_col
dy = 1.0 / N_row

# epsilon appears in WENO_operator
epsilon = pow(10.0, -5)

error_rate = 0.999

# Set the time-independent velocity field.
def Velocity_field(x, y):
	return [-A * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y), A * math.cos(2 * math.pi * x) * math.sin(2 * math.pi * y)]

# Weight matrix used in integration.
weight = np.ones((N_row + 1, N_col + 1))
for i in range(N_row + 1):
	for j in range(N_col + 1):
		if (i > 0) and (i < N_row) and (j > 0) and (j < N_col):
			weight[i][j] += 3
		else:
			weight[i][j] += 1
weight[0][0] = 1
weight[N_row][0] = 1
weight[0][N_col] = 1
weight[N_row][N_col] = 1

# Velocity field based on the parameter chosen.
Velocity_field_x = np.zeros((N_row + 1, N_col + 1))
Velocity_field_y = np.zeros((N_row + 1, N_col + 1))
for i in range(N_row + 1):
	for j in range(N_col + 1):
		Velocity_field_x[i][j] = Velocity_field(dx * j, dy * i)[0]
		Velocity_field_y[i][j] = Velocity_field(dx * j, dy * i)[1]