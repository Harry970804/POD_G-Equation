import numpy as np
from numpy import linalg as LA
np.set_printoptions(threshold=np.nan)
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import time
import collections
from parameter_setting_TI import d, A, S_L, P, N_col, N_row, time_step, dt, dx, dy, epsilon
from support_function_TI import Int, max_operator, index, row_index, col_index, Laplacian
from support_function_TI import forwardx, forwardy, backwardx, backwardy, forwardx2, forwardy2, backwardx2, backwardy2

# G-equation: 
# G_t + V(x,t)DG + S_L|DG| = dS_L Laplace(G)
# G(x,0) = Px
# G(x + z,t) = G(x) + Pz
# V(x,t) = cos(2*pi*y)+cos(2*pi*t)*sin(2*pi*y), cos(2*pi*x)+cos(2*pi*t)*sin(2*pi*x)
# The region [0,1]*[0,1] is divided into N_row * N_col.

# Set the time-independent velocity field.
def Velocity_field(x, y):
	return [-A * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y), A * math.cos(2 * math.pi * x) * math.sin(2 * math.pi * y)]

def Velocity_fieldx(x, y):
	return -A * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)

def Velocity_fieldy(x, y):
	return A * math.cos(2 * math.pi * x) * math.sin(2 * math.pi * y)

# Vectorize-form of the above functions.
vVelocity_fieldx = np.vectorize(Velocity_fieldx)
vVelocity_fieldy = np.vectorize(Velocity_fieldy)

X = np.arange(0, 1, dx)
Y = np.arange(0, 1, dy)
XX, YY = np.meshgrid(X, Y)

Vx = vVelocity_fieldx(XX, YY)
Vy = vVelocity_fieldy(XX, YY)

# WENO, see [Jiang, Peng] Weighted ENO schemes for Hamilton-Jacobi equations.
def WENO_operator(a, b, c, d):
	IS_0 = 13.0 * pow((a - b), 2) + 3.0 * pow((a - 3 * b), 2)
	IS_1 = 13.0 * pow((b - c), 2) + 3.0 * pow((b + c), 2)
	IS_2 = 13.0 * pow((c - d), 2) + 3.0 * pow((d - 3 * c), 2)
	alpha_0 = 1.0 / pow(epsilon + IS_0, 2)
	alpha_1 = 6.0 / pow(epsilon + IS_1, 2)
	alpha_2 = 3.0 / pow(epsilon + IS_2, 2)
	omega_0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2)
	omega_2 = alpha_2 / (alpha_0 + alpha_1 + alpha_2)
	return omega_0 * (a - 2 * b + c) / 3.0 + (omega_2 - 0.5) * (b - 2 * c + d) / 6.0

# Vectorize-form of the above function.
vWENO_operator = np.vectorize(WENO_operator)

# WENO, see [Jiang, Peng] Weighted ENO schemes for Hamilton-Jacobi equations.
def WENO5(solu):
	solum1x = np.zeros((N_row, N_col))
	solum1x[:, 1:N_col] = solu[:, 0:N_col-1]
	solum1x[:, 0] = solu[:, N_col-1] - np.dot(P, [1, 0])

	solum2x = np.zeros((N_row, N_col))
	solum2x[:, 2:N_col] = solu[:, 0:N_col-2]
	solum2x[:, 0] = solu[:, N_col-2] - np.dot(P, [1, 0])
	solum2x[:, 1] = solu[:, N_col-1] - np.dot(P, [1, 0])

	solup1x = np.zeros((N_row, N_col))
	solup1x[:, 0:N_col-1] = solu[:, 1:N_col]
	solup1x[:, N_col-1] = solu[:, 0] + np.dot(P, [1, 0])

	solup2x = np.zeros((N_row, N_col))
	solup2x[:, 0:N_col-2] = solu[:, 2:N_col]
	solup2x[:, N_col-2] = solu[:, 0] + np.dot(P, [1, 0])
	solup2x[:, N_col-1] = solu[:, 1] + np.dot(P, [1, 0])

	solum1y = np.zeros((N_row, N_col))
	solum1y[1:N_row, :] = solu[0:N_row-1, :]
	solum1y[0, :] = solu[N_row-1, :] - np.dot(P, [0, 1])

	solum2y = np.zeros((N_row, N_col))
	solum2y[2:N_row, :] = solu[0:N_row-2, :]
	solum2y[0, :] = solu[N_row-2, :] - np.dot(P, [0, 1])
	solum2y[1, :] = solu[N_row-1, :] - np.dot(P, [0, 1])

	solup1y = np.zeros((N_row, N_col))
	solup1y[0:N_row-1, :] = solu[1:N_row, :]
	solup1y[N_row-1, :] = solu[0, :] + np.dot(P, [0, 1])

	solup2y = np.zeros((N_row, N_col))
	solup2y[0:N_row-2, :] = solu[2:N_row, :]
	solup2y[N_row-2, :] = solu[0, :] + np.dot(P, [0, 1])
	solup2y[N_row-1, :] = solu[1, :] + np.dot(P, [0, 1])

	solu_current_WENO5 = np.zeros((N_row, N_col, 5))
	solu_current_WENO5[:,:,0] = solu
	tmpx = (-forwardx(solum2x) + 7*forwardx(solum1x) + 7*forwardx(solu) -forwardx(solup1x))/12.0
	tmpy = (-forwardy(solum2y) + 7*forwardy(solum1y) + 7*forwardy(solu) -forwardy(solup1y))/12.0
	WENO5x1 = vWENO_operator(backwardx2(forwardx(solup2x)), backwardx2(forwardx(solup1x)), backwardx2(forwardx(solu)), backwardx2(forwardx(solum1x)))
	WENO5x2 = vWENO_operator(backwardx2(forwardx(solum2x)), backwardx2(forwardx(solum1x)), backwardx2(forwardx(solu)), backwardx2(forwardx(solup1x)))
	WENO5y1 = vWENO_operator(backwardy2(forwardy(solup2y)), backwardy2(forwardy(solup1y)), backwardy2(forwardy(solu)), backwardy2(forwardy(solum1y)))
	WENO5y2 = vWENO_operator(backwardy2(forwardy(solum2y)), backwardy2(forwardy(solum1y)), backwardy2(forwardy(solu)), backwardy2(forwardy(solup1y)))
	solu_current_WENO5[:,:,1] = tmpx + WENO5x1
	solu_current_WENO5[:,:,2] = tmpx - WENO5x2
	solu_current_WENO5[:,:,3] = tmpy + WENO5y1
	solu_current_WENO5[:,:,4] = tmpy - WENO5y2

	return solu_current_WENO5

# Hamiltonian, upwinding scheme and Godunov scheme. See A numerical study of turbulent flame speeds of curvature and strain G-equations in cellular flows.
def hamiltonian_viscous_smalld(pmx, ppx, pmy, ppy, vx, vy):
	if vx > 0:
		p_x_vel = pmx
	else:
		p_x_vel = ppx

	if vy > 0:
		p_y_vel = pmy
	else:
		p_y_vel = ppy

	if vx > S_L:
		p_x_nor_2 = pow(pmx, 2)
	elif vx < -S_L:
		p_x_nor_2 = pow(ppx, 2)
	else:
		p_x_nor_2 = max(pow(max(pmx, 0), 2), pow(min(ppx, 0), 2))

	if vy > S_L:
		p_y_nor_2 = pow(pmy, 2)
	elif vy < -S_L:
		p_y_nor_2 = pow(ppy, 2)
	else:
		p_y_nor_2 = max(pow(max(pmy, 0), 2), pow(min(ppy, 0), 2))

	hamiltonian = -(vx * p_x_vel + vy * p_y_vel + S_L * math.sqrt(p_x_nor_2 + p_y_nor_2))
	return hamiltonian

# Vectorize-form of the above function.
vhamiltonian_viscous_smalld = np.vectorize(hamiltonian_viscous_smalld)

# Hamiltonian, upwinding scheme and Godunov scheme. See A numerical study of turbulent flame speeds of curvature and strain G-equations in cellular flows.
# For the case when d is small.
def Hamiltonian_viscous_smalld(solu_current_WENO5):
	tmp = vhamiltonian_viscous_smalld(solu_current_WENO5[:,:,1], solu_current_WENO5[:,:,2], solu_current_WENO5[:,:,3], solu_current_WENO5[:,:,4], Vx, Vy)
	tmp = tmp + d * S_L * Laplacian(solu_current_WENO5[:,:,0])
	return tmp

# Semi-implicit scheme. See A numerical study of turbulent flame speeds of curvature and strain G-equations in cellular flows.
# For the case when d is large.
def semi_implicit_larged(solu_current):
	matrix = np.zeros((N_row * N_col, N_row * N_col))
	vector = np.zeros((N_row * N_col))
	solu_current_WENO5 = WENO5(solu_current)
	for i in range(N_row * N_col):
		row = row_index(i)
		col = col_index(i)
		x = dx * col
		y = dy * row
		tmp = Velocity_field(x, y)
		if tmp[0] > S_L:
			p_x_nor_2 = pow(solu_current_WENO5[row][col][2], 2)
		elif tmp[0] < -S_L:
			p_x_nor_2 = pow(solu_current_WENO5[row][col][1], 2)
		else:
			p_x_nor_2 = max(pow(max(solu_current_WENO5[row][col][2], 0), 2), pow(min(solu_current_WENO5[row][col][1], 0), 2))

		if tmp[1] > S_L:
			p_y_nor_2 = pow(solu_current_WENO5[row][col][4], 2)
		elif tmp[1] < -S_L:
			p_y_nor_2 = pow(solu_current_WENO5[row][col][3], 2)
		else:
			p_y_nor_2 = max(pow(max(solu_current_WENO5[row][col][4], 0), 2), pow(min(solu_current_WENO5[row][col][3], 0), 2))
		vector[i] = solu_current[row][col] / dt - S_L * math.sqrt(p_x_nor_2 + p_y_nor_2)

		if row == 0:
			vector[i] -= P[1] * (tmp[1] / (2 * dy) + d * S_L / pow(dy, 2))
		if col == 0:
			vector[i] -= P[0] * (tmp[0] / (2 * dx) + d * S_L / pow(dx, 2))
		if row == N_row - 1:
			vector[i] -= P[1] * (tmp[1] / (2 * dy) - d * S_L / pow(dy, 2))
		if col == N_col - 1:
			vector[i] -= P[0] * (tmp[0] / (2 * dx) - d * S_L / pow(dx, 2))

		matrix[i][i] = 1.0 / dt + 2 * d * S_L * (1.0 / pow(dx, 2) + 1.0 / pow(dy, 2))
		matrix[i][index(row, col + 1)] = tmp[0] / (2 * dx) - d * S_L / pow(dx, 2)
		matrix[i][index(row, col - 1)] = - tmp[0] / (2 * dx) - d * S_L / pow(dx, 2)
		matrix[i][index(row + 1, col)] = tmp[1] / (2 * dy) - d * S_L / pow(dy, 2)
		matrix[i][index(row - 1, col)] = - tmp[1] / (2 * dy) - d * S_L / pow(dy, 2)

	spmatrix = csr_matrix(matrix)
	solu = spsolve(spmatrix, vector)
	solu_next = solu.reshape((N_row, N_col))
	return solu_next

# RK3 for time discretization, when d is small.
def RK3_smalld(solu_current):
	solu_next = np.zeros((N_row, N_col))
	temp_1 = Hamiltonian_viscous_smalld(WENO5(solu_current))
	temp_2 = Hamiltonian_viscous_smalld(WENO5(solu_current + 1.0/3.0 * dt * temp_1))
	temp_3 = Hamiltonian_viscous_smalld(WENO5(solu_current + 2.0/3.0 * dt * temp_2))
	solu_next = solu_current + (1.0/4.0 * temp_1 + 3.0/4.0 * temp_3) * dt
	return solu_next

# Finite difference solver, from 0s to Ts.
# Save the solution as np array, (time_step * T + 1, N_row + 1, N_col + 1) in
# "d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy"

def FD_solver_T(T):
	solu_real = np.zeros((time_step * T + 1, N_row, N_col))
	solu_initial = np.zeros((N_row, N_col))

	for i in range(N_row):
		for j in range(N_col):
			solu_initial[i][j] = np.dot(P, [dx * j, dy * i])
	solu_real[0] = solu_initial
	t_initial = time.time()

	"""
	#small d: dt*((abs(Velocity_field(x,y)[0])+S_L)/dx+(abs(Velocity_field(x,y)[1])+S_L)/dy)<1
	for i in range(time_step * T):
		solu_real.append((RK3_smalld(solu_real[i], dt * i)))
		print(i)
	"""

	#large d: dt*(S_L/dx+S_L/dy)<1
	for i in range(time_step * T):
		solu_real[i + 1] = semi_implicit_larged(solu_real[i])
		print(i)
	t_real = time.time() - t_initial

	solu = np.zeros((time_step * T + 1, N_row + 1, N_col + 1))
	solu[:, 0:N_row, 0:N_col] = solu_real
	solu[:, N_row, 0:N_col] = solu[:, 0, 0:N_col] + np.dot(P, [0, 1])
	solu[:, :, N_col] = solu[:, :, 0] + np.dot(P, [1, 0])

	"""
	for i in range(T * time_step + 1):
		solu_periodic[i] = solu[i] + matrix_tmp
	solu_meanfree = np.zeros((T * time_step + 1, N_row + 1, N_col + 1))
	for i in range(T * time_step + 1):
		solu_meanfree[i] = solu_periodic[i] - Mean(solu_periodic[i])
	"""

	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD.txt", "w")
	file.write("Time for finite difference:\n")
	file.write(str(t_real)+"\n")
	file.close()

	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution", solu)

	return

