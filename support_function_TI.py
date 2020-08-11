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
from parameter_setting_TI import d, A, S_L, P, N_col, N_row, time_step, time_step_POD, dt, dt_POD, dx, dy
from parameter_setting_TI import weight, Velocity_field_x, Velocity_field_y, error_rate

def Int(a):
	return a - math.floor(a)

def max_operator(a, b, c, d):
	if a < 0:
		a = 0
	if b > 0:
		b = 0
	if c < 0:
		c = 0
	if d > 0:
		d = 0
	return math.sqrt(max(a*a, b*b) + max(c*c, d*d)) - 1

def index(row, col):
	if row > N_row - 1:
		return index(row - N_row, col)
	if row < 0:
		return index(row + N_row, col)
	if col > N_col - 1:
		return index(row, col - N_col)
	if col < 0:
		return index(row, col + N_col)
	return row * N_col + col

def row_index(num):
	return int(math.floor(num / N_col))

def col_index(num):
	return int(num - row_index(num) * N_col)

def forwardx(solu):
	tmp = np.zeros((N_row, N_col))
	tmp[:, 0:N_col-1] = (solu[:, 1:N_col] - solu[:, 0:N_col-1]) / dx
	tmp[:, N_col-1] = (solu[:, 0] + np.dot(P, [1, 0]) - solu[:, N_col-1]) / dx
	return tmp

def forwardy(solu):
	tmp = np.zeros((N_row, N_col))
	tmp[0:N_row-1, :] = (solu[1:N_row, :] - solu[0:N_row-1, :]) / dy
	tmp[N_row-1, :] = (solu[0, :] + np.dot(P, [0, 1]) - solu[N_row-1, :]) / dy
	return tmp

def backwardx(solu):
	tmp = np.zeros((N_row, N_col))
	tmp[:, 1:N_col] = (solu[:, 1:N_col] - solu[:, 0:N_col-1]) / dx
	tmp[:,0] = (solu[:, 0] + np.dot(P, [1, 0]) - solu[:, N_col-1]) / dx
	return tmp

def backwardy(solu):
	tmp = np.zeros((N_row, N_col))
	tmp[1:N_row, :] = (solu[1:N_row, :] - solu[0:N_row-1, :]) / dy
	tmp[0, :] = (solu[0, :] + np.dot(P, [0, 1]) - solu[N_row-1, :]) / dy
	return tmp

def forwardx2(solu):
	tmp = np.zeros((N_row, N_col))
	tmp[:, 0:N_col-1] = solu[:, 1:N_col] - solu[:, 0:N_col-1]
	tmp[:, N_col-1] = solu[:, 0] - solu[:, N_col-1]
	return tmp

def forwardy2(solu):
	tmp = np.zeros((N_row, N_col))
	tmp[0:N_row-1, :] = solu[1:N_row, :] - solu[0:N_row-1, :]
	tmp[N_row-1, :] = solu[0, :] - solu[N_row-1, :]
	return tmp

def backwardx2(solu):
	tmp = np.zeros((N_row, N_col))
	tmp[:, 1:N_col] = solu[:, 1:N_col] - solu[:, 0:N_col-1]
	tmp[:,0] = solu[:, 0] - solu[:, N_col-1]
	return tmp

def backwardy2(solu):
	tmp = np.zeros((N_row, N_col))
	tmp[1:N_row, :] = solu[1:N_row, :] - solu[0:N_row-1, :]
	tmp[0, :] = solu[0, :] - solu[N_row-1, :]
	return tmp

'''
def Laplacian_FD(matrix):
	Dy, Dx = np.gradient(np.array(solu_current, dtype=float), dy, dx, edge_order=2)
	Dyy, Dyx = np.gradient(np.array(Dy, dtype=float), dy, dx, edge_order=2)
	Dxy, Dxx = np.gradient(np.array(Dx, dtype=float), dy, dx, edge_order=2)
	laplacian = np.zeros((N_row, N_col))
	for i in range(N_row):
		for j in range(N_col):
			if (j > 0) and (j < N_col - 1):
				laplacian[i][j] += (matrix[i][j + 1] + matrix[i][j - 1] - 2 * matrix[i][j]) / pow(dx, 2)
			elif j == 0:
				laplacian[i][j] += (-5 * matrix[i][1] + 4 * matrix[i][2] - matrix[i][3] + 2 * matrix[i][0]) / (2 * dx)
			else:
				laplacian[i][j] += (-5 * matrix[i][N_col - 2] + 4 * matrix[i][N_col - 3] - matrix[i][N_col - 4] + 2 * matrix[i][N_col - 1]) / (2 * dx)
			
			if (i > 0) and (i < N_row - 1):
				laplacian[i][j] += (matrix[i + 1][j] + matrix[i - 1][j] - 2 * matrix[i][j]) / pow(dy, 2)
			elif i == 0:
				laplacian[i][j] += (-5 * matrix[1][j] + 4 * matrix[2][j] - matrix[3][j] + 2 * matrix[0][j]) / (2 * dy)
			else:
				laplacian[i][j] += (-5 * matrix[N_row - 2][j] + 4 * matrix[N_row - 3][j] - matrix[N_row - 4][j] + 2 * matrix[N_row - 1][j]) / (2 * dy)
	return laplacian
'''

def Integration(matrix_1, matrix_2):
	matrix_3 = np.multiply(matrix_1, matrix_2)
	matrix_4 = np.multiply(matrix_3, weight)
	return np.sum(matrix_4) * dx * dy / 4.0

def Mean(matrix):
	matrix_one = np.ones((N_row + 1, N_col + 1))
	return Integration(matrix, matrix_one)

def Gradient_without_V(matrix):
	gradient = np.zeros((2, matrix.shape[0], matrix.shape[1]))
	gradient[1], gradient[0] = np.gradient(np.array(matrix, dtype=float), dy, dx, edge_order=2)
	return gradient

def Laplacian(matrix):
	Dy, Dx = np.gradient(np.array(matrix, dtype=float), dy, dx, edge_order=2)
	Dyy, Dyx = np.gradient(np.array(Dy, dtype=float), dy, dx, edge_order=2)
	Dxy, Dxx = np.gradient(np.array(Dx, dtype=float), dy, dx, edge_order=2)
	laplacian = Dxx + Dyy
	'''
	laplacian = np.zeros((N_row + 1, N_col + 1))
	for i in range(N_row + 1):
		for j in range(N_col + 1):
			if (j > 0) and (j < N_col):
				laplacian[i][j] += (matrix[i][j + 1] + matrix[i][j - 1] - 2 * matrix[i][j]) / pow(dx, 2)
			elif j == 0:
				laplacian[i][j] += (-5 * matrix[i][1] + 4 * matrix[i][2] - matrix[i][3] + 2 * matrix[i][0]) / (2 * dx)
			else:
				laplacian[i][j] += (-5 * matrix[i][N_col - 1] + 4 * matrix[i][N_col - 2] - matrix[i][N_col - 3] + 2 * matrix[i][N_col]) / (2 * dx)
			
			if (i > 0) and (i < N_row):
				laplacian[i][j] += (matrix[i + 1][j] + matrix[i - 1][j] - 2 * matrix[i][j]) / pow(dy, 2)
			elif i == 0:
				laplacian[i][j] += (-5 * matrix[1][j] + 4 * matrix[2][j] - matrix[3][j] + 2 * matrix[0][j]) / (2 * dy)
			else:
				laplacian[i][j] += (-5 * matrix[N_row - 1][j] + 4 * matrix[N_row - 2][j] - matrix[N_row - 3][j] + 2 * matrix[N_row][j]) / (2 * dy)
	'''
	return laplacian

def H1_inner_product(matrix_1, matrix_2):
	gradient_1 = Gradient_without_V(matrix_1)
	gradient_2 = Gradient_without_V(matrix_2)
	return Integration(gradient_1[0], gradient_2[0]) + Integration(gradient_1[1], gradient_2[1]) + Integration(matrix_1, matrix_2)

def H1_norm(matrix_1):
	return pow(H1_inner_product(matrix_1, matrix_1), 0.5)

def Bilinear_form(matrix_1, matrix_2):
	gradient = Gradient_without_V(matrix_1)
	laplacian = Laplacian(matrix_1)
	return Integration(np.multiply(gradient[0], Velocity_field_x) + np.multiply(gradient[1], Velocity_field_y) - d * S_L * laplacian, matrix_2)

def Nonlinear_form(matrix):
	gradient = Gradient_without_V(matrix)
	constant_function = P[0] * Velocity_field_x + P[1] * Velocity_field_y
	nonlinear_form = S_L * np.sqrt((P[0] + gradient[0])**2 + (P[1] + gradient[1])**2)
	mean = Mean(nonlinear_form)
	return nonlinear_form - mean + constant_function

def Linear_part(POD_basis):
	POD_dim = len(POD_basis)
	linear = np.zeros((POD_dim, POD_dim))
	for i in range(POD_dim):
		for j in range(POD_dim):
			linear[i][j] = Integration(POD_basis[i], POD_basis[j]) + Bilinear_form(POD_basis[j], POD_basis[i]) * dt_POD
	return linear

def Nonlinear_part(POD_basis, coefficient):
	POD_dim = len(POD_basis)
	nonlinear = np.zeros(POD_dim)
	matrix_tmp = np.zeros((N_row + 1, N_col + 1))
	for i in range(POD_dim):
		matrix_tmp += POD_basis[i] * coefficient[i]
	for i in range(POD_dim):
		nonlinear[i] = Integration(matrix_tmp, POD_basis[i]) - dt_POD * Integration(POD_basis[i], Nonlinear_form(matrix_tmp))
	return nonlinear

# Get POD basis, based on the FD solution from 0s to Ts, including the time difference of snapshots.
def POD_H1_include_time_difference_T(solu, T, plot_flag=1):
	length = len(solu)
	snapshot = np.zeros((2 * length - 1, N_row + 1, N_col + 1))
	for i in range(length):
		snapshot[i] = solu[i]
	for i in range(length, 2 * length - 1):
		snapshot[i] = (solu[i - T * time_step] - solu[i - T * time_step - 1]) / dt
	if os.path.isfile("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix with time difference.npy"):
		K = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix with time difference.npy") 
	else:
		gradient = []
		for i in range(2 * length - 1):
			gradient.append(Gradient_without_V(snapshot[i]))
		K = np.zeros((2 * length - 1, 2 * length - 1))
		for i in range(2 * length - 1):
			print('Calculating mass matrix:', i)
			for j in range(2 * length - 1):
				K[i][j] = Integration(gradient[i][0], gradient[j][0]) + Integration(gradient[i][1], gradient[j][1]) + Integration(snapshot[i], snapshot[j]) / float(2 * T * time_step + 1)
		if plot_flag:
			np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix with time difference.npy", K)

	s, u = np.linalg.eigh(K)
	eigen_sum = 0
	for i in range(len(s)):
		eigen_sum += s[i]
	error = 0
	POD_dim = 0
	for i in range(len(s)):
		error += s[len(s) - 1 - i]
		if error > (eigen_sum * error_rate):
			POD_dim = i + 1
			break
	print('POD_dim =', POD_dim)

	POD_basis = np.zeros((POD_dim, N_row + 1, N_col + 1))
	for i in range(POD_dim):
		for j in range(2 * length - 1):
			POD_basis[i] += u[j][len(s) - 1 - i] * snapshot[j] / math.sqrt(s[len(s) - 1 - i])
	if plot_flag:
		for i in range(len(POD_basis)):
			X = np.arange(0, 1 + dx, dx)
			Y = np.arange(0, 1 + dy, dy)
			Z = POD_basis[i]
			XX, YY = np.meshgrid(X, Y)
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
			ax.view_init(30, 225)
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			fig.colorbar(surf, shrink=0.5, aspect=5)
			fig.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", timedifference="+str(1)+", POD basis No."+str(i)+".eps")
	return GS(POD_basis)

# Get POD basis, based on the FD solution from 0s to Ts, excluding the time difference of snapshots.
def POD_H1_exclude_time_difference_T(solu, T, plot_flag=1):
	length = len(solu)
	snapshot = np.zeros((length, N_row + 1, N_col + 1))
	for i in range(length):
		snapshot[i] = solu[i]
	if os.path.isfile("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix without time difference.npy"):
		K = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix without time difference.npy") 
	else:
		gradient = []
		for i in range(length):
			gradient.append(Gradient_without_V(snapshot[i]))
		K = np.zeros((length, length))
		for i in range(length):
			print('Calculating mass matrix:', i)
			for j in range(length):
				K[i][j] = Integration(gradient[i][0], gradient[j][0]) + Integration(gradient[i][1], gradient[j][1]) + Integration(snapshot[i], snapshot[j]) / float(T * time_step + 1)
		if plot_flag:
			np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix without time difference.npy", K)
		
	s, u = np.linalg.eigh(K)
	eigen_sum = 0
	for i in range(len(s)):
		eigen_sum += s[i]
	error = 0
	POD_dim = 0
	for i in range(len(s)):
		error += s[len(s) - 1 - i]
		if error > (eigen_sum * error_rate):
			POD_dim = i + 1
			break
	print('POD_dim =', POD_dim)

	POD_basis = np.zeros((POD_dim, N_row + 1, N_col + 1))
	for i in range(POD_dim):
		for j in range(length):
			POD_basis[i] += u[j][len(s) - 1 - i] * snapshot[j] / math.sqrt(s[len(s) - 1 - i])
	if plot_flag:
		for i in range(len(POD_basis)):
			X = np.arange(0, 1 + dx, dx)
			Y = np.arange(0, 1 + dy, dy)
			Z = POD_basis[i]
			XX, YY = np.meshgrid(X, Y)
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
			ax.view_init(30, 225)
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			fig.colorbar(surf, shrink=0.5, aspect=5)
			fig.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", timedifference="+str(0)+", POD basis No."+str(i)+".eps")
	return GS(POD_basis)

# The FD solution for d_0 is needed.
def Save_first_n_Meanfree_POD_basis(timedifference, T, d_0, n, plot_flag=1):
	solu = np.load("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy")
	solu_periodic = np.zeros((T * time_step + 1, N_row + 1, N_col + 1))
	matrix_tmp = np.zeros((N_row + 1, N_col + 1))
	for i in range(N_row + 1):
		for j in range(N_col + 1):
			matrix_tmp[i][j] = -np.dot(P, [dx * j, dy * i])
	for i in range(T * time_step + 1):
		solu_periodic[i] = solu[i] + matrix_tmp
	solu_meanfree = np.zeros((T * time_step + 1, N_row + 1, N_col + 1))
	for i in range(T * time_step + 1):
		solu_meanfree[i] = solu_periodic[i] - Mean(solu_periodic[i])
	solu = solu_meanfree

	if timedifference == 0:
		snapshot = np.zeros((time_step * T + 1, N_row + 1, N_col + 1))
		for i in range(time_step * T + 1):
			snapshot[i] = solu[i]
		if os.path.isfile("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix without time difference.npy"):
			K = np.load("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix without time difference.npy") 
		else:
			gradient = []
			for i in range(T * time_step + 1):
				gradient.append(Gradient_without_V(snapshot[i]))
			K = np.zeros((T * time_step + 1, T * time_step + 1))
			for i in range(T * time_step + 1):
				print('Calculating mass matrix:', i)
				for j in range(T * time_step + 1):
					K[i][j] = Integration(gradient[i][0], gradient[j][0]) + Integration(gradient[i][1], gradient[j][1]) + Integration(snapshot[i], snapshot[j]) / float(T * time_step + 1)
			if plot_flag:
				np.save("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix without time difference.npy", K)
		s, u = np.linalg.eigh(K)
		POD_dim = n
		POD_basis = np.zeros((POD_dim, N_row + 1, N_col + 1))
		for i in range(POD_dim):
			for j in range(T * time_step + 1):
				POD_basis[i] += u[j][len(s) - 1 - i] * snapshot[j] / math.sqrt(s[len(s) - 1 - i])
		if plot_flag:
			np.save("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", timedifference="+str(timedifference)+", First "+str(n)+", Meanfree_POD_basis", POD_basis)
	
	if timedifference == 1:
		snapshot = np.zeros((2 * T * time_step + 1, N_row + 1, N_col + 1))
		for i in range(T * time_step + 1):
			snapshot[i] = solu[i]
		for i in range(T * time_step + 1, 2 * time_step + 1):
			snapshot[i] = (solu[i - T * time_step] - solu[i - T * time_step - 1]) / dt
		if os.path.isfile("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix with time difference.npy"):
			K = np.load("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix with time difference.npy") 
		else:
			gradient = []
			for i in range(2 * T * time_step + 1):
				gradient.append(Gradient_without_V(snapshot[i]))
			K = np.zeros((2 * T * time_step + 1, 2 * T * time_step + 1))
			for i in range(2 * T * time_step + 1):
				print('Calculating mass matrix:', i)
				for j in range(2 * T * time_step + 1):
					K[i][j] = Integration(gradient[i][0], gradient[j][0]) + Integration(gradient[i][1], gradient[j][1]) + Integration(snapshot[i], snapshot[j]) / float(2 * T * time_step + 1)
			if plot_flag:
				np.save("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", correlation matrix with time difference.npy", K)

		s, u = np.linalg.eigh(K)
		POD_dim = n
		POD_basis = np.zeros((POD_dim, N_row + 1, N_col + 1))
		for i in range(POD_dim):
			for j in range(2 * T * time_step + 1):
				POD_basis[i] += u[j][len(s) - 1 - i] * snapshot[j] / math.sqrt(s[len(s) - 1 - i])
		if plot_flag:
			np.save("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", timedifference="+str(timedifference)+", First "+str(n)+", Meanfree_POD_basis", POD_basis)
	
	if plot_flag:
		for i in range(len(POD_basis)):
			X = np.arange(0, 1 + dx, dx)
			Y = np.arange(0, 1 + dy, dy)
			Z = POD_basis[i]
			XX, YY = np.meshgrid(X, Y)
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
			ax.view_init(30, 225)
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			fig.colorbar(surf, shrink=0.5, aspect=5)
			fig.savefig("d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", timedifference="+str(timedifference)+", POD basis No."+str(i)+".eps")
	return GS(POD_basis)

def Proj(mat_list, mat):
	energy = 0
	for l in range(len(mat_list)):
		mat_list[l] -= mat * H1_inner_product(mat, mat_list[l])
		energy += H1_norm(mat_list[l]) ** 2
	return energy, mat_list

def GS(mat_list):
	for i in range(len(mat_list)):
		for j in range(i):
			mat_list[i] -= mat_list[j] * H1_inner_product(mat_list[i], mat_list[j])
		mat_list[i] = mat_list[i] / H1_norm(mat_list[i])
	return mat_list
		