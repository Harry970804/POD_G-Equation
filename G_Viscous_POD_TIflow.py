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
from parameter_setting_TI import d, A, S_L, P, N_col, N_row, time_step, time_step_POD, dt, dt_POD, dx, dy, error_rate
from parameter_setting_TI import weight, Velocity_field_x, Velocity_field_y
from support_function_TI import Integration, Mean, Gradient_without_V, Laplacian, H1_inner_product, H1_norm
from support_function_TI import Bilinear_form, Nonlinear_form, Linear_part, Nonlinear_part
from support_function_TI import POD_H1_exclude_time_difference_T, POD_H1_include_time_difference_T, Save_first_n_Meanfree_POD_basis

# G-equation: 
# G_t + V(x,t)DG + S_L|DG| = dS_L Laplace(G)
# G(x,0) = Px
# G(x + z,t) = G(x) + Pz
# V(x,t) = cos(2*pi*y)+cos(2*pi*t)*sin(2*pi*y), cos(2*pi*x)+cos(2*pi*t)*sin(2*pi*x)
# The region [0,1]*[0,1] is divided into N_row * N_col.

def Meanfree_POD_backward_solver(timedifference, T, scheme="backward"):
	solu = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy")
	solu_periodic = np.zeros((T * time_step + 1, N_row + 1, N_col + 1))
	matrix_tmp = np.zeros((N_row + 1, N_col + 1))
	for i in range(N_row + 1):
		for j in range(N_col + 1):
			matrix_tmp[i][j] = -np.dot(P, [dx * j, dy * i])

	# Get the mean-free part of the FD solution.
	for i in range(T * time_step + 1):
		solu_periodic[i] = solu[i] + matrix_tmp
	solu_meanfree = np.zeros((T * time_step + 1, N_row + 1, N_col + 1))
	for i in range(T * time_step + 1):
		solu_meanfree[i] = solu_periodic[i] - Mean(solu_periodic[i])
	
	# Get the POD basis, the # of basis is determined by the error_rate.
	if timedifference == 1:
		POD_basis = POD_H1_include_time_difference_T(solu_meanfree, T)
	if timedifference == 0:
		POD_basis = POD_H1_exclude_time_difference_T(solu_meanfree, T)
	POD_dim = len(POD_basis)
	
	# POD solver, backward in time.
	linear = Linear_part(POD_basis)
	POD_coefficient = []
	initial = np.zeros((POD_dim))
	for i in range(POD_dim):
		initial[i] = Integration(solu_meanfree[0], POD_basis[i])
	POD_coefficient = []
	POD_coefficient.append(initial)

	t_initial = time.time()
	t_nonlinear = 0
	for i in range(T * time_step_POD):
		print (i)
		tmp_t = time.time()
		tmp = Nonlinear_part(POD_basis, POD_coefficient[i])
		t_nonlinear += time.time() - tmp_t
		POD_coefficient.append(np.linalg.solve(linear, tmp))
	t_POD = time.time() - t_initial

	# Get the meanfree solution.
	POD_solu_meanfree = []
	for i in range(T * time_step_POD + 1):
		tmp = np.zeros((N_row + 1, N_col + 1))
		for j in range(POD_dim):
			tmp += POD_coefficient[i][j] * POD_basis[j]
		POD_solu_meanfree.append(tmp)

	# Compute the relative accuracy of the meanfree solution.
	POD_accuracy_meanfree = (pow(Integration(POD_solu_meanfree[T * time_step_POD] - solu_meanfree[T * time_step], POD_solu_meanfree[T * time_step_POD] - solu_meanfree[T * time_step]), 0.5) / pow(Integration(solu_meanfree[T * time_step], solu_meanfree[T * time_step]), 0.5))
	POD_error_relative_meanfree = []
	if time_step >= time_step_POD:	
		tmp = int(time_step / time_step_POD)
		for i in range(T * time_step_POD + 1):
			tmp_1 = pow(Integration(POD_solu_meanfree[i] - solu_meanfree[i * tmp], POD_solu_meanfree[i] - solu_meanfree[i * tmp]), 0.5)
			tmp_2 = pow(Integration(solu_meanfree[i * tmp], solu_meanfree[i * tmp]), 0.5)
			POD_error_relative_meanfree.append(tmp_1 / tmp_2)
	else:
		tmp = time_step_POD / time_step
		for i in range(T * time_step + 1):
			tmp_1 = pow(Integration(POD_solu_meanfree[i * tmp] - solu_meanfree[i], POD_solu_meanfree[i * tmp] - solu_meanfree[i]), 0.5)
			tmp_2 = pow(Integration(solu_meanfree[i], solu_meanfree[i]), 0.5)
			POD_error_relative_meanfree.append(tmp_1 / tmp_2)

	# Save the solution, relative accuracy and the time used.
	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Meanfree_backward.txt", "w")
	file.write("POD_dim="+str(POD_dim)+"\n")
	file.write("Meanfree POD relative error:\n")
	file.write(str(POD_error_relative_meanfree)+"\n")
	file.write("Meanfree POD accuracy:\n")
	file.write(str(POD_accuracy_meanfree)+"\n")
	file.write("Time for POD:\n")
	file.write(str(t_POD)+"\n")
	file.write("Time for nonlinear part in POD:\n")
	file.write(str(t_nonlinear)+"\n")
	file.close()

	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Meanfree_FD_solution", solu_meanfree)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", First "+str(POD_dim)+", Meanfree_POD_basis", POD_basis)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Meanfree_POD_coefficient_backward", POD_coefficient)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Meanfree_POD_solution_backward", POD_solu_meanfree)

	X = np.arange(0, 1 + dx, dx)
	Y = np.arange(0, 1 + dy, dy)
	Z1 = solu_meanfree[T * time_step]
	Z2 = POD_solu_meanfree[T * time_step_POD]
	Z3 = solu_meanfree[T * time_step] - POD_solu_meanfree[T * time_step_POD]

	XX, YY = np.meshgrid(X, Y)

	# Plot the meanfree FD solution at time T.
	fig1 = plt.figure()
	ax1 = fig1.gca(projection='3d')
	surf1 = ax1.plot_surface(XX, YY, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax1.view_init(30, 225)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	fig1.colorbar(surf1, shrink=0.5, aspect=5)
	fig1.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Meanfree_FD.eps")

	# Plot the meanfree POD solution at time T.
	fig2 = plt.figure()
	ax2 = fig2.gca(projection='3d')
	surf2 = ax2.plot_surface(XX, YY, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax2.view_init(30, 225)
	ax2.set_xlabel("x")
	ax2.set_ylabel("y")
	fig2.colorbar(surf2, shrink=0.5, aspect=5)
	fig2.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", POD_dim="+str(POD_dim)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Meanfree_backward.eps")

	# Plot the difference between 2 meanfree solutions at time T.
	fig3 = plt.figure()
	ax3 = fig3.gca(projection='3d')
	surf3 = ax3.plot_surface(XX, YY, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax3.view_init(30, 225)
	ax3.set_xlabel("x")
	ax3.set_ylabel("y")
	fig3.colorbar(surf3, shrink=0.5, aspect=5)
	fig3.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", POD_dim="+str(POD_dim)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", difference, Meanfree_backward.eps")
	
	# Recover the meanfree POD solution.
	solu = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy")
	POD_solu_meanfree = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Meanfree_POD_solution_backward.npy")
	POD_solu_recover = POD_solu_meanfree
	mean = np.zeros(T * time_step_POD)
	for i in range(T * time_step_POD):
		gradient = Gradient_without_V(POD_solu_recover[i + 1])
		tmp = S_L * np.sqrt((P[0] + gradient[0])**2 + (P[1] + gradient[1])**2)
		mean[i] = Mean(tmp)
	for i in range(T * time_step_POD + 1):
		print (i)
		for j in range(i):
			POD_solu_recover[i] -= dt_POD * mean[j]

	matrix_tmp = np.zeros((N_row + 1, N_col + 1))
	for i in range(N_row + 1):
		for j in range(N_col + 1):
			matrix_tmp[i][j] = np.dot(P, [dx * j, dy * i])

	for i in range(T * time_step_POD + 1):
		POD_solu_recover[i] += matrix_tmp

	# Compute the relative accuracy of the recovered solution.	
	POD_accuracy_recover = (pow(Integration(POD_solu_recover[T * time_step_POD] - solu[T * time_step], POD_solu_recover[T * time_step_POD] - solu[T * time_step]), 0.5) / pow(Integration(solu[T * time_step], solu[T * time_step]), 0.5))
	POD_error_relative_recover = []
	if time_step >= time_step_POD:	
		tmp = int(time_step / time_step_POD)
		for i in range(T * time_step_POD + 1):
			tmp_1 = pow(Integration(POD_solu_recover[i] - solu[i * tmp], POD_solu_recover[i] - solu[i * tmp]), 0.5)
			tmp_2 = pow(Integration(solu[i * tmp], solu[i * tmp]), 0.5)
			POD_error_relative_recover.append(tmp_1 / tmp_2)
	else:
		tmp = time_step_POD / time_step
		for i in range(T * time_step + 1):
			tmp_1 = pow(Integration(POD_solu_recover[i * tmp] - solu[i], POD_solu_recover[i * tmp] - solu[i]), 0.5)
			tmp_2 = pow(Integration(solu[i], solu[i]), 0.5)
			POD_error_relative_recover.append(tmp_1 / tmp_2)

	# Save the solution, relative accuracy.
	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Recover_"+str(scheme)+".txt", "w")
	file.write("POD_dim="+str(POD_dim)+"\n")
	file.write("Recover POD Relative error:\n")
	file.write(str(POD_error_relative_recover)+"\n")
	file.write("Recover POD accuracy:\n")
	file.write(str(POD_accuracy_recover)+"\n")
	file.close()
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Recover_POD_solution_"+str(scheme), POD_solu_recover)

	X = np.arange(0, 1 + dx, dx)
	Y = np.arange(0, 1 + dy, dy)
	Z1 = solu[T * time_step]
	Z2 = POD_solu_recover[T * time_step_POD]
	Z3 = solu[T * time_step] - POD_solu_recover[T * time_step_POD]

	XX, YY = np.meshgrid(X, Y)

	# Plot the FD solution at time T.
	fig1 = plt.figure()
	ax1 = fig1.gca(projection='3d')
	surf1 = ax1.plot_surface(XX, YY, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax1.view_init(30, 225)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	fig1.colorbar(surf1, shrink=0.5, aspect=5)
	fig1.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD.eps")

	# Plot the recovered POD solution at time T.
	fig2 = plt.figure()
	ax2 = fig2.gca(projection='3d')
	surf2 = ax2.plot_surface(XX, YY, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax2.view_init(30, 225)
	ax2.set_xlabel("x")
	ax2.set_ylabel("y")
	fig2.colorbar(surf2, shrink=0.5, aspect=5)
	fig2.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Recover_"+str(scheme)+".eps")

	# Plot the difference between 2 solutions at time T.
	fig3 = plt.figure()
	ax3 = fig3.gca(projection='3d')
	surf3 = ax3.plot_surface(XX, YY, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax3.view_init(30, 225)
	ax3.set_xlabel("x")
	ax3.set_ylabel("y")
	fig3.colorbar(surf3, shrink=0.5, aspect=5)
	fig3.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", difference, Recover_"+str(scheme)+".eps")
	return

def Burned_volume(timedifference, T, scheme="backward"):
	solu = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy")
	POD_solu_recover = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Recover_POD_solution_"+str(scheme)+".npy")
	volume_FD = np.zeros(T * time_step + 1)
	for i in range(T * time_step + 1):
		volume_FD[i] = - Mean(np.floor(solu[i]))
	volume_POD = np.zeros(T * time_step_POD + 1)
	for i in range(T * time_step_POD + 1):
		volume_POD[i] = - Mean(np.floor(POD_solu_recover[i]))
	POD_error_relative_volume = []
	if time_step >= time_step_POD:	
		tmp = int(time_step / time_step_POD)
		for i in range(time_step_POD * T + 1):
			tmp_1 = np.absolute(volume_POD[i] - volume_FD[i * tmp])
			tmp_2 = volume_FD[i * tmp]
			POD_error_relative_volume.append(tmp_1 / tmp_2)
	else:
		tmp = time_step_POD / time_step
		for i in range(time_step * T + 1):
			tmp_1 = np.absolute(volume_POD[i * tmp] - volume_FD[i])
			tmp_2 = volume_FD[i]
			POD_error_relative_volume.append(tmp_1 / tmp_2)
	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme)+".txt", "w")
	file.write("Burned volume by finite difference:\n")
	file.write(str(volume_FD)+"\n")
	file.write("Burned volume by POD:\n")
	file.write(str(volume_POD)+"\n")
	file.write("Relative error:\n")
	file.write(str(POD_error_relative_volume)+"\n")
	file.close()
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Burned_Volume", volume_FD)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme), volume_POD)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(np.arange(0, T + dt_POD/2, dt_POD), volume_POD, 'r', label='volume by POD')
	ax.plot(np.arange(0, T + dt/2, dt), volume_FD, 'b--', label='volume by finite difference')
	ax.legend()
	ax.set_xlabel('t')
	ax.set_ylabel('A(t)')
	fig.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme)+".eps")
	return

def Burned_speed(timedifference, T, scheme="backward"):
	volume_FD = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Burned_Volume.npy")
	volume_POD = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme)+".npy")
	t0 = 20
	speed_FD = np.zeros(T * time_step + 1)
	for i in range(int(t0)):
		speed_FD[i] = volume_FD[t0] / (dt * t0)
	for i in range(T * time_step - t0 + 1):
		speed_FD[i + t0] = volume_FD[i + t0] / (dt * (i + t0))
	t0 = int(20 * time_step_POD / time_step)
	speed_POD = np.zeros(T * time_step_POD + 1)
	for i in range(int(t0)):
		speed_POD[i] = volume_POD[t0] / (dt * t0)
	for i in range(T * time_step_POD - t0 + 1):
		speed_POD[i + t0] = volume_POD[i + t0] / (dt_POD * (i + t0))
	POD_error_relative_speed = []
	if time_step >= time_step_POD:	
		tmp = int(time_step / time_step_POD)
		for i in range(time_step_POD * T + 1):
			tmp_1 = np.absolute(speed_POD[i] - speed_FD[i * tmp])
			tmp_2 = speed_FD[i * tmp]
			POD_error_relative_speed.append(tmp_1 / tmp_2)
	else:
		tmp = time_step_POD / time_step
		for i in range(time_step * T + 1):
			tmp_1 = np.absolute(speed_POD[i * tmp] - speed_FD[i])
			tmp_2 = speed_FD[i]
			POD_error_relative_speed.append(tmp_1 / tmp_2)
	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Burned_speed_"+str(scheme)+".txt", "w")
	file.write("Burned speed by finite difference:\n")
	file.write(str(speed_FD)+"\n")
	file.write("Burned speed by POD:\n")
	file.write(str(speed_POD)+"\n")
	file.write("Relative error:\n")
	file.write(str(POD_error_relative_speed)+"\n")
	file.close()
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Burned_speed", speed_FD)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Burned_speed_"+str(scheme), speed_POD)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(np.arange(0, T + dt_POD/2, dt_POD), speed_POD, 'r', label='speed by POD')
	ax.plot(np.arange(0, T + dt/2, dt), speed_FD, 'b--', label='speed by finite difference')
	ax.legend()
	ax.set_xlabel('t')
	ax.set_ylabel('A(t)/t')
	fig.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Burned_speed_"+str(scheme)+".eps")
	return

def Solve_different_d_backward(timedifference, T, d_0, n, scheme="backward"):
	POD_basis = Save_first_n_Meanfree_POD_basis(timedifference, T, d_0, n)
	solu = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy")
	solu_periodic = np.zeros((time_step * T + 1, N_row + 1, N_col + 1))
	matrix_tmp = np.zeros((N_row + 1, N_col + 1))
	for i in range(N_row + 1):
		for j in range(N_col + 1):
			matrix_tmp[i][j] = -np.dot(P, [dx * j, dy * i])
	for i in range(time_step * T + 1):
		solu_periodic[i] = solu[i] + matrix_tmp
	solu_meanfree = np.zeros((time_step * T + 1, N_row + 1, N_col + 1))
	for i in range(time_step * T + 1):
		solu_meanfree[i] = solu_periodic[i] - Mean(solu_periodic[i])
	POD_dim = len(POD_basis)
	linear = Linear_part(POD_basis)
	POD_coefficient = []
	initial = np.zeros((POD_dim))
	for i in range(POD_dim):
		initial[i] = Integration(solu_meanfree[0], POD_basis[i])
	POD_coefficient = []
	POD_coefficient.append(initial)

	t_initial = time.time()
	for i in range(T * time_step_POD):
		print (i)
		tmp = Nonlinear_part(POD_basis, POD_coefficient[i])
		POD_coefficient.append(np.linalg.solve(linear, tmp))
	t_POD = time.time() - t_initial

	POD_solu_meanfree = []
	for i in range(time_step_POD * T + 1):
		tmp = np.zeros((N_row + 1, N_col + 1))
		for j in range(POD_dim):
			tmp += POD_coefficient[i][j] * POD_basis[j]
		POD_solu_meanfree.append(tmp)

	POD_accuracy_meanfree = (pow(Integration(POD_solu_meanfree[time_step_POD * T] - solu_meanfree[time_step * T], POD_solu_meanfree[time_step_POD * T] - solu_meanfree[time_step * T]), 0.5) / pow(Integration(solu_meanfree[time_step * T], solu_meanfree[time_step * T]), 0.5))
	POD_error_relative_meanfree = []
	if time_step >= time_step_POD:	
		tmp = int(time_step / time_step_POD)
		for i in range(time_step_POD * T + 1):
			tmp_1 = pow(Integration(POD_solu_meanfree[i] - solu_meanfree[i * tmp], POD_solu_meanfree[i] - solu_meanfree[i * tmp]), 0.5)
			tmp_2 = pow(Integration(solu_meanfree[i * tmp], solu_meanfree[i * tmp]), 0.5)
			POD_error_relative_meanfree.append(tmp_1 / tmp_2)
	else:
		tmp = time_step_POD / time_step
		for i in range(time_step * T + 1):
			tmp_1 = pow(Integration(POD_solu_meanfree[i * tmp] - solu_meanfree[i], POD_solu_meanfree[i * tmp] - solu_meanfree[i]), 0.5)
			tmp_2 = pow(Integration(solu_meanfree[i], solu_meanfree[i]), 0.5)
			POD_error_relative_meanfree.append(tmp_1 / tmp_2)

	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Meanfree_backward.txt", "w")
	file.write("POD_dim="+str(POD_dim)+"\n")
	file.write("Meanfree Relative error:\n")
	file.write(str(POD_error_relative_meanfree)+"\n")
	file.write("Meanfree POD accuracy:\n")
	file.write(str(POD_accuracy_meanfree)+"\n")
	file.write("Time for POD:\n")
	file.write(str(t_POD)+"\n")
	file.close()

	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Meanfree_FD_solution", solu_meanfree)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Meanfree_POD_coefficient_backward", POD_coefficient)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Meanfree_POD_solution_backward", POD_solu_meanfree)

	X = np.arange(0, 1 + dx, dx)
	Y = np.arange(0, 1 + dy, dy)
	Z1 = solu_meanfree[time_step * T]
	Z2 = POD_solu_meanfree[time_step_POD * T]
	Z3 = solu_meanfree[time_step * T] - POD_solu_meanfree[time_step_POD * T]

	XX, YY = np.meshgrid(X, Y)

	fig1 = plt.figure()
	ax1 = fig1.gca(projection='3d')
	surf1 = ax1.plot_surface(XX, YY, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax1.view_init(30, 225)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	fig1.colorbar(surf1, shrink=0.5, aspect=5)
	fig1.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Meanfree_FD.eps")

	fig2 = plt.figure()
	ax2 = fig2.gca(projection='3d')
	surf2 = ax2.plot_surface(XX, YY, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax2.view_init(30, 225)
	ax2.set_xlabel("x")
	ax2.set_ylabel("y")
	fig2.colorbar(surf2, shrink=0.5, aspect=5)
	fig2.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", POD_dim="+str(POD_dim)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Meanfree_backward.eps")

	fig3 = plt.figure()
	ax3 = fig3.gca(projection='3d')
	surf3 = ax3.plot_surface(XX, YY, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax3.view_init(30, 225)
	ax3.set_xlabel("x")
	ax3.set_ylabel("y")
	fig3.colorbar(surf3, shrink=0.5, aspect=5)
	fig3.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", POD_dim="+str(POD_dim)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", difference, Meanfree_backward.eps")
	
	#Recover the solution.
	POD_solu_recover = POD_solu_meanfree
	mean = np.zeros(time_step_POD * T)
	for i in range(time_step_POD * T):
		gradient = Gradient_without_V(POD_solu_recover[i + 1])
		tmp = S_L * np.sqrt((P[0] + gradient[0])**2 + (P[1] + gradient[1])**2)
		mean[i] = Integration(tmp, np.ones((N_row + 1, N_col + 1)))
	for i in range(time_step_POD * T + 1):
		print (i)
		for j in range(i):
			POD_solu_recover[i] -= dt_POD * mean[j]

	matrix_tmp = np.zeros((N_row + 1, N_col + 1))
	for i in range(N_row + 1):
		for j in range(N_col + 1):
			matrix_tmp[i][j] = np.dot(P, [dx * j, dy * i])

	for i in range(time_step_POD * T + 1):
		POD_solu_recover[i] += matrix_tmp

	POD_accuracy_recover = (pow(Integration(POD_solu_recover[time_step_POD * T] - solu[time_step * T], POD_solu_recover[time_step_POD * T] - solu[time_step * T]), 0.5) / pow(Integration(solu[time_step * T], solu[time_step * T]), 0.5))
	POD_error_relative_recover = []
	if time_step >= time_step_POD:	
		tmp = int(time_step / time_step_POD)
		for i in range(time_step_POD * T + 1):
			tmp_1 = pow(Integration(POD_solu_recover[i] - solu[i * tmp], POD_solu_recover[i] - solu[i * tmp]), 0.5)
			tmp_2 = pow(Integration(solu[i * tmp], solu[i * tmp]), 0.5)
			POD_error_relative_recover.append(tmp_1 / tmp_2)
	else:
		tmp = time_step_POD / time_step
		for i in range(time_step * T + 1):
			tmp_1 = pow(Integration(POD_solu_recover[i * tmp] - solu[i], POD_solu_recover[i * tmp] - solu[i]), 0.5)
			tmp_2 = pow(Integration(solu[i], solu[i]), 0.5)
			POD_error_relative_recover.append(tmp_1 / tmp_2)

	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Recover_"+str(scheme)+".txt", "w")
	file.write("POD_dim="+str(POD_dim)+"\n")
	file.write("Recover Relative error:\n")
	file.write(str(POD_error_relative_recover)+"\n")
	file.write("Recover POD accuracy:\n")
	file.write(str(POD_accuracy_recover)+"\n")
	file.close()
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Recover_POD_solution_"+str(scheme), POD_solu_recover)

	X = np.arange(0, 1 + dx, dx)
	Y = np.arange(0, 1 + dy, dy)
	Z1 = solu[time_step * T]
	Z2 = POD_solu_recover[time_step_POD * T]
	Z3 = solu[time_step * T] - POD_solu_recover[time_step_POD * T]

	XX, YY = np.meshgrid(X, Y)

	fig1 = plt.figure()
	ax1 = fig1.gca(projection='3d')
	surf1 = ax1.plot_surface(XX, YY, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax1.view_init(30, 225)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	fig1.colorbar(surf1, shrink=0.5, aspect=5)
	fig1.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD.eps")

	fig2 = plt.figure()
	ax2 = fig2.gca(projection='3d')
	surf2 = ax2.plot_surface(XX, YY, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax2.view_init(30, 225)
	ax2.set_xlabel("x")
	ax2.set_ylabel("y")
	fig2.colorbar(surf2, shrink=0.5, aspect=5)
	fig2.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Recover_"+str(scheme)+".eps")

	fig3 = plt.figure()
	ax3 = fig3.gca(projection='3d')
	surf3 = ax3.plot_surface(XX, YY, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax3.view_init(30, 225)
	ax3.set_xlabel("x")
	ax3.set_ylabel("y")
	fig3.colorbar(surf3, shrink=0.5, aspect=5)
	fig3.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", difference, Recover_"+str(scheme)+".eps")
	return

def Burned_volume_different_d(timedifference, T, d_0, n, scheme="backward"):
	solu = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy")
	POD_solu_recover = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Recover_POD_solution_"+str(scheme)+".npy")
	volume_FD = np.zeros(T * time_step + 1)
	for i in range(T * time_step + 1):
		volume_FD[i] = - Mean(np.floor(solu[i]))
	volume_POD = np.zeros(T * time_step_POD + 1)
	for i in range(T * time_step_POD + 1):
		volume_POD[i] = - Mean(np.floor(POD_solu_recover[i]))
	POD_error_relative_volume = []
	if time_step >= time_step_POD:	
		tmp = int(time_step / time_step_POD)
		for i in range(time_step_POD * T + 1):
			tmp_1 = np.absolute(volume_POD[i] - volume_FD[i * tmp])
			tmp_2 = volume_FD[i * tmp]
			POD_error_relative_volume.append(tmp_1 / tmp_2)
	else:
		tmp = time_step_POD / time_step
		for i in range(time_step * T + 1):
			tmp_1 = np.absolute(volume_POD[i * tmp] - volume_FD[i])
			tmp_2 = volume_FD[i]
			POD_error_relative_volume.append(tmp_1 / tmp_2)
	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme)+".txt", "w")
	file.write("Burned volume by finite difference:\n")
	file.write(str(volume_FD)+"\n")
	file.write("Burned volume by POD:\n")
	file.write(str(volume_POD)+"\n")
	file.write("Relative error:\n")
	file.write(str(POD_error_relative_volume)+"\n")
	file.close()
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Burned_Volume", volume_FD)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme), volume_POD)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(np.arange(0, T + dt_POD/2, dt_POD), volume_POD, 'r', label='volume by POD')
	ax.plot(np.arange(0, T + dt/2, dt), volume_FD, 'b--', label='volume by finite difference')
	ax.legend()
	ax.set_xlabel('t')
	ax.set_ylabel('A(t)')
	fig.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme)+".eps")
	return

def Burned_speed_different_d(timedifference, T, d_0, n, scheme="backward"):
	volume_FD = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Burned_Volume.npy")
	volume_POD = np.load("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme)+".npy")
	t0 = 20
	speed_FD = np.zeros(T * time_step + 1)
	for i in range(int(t0)):
		speed_FD[i] = volume_FD[t0] / (dt * t0)
	for i in range(T * time_step - t0 + 1):
		speed_FD[i + t0] = volume_FD[i + t0] / (dt * (i + t0))
	t0 = int(20 * time_step_POD / time_step)
	speed_POD = np.zeros(T * time_step_POD + 1)
	for i in range(int(t0)):
		speed_POD[i] = volume_POD[t0] / (dt * t0)
	for i in range(T * time_step_POD - t0 + 1):
		speed_POD[i + t0] = volume_POD[i + t0] / (dt_POD * (i + t0))
	POD_error_relative_speed = []
	if time_step >= time_step_POD:	
		tmp = int(time_step / time_step_POD)
		for i in range(time_step_POD * T + 1):
			tmp_1 = np.absolute(speed_POD[i] - speed_FD[i * tmp])
			tmp_2 = speed_FD[i * tmp]
			POD_error_relative_speed.append(tmp_1 / tmp_2)
	else:
		tmp = time_step_POD / time_step
		for i in range(time_step * T + 1):
			tmp_1 = np.absolute(speed_POD[i * tmp] - speed_FD[i])
			tmp_2 = speed_FD[i]
			POD_error_relative_speed.append(tmp_1 / tmp_2)
	file = open("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Burned_speed_"+str(scheme)+".txt", "w")
	file.write("Burned speed by finite difference:\n")
	file.write(str(speed_FD)+"\n")
	file.write("Burned speed by POD:\n")
	file.write(str(speed_POD)+"\n")
	file.write("Relative error:\n")
	file.write(str(POD_error_relative_speed)+"\n")
	file.close()
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Burned_speed", speed_FD)
	np.save("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Burned_speed_"+str(scheme), speed_POD)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(np.arange(0, T + dt_POD/2, dt_POD), speed_POD, 'r', label='speed by POD')
	ax.plot(np.arange(0, T + dt/2, dt), speed_FD, 'b--', label='speed by finite difference')
	ax.legend()
	ax.set_xlabel('t')
	ax.set_ylabel('A(t)/t')
	fig.savefig("d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Burned_speed_"+str(scheme)+".eps")
	return
