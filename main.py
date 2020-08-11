# FD_solver_T(T)
# T is the time until which you want to solve the equation.
from G_Viscous_FD_TIflow import FD_solver_T

'''
	Meanfree_POD_backward_solver(timedifference, T, scheme="backward")

	timedifference = 0 if you want to exculde the time difference when generating POD basis.
	timedifference = 1 if you want to inculde the time difference when generating POD basis.
	In this solver, the # of basis is determined by the error_rate, see parameter_setting.py.
	
	Need to have the finite difference solution:
	"d="+str(d)+", d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy"

	It will return 
		the POD basis, npy
		the meanfree POD solution, npy
		the recovered POD solution, npy
		the plots of: POD basis, FD & POD solutions, including meanfree & recovered.
		a txt file containing the relative accuracy & running time.
'''
from G_Viscous_POD_TIflow import Meanfree_POD_backward_solver

'''
	Burned_volume(timedifference, T, scheme="backward")

	This function will plot the burned volume based on the 
		FD solution &
		POD solution (after recovery)
	The files
		"d="+str(d)+", d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy"
		"d="+str(d)+", d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", first "+str(n)+" POD basis from d_0="+str(d_0)+", timedifference="+str(timedifference)+", Recover_POD_solution_"+str(scheme)+".npy"
	are needed. So run it after you run FD_solver_T.
'''
from G_Viscous_POD_TIflow import Burned_volume

'''
	Burned_speed(timedifference, T, scheme="backward")

	This function will plot the burned speed based on the burned volume of
		FD solution &
		POD solution (after recovery)
	The files
		"d="+str(d)+", d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", Burned_Volume.npy"
		"d="+str(d)+", d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", time_step_POD="+str(time_step_POD)+", error_rate="+str(error_rate)+", timedifference="+str(timedifference)+", Burned_Volume_"+str(scheme)+".npy"
	are needed. So run it after you run Burned_volume.
'''
from G_Viscous_POD_TIflow import Burned_speed

'''
	Solve_different_d_backward(timedifference, T, d_0, n, scheme="backward")

	The function will solve the equation by the first n POD basis from d_0.
	
	FD_solution of d and d_0 are needed:
		"d="+str(d)+", d="+str(d)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy"
		"d="+str(d)+", d="+str(d_0)+", S_L="+str(S_L)+", A="+str(A)+", P="+str(P)+", N_x="+str(N_col)+", N_y="+str(N_row)+", time_step="+str(time_step)+", T="+str(T)+", FD_solution.npy"
	So run it after you run FD_solver_T for d and d_0.

	It will return 
		the POD basis, npy
		the meanfree POD solution, npy
		the recovered POD solution, npy
		the plots of: POD basis, FD & POD solutions, including meanfree & recovered.
		a txt file containing the relative accuracy & running time.
'''
from G_Viscous_POD_TIflow import Solve_different_d_backward

'''
	Burned_volume_different_d(timedifference, T, d_0, n, scheme="backward")
	Run it after you run Solve_different_d_backward.
'''
from G_Viscous_POD_TIflow import Burned_volume_different_d

'''
	Burned_speed_different_d(timedifference, T, d_0, n, scheme="backward")
	Run it after you run Burned_volume_different_d.
'''
from G_Viscous_POD_TIflow import Burned_speed_different_d

'''
	Solve_Interpolate_d_backward(timedifference, T, dlist, n, scheme="backward")

	The function will solve the equation by the first n POD basis interpolated from dlist.
	E.g. dlist = [0.1, 0.2, 0.8, 0.9] and d = 0.5
	FD_solution of d and all d_0 in dlist are needed.
	So run it after you run FD_solver_T for d and all d_0 in dlist.

	It will return 
		the interpolated POD basis, npy
		the meanfree POD solution, npy
		the recovered POD solution, npy
		the plots of: POD basis, FD & POD solutions, including meanfree & recovered.
		a txt file containing the relative accuracy & running time.
'''
from G_Viscous_POD_Interpolate_TIflow import Solve_Interpolate_d_backward

'''
	Burned_volume_different_d(timedifference, T, dlist, n, scheme="backward")
	Run it after you run Solve_Interpolate_d_backward.
'''
from G_Viscous_POD_Interpolate_TIflow import Burned_volume_interpolate_d

'''
	Burned_speed_Interpolate_d(timedifference, T, dlist, n, scheme="backward")
	Run it after you run Burned_volume_Interpolate_d.
'''
from G_Viscous_POD_Interpolate_TIflow import Burned_speed_Interpolate_d

from G_Viscous_POD_Adaptive_TIflow import Adaptive_POD_backward_solver, Adaptive_Burned_volume, Adaptive_Burned_speed

#Adaptive_POD_backward_solver(0, 4)
#Adaptive_Burned_volume(0, 4)
#Adaptive_Burned_speed(0, 4)

#FD_solver_T(1)
Meanfree_POD_backward_solver(0, 1)
#Meanfree_POD_backward_solver(1, 1)
'''
Solve_different_d_backward(0, 1, 0.02, 6)
Burned_volume_different_d(0, 1, 0.02, 6)
Burned_speed_different_d(0, 1, 0.02, 6)

Solve_Interpolate_d_backward(0, 1, [0.02, 0.07], 6)
Burned_volume_interpolate_d(0, 1, [0.02, 0.07], 6)
Burned_speed_Interpolate_d(0, 1, [0.02, 0.07], 6)
'''






'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

time_step = 1000
time_step_POD = 1000
dt = 1.0 / time_step
dt_POD = 1.0 / time_step_POD
T = 1.0

speed_FD = np.load("d=0.05, S_L=1.0, A=4.0, P=[1 0], N_x=80, N_y=80, time_step=1000, T=1, Burned_speed.npy")
speed_enriched_POD = np.load("d=0.05, S_L=1.0, A=4.0, P=[1 0], N_x=80, N_y=80, time_step=1000, T=1, time_step_POD=1000, first 6 POD basis interpolate from [0.02, 0.07], timedifference=0, Burned_speed_backward.npy")
speed_POD = np.load("d=0.05, S_L=1.0, A=4.0, P=[1 0], N_x=80, N_y=80, time_step=1000, T=1, time_step_POD=1000, first 6 POD basis from d_0=0.02, timedifference=0, Burned_speed_backward.npy")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(0, T + dt/2, dt), speed_FD, 'r', label='speed by finite difference')
ax.plot(np.arange(0, T + dt_POD/2, dt_POD), speed_POD, 'b--', label='speed by POD')
ax.plot(np.arange(0, T + dt_POD/2, dt_POD), speed_enriched_POD, 'g-.', label='speed by enriched POD')
ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('A(t)/t')
fig.savefig("d=0.05, S_L=1.0, A=4.0, P=[1 0], d_0=0.02, dlist=[0.02,0.07].eps")
'''