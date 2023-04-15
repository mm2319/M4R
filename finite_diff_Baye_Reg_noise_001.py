from Network import FeedForwardNetwork
from NN_deri_Lorenz import train_lorenz, obtain_lorenz_data
from NN_deri_two_compart import train_twocompart, obtain_twocompart_data
from NN_deri_nonlinear import train_nonlinear,obtain_nonlinear_data
from scipy.integrate import odeint
import numpy as np
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import pymc.sampling_jax
import derivative
from lorenz import create_data_lorenz
from non_linear import create_data_nonlinear
from two_compartment import create_data_twocompart
from Derivative_Data_Lorenz import obtain_train_data_Lorenz
from Derivative_Data_NonLinear import obtain_train_data_NonLinear
from Derivative_Data_Two_Compart import obtain_train_data_Two_compart
from Bayesian_Regression_Disc_Spike_and_Slab import Bayesian_regression_disc_spike_slab
from Bayesian_Regression_Cont_Spike_and_Slab import Bayesian_regression_conti_spike_slab
from Bayesian_Regression_SS_Selection_2 import Bayesian_regression_SS_Selction
print("$"*25)
print("for the discrete spike and slab prior")
print("$"*25)

T, Y = create_data_twocompart(p=0.01)

result_1 = derivative.dxdt(Y[:,0], T, kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y[:,1], T, kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_Two_compart( result_1, result_2,  1000, Y)

start_1,trace_1 = Bayesian_regression_disc_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_disc_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of two compartment model is",start_1['z_1'])
print("the value of pn_1 in model_1 of two compartment model is",start_1['pn_1'])
print("the value of z_1 in model_2 of two compartment model is",start_2['z_1'])
print("the value of pn_1 in model_2 of two compartment model is",start_2['pn_1'])

T, Y = create_data_nonlinear(p=0.01)

result_1 = derivative.dxdt(Y[:,0], T, kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y[:,1], T, kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_NonLinear( result_1, result_2, 1000,  Y)

start_1,trace_1 = Bayesian_regression_disc_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_disc_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of nonlinear model is",start_1['z_1'])
print("the value of pn_1 in model_1 of nonlinear model is",start_1['pn_1'])
print("the value of z_1 in model_2 of nonlinear model is",start_2['z_1'])
print("the value of pn_1 in model_2 of nonlinear model is",start_2['pn_1'])

T, Y = create_data_lorenz(p=0.01)

result_1 = derivative.dxdt(Y[:,0], T, kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y[:,1], T, kind="finite_difference", k=2)
result_3 = derivative.dxdt(Y[:,2], T, kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train, x_3_train, y_3_train = obtain_train_data_Lorenz( result_1, result_2, result_3,  1000,  Y)

start_1,trace_1 = Bayesian_regression_disc_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_disc_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])
start_3,trace_3 = Bayesian_regression_disc_spike_slab(y_3_train,x_3_train,np.shape(x_1_train[0])[0])
print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['pn_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['pn_1'])
print("the value of z_1 in model_3 of lorenz model is",start_3['z_1'])
print("the value of pn_1 in model_3 of lorenz model is",start_3['pn_1'])


print("$"*25)
print("for the modified spike and slab prior")
print("$"*25)

T, Y = create_data_twocompart(p=0.01)

result_1 = derivative.dxdt(Y[:,0], T, kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y[:,1], T, kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_Two_compart( result_1, result_2,1000,  Y)

start_1,trace_1 = Bayesian_regression_SS_Selction(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_SS_Selction(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of two compartment model is",start_1['z_1'])
print("the value of pn_1 in model_1 of two compartment model is",start_1['pn_1'])
print("the value of z_1 in model_2 of two compartment model is",start_2['z_1'])
print("the value of pn_1 in model_2 of two compartment model is",start_2['pn_1'])

T, Y = create_data_nonlinear(p=0.01)

result_1 = derivative.dxdt(Y[:,0], T, kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y[:,1], T, kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_NonLinear( result_1, result_2,  1000,  Y)

start_1,trace_1 = Bayesian_regression_SS_Selction(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_SS_Selction(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of nonlinear model is",start_1['z_1'])
print("the value of pn_1 in model_1 of nonlinear model is",start_1['pn_1'])
print("the value of z_1 in model_2 of nonlinear model is",start_2['z_1'])
print("the value of pn_1 in model_2 of nonlinear model is",start_2['pn_1'])

T, Y = create_data_lorenz(p=0.01)

result_1 = derivative.dxdt(Y[:,0], T, kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y[:,1], T, kind="finite_difference", k=2)
result_3 = derivative.dxdt(Y[:,2], T, kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train, x_3_train, y_3_train = obtain_train_data_Lorenz( result_1, result_2, result_3, 1000,  Y)

start_1,trace_1 = Bayesian_regression_SS_Selction(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_SS_Selction(y_2_train,x_2_train,np.shape(x_1_train[0])[0])
start_3,trace_3 = Bayesian_regression_SS_Selction(y_3_train,x_3_train,np.shape(x_1_train[0])[0])
print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['pn_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['pn_1'])
print("the value of z_1 in model_3 of lorenz model is",start_3['z_1'])
print("the value of pn_1 in model_3 of lorenz model is",start_3['pn_1'])

