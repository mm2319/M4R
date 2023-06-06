from Network import FeedForwardNetwork
from NN_deri_Lorenz import train_lorenz, obtain_lorenz_data
from NN_deri_two_compart import train_twocompart, obtain_twocompart_data
from NN_deri_nonlinear import train_nonlinear,obtain_nonlinear_data
from scipy.integrate import odeint
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
from lorenz import create_data_lorenz
from non_linear import create_data_nonlinear
from two_compartment import create_data_twocompart
from Derivative_Data_Lorenz import obtain_train_data_Lorenz
from Derivative_Data_NonLinear import obtain_train_data_NonLinear
from Derivative_Data_Two_Compart import obtain_train_data_Two_compart
from Bayesian_Regression_Disc_Spike_and_Slab import Bayesian_regression_disc_spike_slab
from Bayesian_Regression_Cont_Spike_and_Slab import Bayesian_regression_conti_spike_slab
from Bayesian_Regression_SS_Selection_2 import Bayesian_regression_SS_Selction
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
network_twocompart = FeedForwardNetwork(2)
network_nonlinear = FeedForwardNetwork(2)
network_lorenz = FeedForwardNetwork(3)
T, Y = create_data_lorenz(p=0.01)
network_lorenz.load_weights('./NN_lorenz_checkpoints_001/my_checkpoint')
T, Y = create_data_nonlinear(p=0.01)
network_nonlinear.load_weights('./NN_nonlinear_checkpoints_001/my_checkpoint')
T, Y = create_data_twocompart(p=0.01)
network_twocompart.load_weights('./NN_twocompart_checkpoints_001/my_checkpoint')
print("$"*25)
print("for the continuous spike and slab prior")
print("$"*25)
T, Y = create_data_twocompart(p=0.01)
x_1_train, y_1_train, y_2_train  = obtain_twocompart_data(network_twocompart, T)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_1_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of two compartment model is",start_1['z_1'])
print("the value of beta_1 in model_1 of two compartment model is",start_1['beta_1'])
print("the value of z_1 in model_2 of two compartment model is",start_2['z_1'])
print("the value of beta_1 in model_2 of two compartment model is",start_2['beta_1'])
np.save('nn_BR_CSS_001_tc_1',start_1['beta_1'])
np.save('nn_BR_CSS_001_tc_2',start_2['beta_1'])

T, Y = create_data_nonlinear(p=0.01)
x_1_train, y_1_train, y_2_train  = obtain_nonlinear_data(network_nonlinear, T)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_1_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of nonlinear compartment model is",start_1['z_1'])
print("the value of beta_1 in model_1 of nonlinear compartment model is",start_1['beta_1'])
print("the value of z_1 in model_2 of nonlinear compartment model is",start_2['z_1'])
print("the value of beta_1 in model_2 of nonlinear compartment model is",start_2['beta_1'])
np.save('nn_BR_CSS_001_nl_1',start_1['beta_1'])
np.save('nn_BR_CSS_001_nl_2',start_2['beta_1'])

T, Y = create_data_lorenz(p=0.01)
x_1_train, y_1_train, y_2_train, y_3_train  = obtain_lorenz_data(network_lorenz, T)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_1_train,np.shape(x_1_train[0])[0])
start_3,trace_3 = Bayesian_regression_conti_spike_slab(y_3_train,x_1_train,np.shape(x_1_train[0])[0])
print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of beta_1 in model_1 of lorenz model is",start_1['beta_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of beta_1 in model_2 of lorenz model is",start_2['beta_1'])
print("the value of z_1 in model_3 of lorenz model is",start_3['z_1'])
print("the value of beta_1 in model_3 of lorenz model is",start_3['beta_1'])
np.save('nn_BR_CSS_001_lr_1',start_1['beta_1'])
np.save('nn_BR_CSS_001_lr_2',start_2['beta_1'])
np.save('nn_BR_CSS_001_lr_3',start_3['beta_1'])
