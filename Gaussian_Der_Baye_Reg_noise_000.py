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

import derivative
from lorenz import create_data_lorenz
from non_linear import create_data_nonlinear
from two_compartment import create_data_twocompart
from Derivative_Data_Lorenz import obtain_train_data_Lorenz
from Derivative_Data_NonLinear import obtain_train_data_NonLinear
from Derivative_Data_Two_Compart import obtain_train_data_Two_compart
from Bayesian_Regression_Disc_Spike_and_Slab import Bayesian_regression_disc_spike_slab
from Bayesian_Regression_SS_Selection_2 import Bayesian_regression_SS_Selction
from Gaussian_process_der import GP, GP_derivative,rbf,rbf_fd,rbf_pd
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
# finds the hyperparameters for two_compart
T, Y = create_data_twocompart(p=0.0)
@use_named_args([Real(1e-7, 1e+1, name='theta_1'),
        Real(1e-7, 1e+1, name='theta_2'),
        Real(1e-7, 1e+1, name='sigma')])    
def evaluate_model_1(**params):
    gp = GP(kernel=rbf,kernel_diff=rbf_pd)
    theta=[params["theta_1"],params["theta_2"]] 
    y_pred = gp.loglikelihood(
                  x_star=np.linspace(0,10,1000),  # set to test points
                  X = np.array(T),     # set to observed x
                  y = np.array(Y[:,0]),       # set to observed y
                  size=1,    # draw 100 posterior samples 
                  theta=theta,
                  sigma=params["sigma"]
                )
    negative_logli= y_pred
    return -negative_logli
# Bayesian Optimisation
bounds = [(1e-7, 1.e+1), (1e-7, 1.e+1), (1e-7, 1.e+1)]
para_two_compart_1 = gp_minimize(evaluate_model_1, bounds, n_calls=250)
@use_named_args([Real(1e-7, 1e+1, name='theta_1'),
        Real(1e-7, 1e+1, name='theta_2'),
        Real(1e-7, 1e+1, name='sigma')]) 
def evaluate_model_1(**params):
    gp = GP(kernel=rbf,kernel_diff=rbf_pd)
    theta=[params["theta_1"],params["theta_2"]] 
    y_pred = gp.loglikelihood(
                  x_star=np.linspace(0,10,1000),  # set to test points
                  X = np.array(T),     # set to observed x
                  y = np.array(Y[:,1]),       # set to observed y
                  size=1,    # draw 100 posterior samples 
                  theta=theta,
                  sigma=params["sigma"]
                )
    negative_logli= y_pred
    return -negative_logli
bounds = [(1e-7, 1.e+1), (1e-7, 1.e+1), (1e-7, 1.e+1)]
para_two_compart_2 = gp_minimize(evaluate_model_1, bounds, n_calls=250)

# finds the hyperparameters for nonlinear
T, Y = create_data_nonlinear(p=0.0)
@use_named_args([Real(1e-7, 1e+1, name='theta_1'),
        Real(1e-7, 1e+1, name='theta_2'),
        Real(1e-7, 1e+1, name='sigma')])    
def evaluate_model_1(**params):
    gp = GP(kernel=rbf,kernel_diff=rbf_pd)
    theta=[params["theta_1"],params["theta_2"]] 
    y_pred = gp.loglikelihood(
                  x_star=np.linspace(0,10,1000),  # set to test points
                  X = np.array(T),     # set to observed x
                  y = np.array(Y[:,0]),       # set to observed y
                  size=1,    # draw 100 posterior samples 
                  theta=theta,
                  sigma=params["sigma"]
                )
    negative_logli= y_pred
    return -negative_logli
# Bayesian Optimisation
bounds = [(1e-7, 1.e+1), (1e-7, 1.e+1), (1e-7, 1.e+1)]
para_nonlinear_1 = gp_minimize(evaluate_model_1, bounds, n_calls=250)
@use_named_args([Real(1e-7, 1e+1, name='theta_1'),
        Real(1e-7, 1e+1, name='theta_2'),
        Real(1e-7, 1e+1, name='sigma')]) 
def evaluate_model_1(**params):
    gp = GP(kernel=rbf,kernel_diff=rbf_pd)
    theta=[params["theta_1"],params["theta_2"]] 
    y_pred = gp.loglikelihood(
                  x_star=np.linspace(0,10,1000),  # set to test points
                  X = np.array(T),     # set to observed x
                  y = np.array(Y[:,1]),       # set to observed y
                  size=1,    # draw 100 posterior samples 
                  theta=theta,
                  sigma=params["sigma"]
                )
    negative_logli= y_pred
    return -negative_logli
bounds = [(1e-7, 1.e+1), (1e-7, 1.e+1), (1e-7, 1.e+1)]
para_nonlinear_2 = gp_minimize(evaluate_model_1, bounds, n_calls=250)

# finds the hyperparameters for lorenz
T, Y = create_data_lorenz(p=0.0)
@use_named_args([Real(1e-7, 1e+1, name='theta_1'),
        Real(1e-7, 1e+1, name='theta_2'),
        Real(1e-7, 1e+1, name='sigma')])    
def evaluate_model_1(**params):
    gp = GP(kernel=rbf,kernel_diff=rbf_pd)
    theta=[params["theta_1"],params["theta_2"]] 
    y_pred = gp.loglikelihood(
                  x_star=np.linspace(0,10,1000),  # set to test points
                  X = np.array(T),     # set to observed x
                  y = np.array(Y[:,0]),       # set to observed y
                  size=1,    # draw 100 posterior samples 
                  theta=theta,
                  sigma=params["sigma"]
                )
    negative_logli= y_pred
    return -negative_logli
# Bayesian Optimisation
bounds = [(1e-7, 1.e+1), (1e-7, 1.e+1), (1e-7, 1.e+1)]
para_lorenz_1 = gp_minimize(evaluate_model_1, bounds, n_calls=250)
@use_named_args([Real(1e-7, 1e+1, name='theta_1'),
        Real(1e-7, 1e+1, name='theta_2'),
        Real(1e-7, 1e+1, name='sigma')]) 
def evaluate_model_1(**params):
    gp = GP(kernel=rbf,kernel_diff=rbf_pd)
    theta=[params["theta_1"],params["theta_2"]] 
    y_pred = gp.loglikelihood(
                  x_star=np.linspace(0,10,1000),  # set to test points
                  X = np.array(T),     # set to observed x
                  y = np.array(Y[:,1]),       # set to observed y
                  size=1,    # draw 100 posterior samples 
                  theta=theta,
                  sigma=params["sigma"]
                )
    negative_logli= y_pred
    return -negative_logli
bounds = [(1e-7, 1.e+1), (1e-7, 1.e+1), (1e-7, 1.e+1)]
para_lorenz_2 = gp_minimize(evaluate_model_1, bounds, n_calls=250)
@use_named_args([Real(1e-7, 1e+1, name='theta_1'),
        Real(1e-7, 1e+1, name='theta_2'),
        Real(1e-7, 1e+1, name='sigma')]) 
def evaluate_model_1(**params):
    gp = GP(kernel=rbf,kernel_diff=rbf_pd)
    theta=[params["theta_1"],params["theta_2"]] 
    y_pred = gp.loglikelihood(
                  x_star=np.linspace(0,10,1000),  # set to test points
                  X = np.array(T),     # set to observed x
                  y = np.array(Y[:,2]),       # set to observed y
                  size=1,    # draw 100 posterior samples 
                  theta=theta,
                  sigma=params["sigma"]
                )
    negative_logli= y_pred
    return -negative_logli
bounds = [(1e-7, 1.e+1), (1e-7, 1.e+1), (1e-7, 1.e+1)]
para_lorenz_3 = gp_minimize(evaluate_model_1, bounds, n_calls=250)

print("$"*25)
print("for the discrete spike and slab prior")
print("$"*25)
T, Y = create_data_twocompart(p=0)

toy_xp = np.linspace(0., 10., 1000)
gp = GP_derivative(kernel=rbf, kernel_diff=rbf_fd)
result_1 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_two_compart_1.x[0],para_two_compart_1.x[1]],
  sigma=para_two_compart_1.x[2]
)
toy_xp = np.linspace(0., 10., 1000)
gp = GP_derivative(kernel=rbf, kernel_diff=rbf_fd)
result_2 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_two_compart_2.x[0],para_two_compart_2.x[1]],
  sigma=para_two_compart_2.x[2]
  )

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_Two_compart( result_1, result_2, num_samples = 1000, Y = Y)

start_1,trace_1 = Bayesian_regression_disc_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_disc_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['pn_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['pn_1'])


T, Y = create_data_nonlinear(p=0)

result_1 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_nonlinear_1.x[0],para_nonlinear_1.x[1]],
  sigma=para_nonlinear_1.x[2]
)
result_2 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_nonlinear_2.x[0],para_nonlinear_2.x[1]],
  sigma=para_nonlinear_2.x[2]
)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_NonLinear( result_1, result_2, num_samples = 1000, Y = Y)

start_1,trace_1 = Bayesian_regression_disc_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_disc_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['pn_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['pn_1'])


T, Y = create_data_lorenz(p=0)

result_1 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_lorenz_1.x[0],para_lorenz_1.x[1]],
  sigma=para_lorenz_1.x[2]
)
result_2 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_lorenz_2.x[0],para_lorenz_2.x[1]],
  sigma=para_lorenz_2.x[2]
)
result_3 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_lorenz_3.x[0],para_lorenz_3.x[1]],
  sigma=para_lorenz_3.x[2]
)

x_1_train, y_1_train, x_2_train, y_2_train, x_3_train, y_3_train = obtain_train_data_Lorenz( result_1, result_2, result_3, num_samples = 1000, Y = Y)

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
T, Y = create_data_twocompart(p=0)

toy_xp = np.linspace(0., 10., 1000)
gp = GP_derivative(kernel=rbf, kernel_diff=rbf_fd)
result_1 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_two_compart_1.x[0],para_two_compart_1.x[1]],
  sigma=para_two_compart_1.x[2]
)
toy_xp = np.linspace(0., 10., 1000)
gp = GP_derivative(kernel=rbf, kernel_diff=rbf_fd)
result_2 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_two_compart_2.x[0],para_two_compart_2.x[1]],
  sigma=para_two_compart_2.x[2]
)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_Two_compart( result_1, result_2, num_samples = 1000, Y = Y)

start_1,trace_1 = Bayesian_regression_SS_Selction(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_SS_Selction(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['p_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['pn_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['p_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['pn_1'])

T, Y = create_data_nonlinear(p=0)

result_1 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_nonlinear_1.x[0],para_nonlinear_1.x[1]],
  sigma=para_nonlinear_1.x[2]
)
result_2 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_nonlinear_2.x[0],para_nonlinear_2.x[1]],
  sigma=para_nonlinear_2.x[2]
)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_NonLinear( result_1, result_2, num_samples = 1000, Y = Y)

start_1,trace_1 = Bayesian_regression_SS_Selction(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_SS_Selction(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['p_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['pn_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['p_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['pn_1'])

T, Y = create_data_lorenz(p=0)

result_1 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_lorenz_1.x[0],para_lorenz_1.x[1]],
  sigma=para_lorenz_1.x[2]
)
result_2 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_lorenz_2.x[0],para_lorenz_2.x[1]],
  sigma=para_lorenz_2.x[2]
)
result_3 = gp.predict(
  x_star=toy_xp,  # set to test points
  X = np.array(T),     # set to observed x
  y = np.array(Y[:,0]),       # set to observed y
  size=1,    # draw 100 posterior samples 
  theta=[para_lorenz_3.x[0],para_lorenz_3.x[1]],
  sigma=para_lorenz_3.x[2]
)

x_1_train, y_1_train, x_2_train, y_2_train, x_3_train, y_3_train = obtain_train_data_Lorenz( result_1, result_2, result_3, num_samples = 1000, Y = Y)

start_1,trace_1 = Bayesian_regression_SS_Selction(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_SS_Selction(y_2_train,x_2_train,np.shape(x_1_train[0])[0])
start_3,trace_3 = Bayesian_regression_SS_Selction(y_3_train,x_3_train,np.shape(x_1_train[0])[0])
print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['p_1'])
print("the value of pn_1 in model_1 of lorenz model is",start_1['pn_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['p_1'])
print("the value of pn_1 in model_2 of lorenz model is",start_2['pn_1'])
print("the value of z_1 in model_3 of lorenz model is",start_3['z_1'])
print("the value of pn_1 in model_3 of lorenz model is",start_3['p_1'])
print("the value of pn_1 in model_3 of lorenz model is",start_3['pn_1'])
