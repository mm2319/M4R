from Network import FeedForwardNetwork
from NN_deri_Lorenz import train_lorenz, obtain_lorenz_data
from NN_deri_two_compart import train_twocompart, obtain_twocompart_data
from NN_deri_nonlinear import train_nonlinear,obtain_nonlinear_data
from scipy.integrate import odeint
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
from Bayesian_Regression_Cont_Spike_and_Slab import Bayesian_regression_conti_spike_slab
from Bayesian_Regression_SS_Selection_2 import Bayesian_regression_SS_Selction
from Gaussian_process_der import GP, GP_derivative,rbf,rbf_fd,rbf_pd_2,rbf_pd_1,RBF_partial_diff_first,RBF_partial_diff_second
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from scipy.optimize import minimize
import autograd.numpy as np
from autograd import value_and_grad
from skopt.plots import plot_convergence
np.random.seed(0)
def optim_hyperparams(init_params, data_X, data_y, gp, method="L-BFGS-B", maxiter=500):
  """
  Find the best kernel hyper-parameters that maximise the log marginal
  likelihood.
  """
  # define negative log marginal likelihood as objective
  # input is unpacked to theta and sigma
  gp = GP(kernel=rbf,kernel_diff=rbf_fd)
  objective = lambda params: gp.loglikelihood(
                x_star=np.arange(0,10,0.01),  # set to test points
                X = data_X,     # set to observed x
                y = data_y,       # set to observed y
                size=1,    # draw 100 posterior samples 
                theta=params[:-1],
                sigma=params[-1]
              )

  optim_res = minimize(
      fun=value_and_grad(objective),
      jac=True,
      x0=init_params, 
      method=method,
      bounds = [[1e-10, 1.e3], [1e-10, 1.e3], [1e-10, 1.e3]],
      options={"return_all": True, "maxiter": maxiter}
  )
  return optim_res

def optim_hyperparams_multiple_runs(init_params_list, data_X, data_y, gp, maxiter=500):
  """
  Run hyper-parameter search with multiple starting points.
  """
  optim_res_list = []
  log_lik_history_list = []
  nrep = len(init_params_list)

  for i in range(nrep):
    init_params = init_params_list[i]

    # find best params
    optim_res = optim_hyperparams(
        init_params, data_X, data_y, gp, method="BFGS", maxiter=maxiter,
    )
    # log_lik_history = show_optim_history(optim_res.allvecs, gp, data_X, data_y)
    params = optim_res.x
    # compute log marginal lkhd
    log_lik = gp.loglikelihood(
                x_star=np.arange(0,10,0.01),  # set to test points
                X = data_X,     # set to observed x
                y = data_y,       # set to observed y
                size=1,    # draw 100 posterior samples 
                theta=params[:-1],
                sigma=params[-1]
              )

    # store results
    optim_res_list.append(optim_res)
    log_lik_history_list.append(log_lik)

  return optim_res_list, log_lik_history_list 
def grid_search_initial():
  init_params_list = []
  init_params = [1.e-3,1.e-2,1.e-1,0.1,1,10,100]
  for i in init_params:
    for j in init_params:
      for k in init_params:
        init_params_list.append([i,j,k])
  return np.array(init_params_list)
init_params_list = grid_search_initial()
gp = GP(kernel=rbf,kernel_diff=rbf_fd)
# finds the hyperparameters for two_compart
np.random.seed(0)
T, Y_tc = create_data_twocompart(p=0.0)

optim_res_list, log_lik_history_list  = optim_hyperparams_multiple_runs(
    init_params_list, T, Y_tc[:,0], gp)
best_idx = np.argmin(log_lik_history_list)
best_optim_res = optim_res_list[best_idx]
best_log_lik = log_lik_history_list[best_idx]
para_two_compart_1 = best_optim_res

optim_res_list, log_lik_history_list  = optim_hyperparams_multiple_runs(
    init_params_list, T, Y_tc[:,1], gp)
best_idx = np.argmin(log_lik_history_list)
best_optim_res = optim_res_list[best_idx]
best_log_lik = log_lik_history_list[best_idx]
para_two_compart_2 = best_optim_res

Y_compart = []
y_pred_1 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_tc[:,0]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_two_compart_1.x[0],para_two_compart_1.x[1]],
              sigma=para_two_compart_1.x[2]
              )
y_pred_2 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_tc[:,1]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_two_compart_2.x[0],para_two_compart_2.x[1]],
              sigma=para_two_compart_2.x[2]
              )
Y_compart.append(y_pred_1)
Y_compart.append(y_pred_2)
Y_compart = np.array(Y_compart).T


# finds the hyperparameters for nonlinear
T, Y_nl = create_data_nonlinear(p=0.0)

optim_res_list, log_lik_history_list  = optim_hyperparams_multiple_runs(
    init_params_list, T, Y_nl[:,0], gp)
best_idx = np.argmin(log_lik_history_list)
best_optim_res = optim_res_list[best_idx]
best_log_lik = log_lik_history_list[best_idx]
para_nonlinear_1 = best_optim_res


optim_res_list, log_lik_history_list  = optim_hyperparams_multiple_runs(
    init_params_list, T, Y_nl[:,1], gp)
best_idx = np.argmin(log_lik_history_list)
best_optim_res = optim_res_list[best_idx]
best_log_lik = log_lik_history_list[best_idx]
para_nonlinear_2 = best_optim_res

Y_nonlinear = []
y_pred_1 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_nl[:,0]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_nonlinear_1.x[0],para_nonlinear_1.x[1]],
              sigma=para_nonlinear_1.x[2]
              )
y_pred_2 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_nl[:,1]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_nonlinear_2.x[0],para_nonlinear_2.x[1]],
              sigma=para_nonlinear_2.x[2]
              )
Y_nonlinear.append(y_pred_1)
Y_nonlinear.append(y_pred_2)
Y_nonlinear = np.array(Y_nonlinear).T


# finds the hyperparameters for lorenz
T, Y_lr = create_data_lorenz(p=0.0)

optim_res_list, log_lik_history_list  = optim_hyperparams_multiple_runs(
    init_params_list, T, Y_lr[:,0], gp)
best_idx = np.argmin(log_lik_history_list)
best_optim_res = optim_res_list[best_idx]
best_log_lik = log_lik_history_list[best_idx]
para_lorenz_1 = best_optim_res

optim_res_list, log_lik_history_list  = optim_hyperparams_multiple_runs(
    init_params_list, T, Y_lr[:,1], gp)
best_idx = np.argmin(log_lik_history_list)
best_optim_res = optim_res_list[best_idx]
best_log_lik = log_lik_history_list[best_idx]
para_lorenz_2 = best_optim_res

optim_res_list, log_lik_history_list  = optim_hyperparams_multiple_runs(
    init_params_list, T, Y_lr[:,2], gp)
best_idx = np.argmin(log_lik_history_list)
best_optim_res = optim_res_list[best_idx]
best_log_lik = log_lik_history_list[best_idx]
para_lorenz_3 = best_optim_res

Y_lorenz = []
y_pred_1 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_lr[:,0]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_lorenz_1.x[0],para_lorenz_1.x[1]],
              sigma=para_lorenz_1.x[2]
              )
y_pred_2 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_lr[:,1]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_lorenz_2.x[0],para_lorenz_2.x[1]],
              sigma=para_lorenz_2.x[2]
              )
y_pred_3 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_lr[:,2]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_lorenz_3.x[0],para_lorenz_3.x[1]],
              sigma=para_lorenz_3.x[2]
              )
Y_lorenz.append(y_pred_1)
Y_lorenz.append(y_pred_2)
Y_lorenz.append(y_pred_3)
Y_lorenz= np.array(Y_lorenz).T
print("$"*25)
print("for the continuous spike and slab prior")
print("$"*25)

result_1 = derivative.dxdt(Y_compart[:,0], np.arange(0,10,0.1), kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y_compart[:,1], np.arange(0,10,0.1), kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_Two_compart( result_1, result_2, num_samples = 100, Y = Y_compart)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])
print("the value of z_1 in model_1 of two compartment model is",start_1['z_1'])
print("the value of beta_1 in model_1 of two compartment model is",start_1['beta_1'])
print("the value of z_1 in model_2 of two compartment model is",start_2['z_1'])
print("the value of beta_1 in model_2 of two compartment model is",start_2['beta_1'])
np.save('gpfd_BR_CSS_000_tc_1',start_1['beta_1'])
np.save('gpfd_BR_CSS_000_tc_2',start_2['beta_1'])

result_1 = derivative.dxdt(Y_nonlinear[:,0], np.arange(0,10,0.1), kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y_nonlinear[:,1], np.arange(0,10,0.1), kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_NonLinear( result_1, result_2, num_samples = 100, Y = Y_nonlinear)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of nonlinear compartment model is",start_1['z_1'])
print("the value of beta_1 in model_1 of nonlinear compartment model is",start_1['beta_1'])
print("the value of z_1 in model_2 of nonlinear compartment model is",start_2['z_1'])
print("the value of beta_1 in model_2 of nonlinear compartment model is",start_2['beta_1'])
np.save('gpfd_BR_CSS_000_nl_1',start_1['beta_1'])
np.save('gpfd_BR_CSS_000_nl_2',start_2['beta_1'])

result_1 = derivative.dxdt(Y_lorenz[:,0], np.arange(0,10,0.1), kind="finite_difference", k=2)
result_2 = derivative.dxdt(Y_lorenz[:,1], np.arange(0,10,0.1), kind="finite_difference", k=2)
result_3 = derivative.dxdt(Y_lorenz[:,2], np.arange(0,10,0.1), kind="finite_difference", k=2)

x_1_train, y_1_train, x_2_train, y_2_train, x_3_train, y_3_train = obtain_train_data_Lorenz( result_1, result_2, result_3, num_samples = 100, y = Y_lorenz)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])
start_3,trace_3 = Bayesian_regression_conti_spike_slab(y_3_train,x_3_train,np.shape(x_1_train[0])[0])
print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of beta_1 in model_1 of lorenz model is",start_1['beta_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of beta_1 in model_2 of lorenz model is",start_2['beta_1'])
print("the value of z_1 in model_3 of lorenz model is",start_3['z_1'])
print("the value of beta_1 in model_3 of lorenz model is",start_3['beta_1'])
np.save('gpfd_BR_CSS_000_lr_1',start_1['beta_1'])
np.save('gpfd_BR_CSS_000_lr_2',start_2['beta_1'])
np.save('gpfd_BR_CSS_000_lr_3',start_3['beta_1'])