import derivative
from scipy.integrate import odeint
import numpy as np
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
def Bayesian_regression_SS_Selction(Y_1, X_1, size_fun_lib, further_prior=True):
    basic_model = pm.Model()
    Y1 = np.array(Y_1)
    X_1 = np.array(X_1)
    with basic_model:
        p_1 = pm.Uniform('p_1', 0., 1., shape = size_fun_lib)
        sigma = pm.Gamma('sigma',1.,0.1,shape=1)
        z_1  = pm.Laplace('z_1', mu=0, b=1, shape=size_fun_lib)
        pn_1 = pm.Bernoulli('pn_1', p_1, shape=size_fun_lib)
        beta_1 = pm.Deterministic('beta_1', z_1 * pn_1)
        mu_1 = pm.Deterministic(name="mu_1", var = pm.math.matrix_dot(X_1,beta_1))
        Y_obs_1 = pm.Normal('Y_obs_1', mu=mu_1, sigma= sigma, observed = Y1)
    with basic_model:   
        start = pm.find_MAP()
        trace_rh = pm.sample(4000, tune=1000, cores=1, random_seed=1, nuts={'target_accept':0.9}, init="adapt_diag")
    return start, trace_rh