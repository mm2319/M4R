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
from lorenz import create_data_lorenz
from non_linear import create_data_nonlinear
from two_compartment import create_data_twocompart
from Derivative_Data_Lorenz import obtain_train_data_Lorenz
from Derivative_Data_NonLinear import obtain_train_data_NonLinear
from Derivative_Data_Two_Compart import obtain_train_data_Two_compart
from Bayesian_Regression_Disc_Spike_and_Slab import Bayesian_regression_disc_spike_slab
from Bayesian_Regression_Cont_Spike_and_Slab import Bayesian_regression_conti_spike_slab
from Bayesian_Regression_SS_Selection_2 import Bayesian_regression_SS_Selction
network_twocompart = FeedForwardNetwork(2)
T, Y = create_data_twocompart(p=0.)
train_twocompart(network_twocompart,  6000,T=T,Y=Y)
network_twocompart.save_weights('./NN_twocompart_checkpoints_000/my_checkpoint', save_format="tf")

