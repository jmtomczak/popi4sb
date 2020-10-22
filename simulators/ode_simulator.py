import numpy as np
from utils.pysces_utils import overwrite_params


# SIMULATOR
def f(theta, mod, params):
    # overwrite parameters values of the model
    mod = overwrite_params(theta, mod, params)
    # run simulation
    mod.Simulate()

    # get synthetic data
    sim_data = mod.data_sim.getSpecies()

    return sim_data[:, 1:].T


# Fitness function defined as E(x) = log p(x)
def fitness(x, theta, mod, params, dist, config_model, config_method):
    S_sim = f(theta, mod, params)
    if config_model['real_data']:
        S_obs = x / np.expand_dims(np.max(x, 1), 1)
        S_sim = S_sim[config_model['indices']]/ np.expand_dims(np.max(x, 1), 1)
    else:
        S_obs = x[config_model['indices']] / np.expand_dims(np.max(x[config_model['indices']], 1), 1)
        S_sim = S_sim[config_model['indices']] / np.expand_dims(np.max(x[config_model['indices']], 1), 1)

    if np.isnan(S_sim).any():
        fitness = np.asarray([[10000.]])
    else:
        if config_method['dist_name'] == 'norm':
            difference = dist.logpdf(S_obs, loc=S_sim, scale=config_method['scale'])
        elif config_method['dist_name'] == 'abs':
            difference = -np.abs(S_obs - S_sim)
        else:
            raise ValueError('Wrong distribution name!')

        log_pdf = np.sum(difference, 1, keepdims=True)
        fitness = -np.sum(log_pdf, 0, keepdims=True)

    return fitness


# Calculating fitness for a batch
def calculate_fitness(x, theta, mod, params, dist, config_model, config_method):
    E_array = np.zeros((theta.shape[0], 1))

    for i in range(theta.shape[0]):
        E_array[i] = fitness(x, theta[i], mod, params, dist, config_model, config_method)

    return E_array