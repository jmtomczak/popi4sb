import os
import json
import pickle
import time
import pysces

import numpy as np
from inspect import isclass

from utils.general import dict_to_array
from utils.pysces_utils import read_real_parameters, generate_data, remove_fixed

from simulators.ode_simulator import calculate_fitness
import algorithms.population_optimization_algorithms as EA
from utils.config import Config


def run(dir_method, json_method,
        dir_model, json_model,
        dir_results,
        dir_solver, json_solver,
        dir_data=None, file_data=None,
        exp_sign='_exp_'):

    config_method = Config(os.path.join(dir_method, json_method))
    config_model = Config(os.path.join(dir_model, json_model))

    config_solver = Config(os.path.join(dir_solver, json_solver))

    # Experiment name
    exp_name = exp_sign + config_method.config['method_name'] + '_'

    # Load PySCES model
    mod = pysces.model(config_model.config['mod_name'], dir=dir_model)

    # Solver settings
    mod.__settings__["mode_sim_max_iter"] = config_solver.config["mode_sim_max_iter"]
    mod.__settings__['lsoda_atol'] = config_solver.config['lsoda_atol']
    mod.__settings__['lsoda_rtol'] = config_solver.config['lsoda_rtol']
    mod.__settings__['lsoda_mxordn'] = config_solver.config['lsoda_mxordn']
    mod.__settings__['lsoda_mxords'] = config_solver.config['lsoda_mxords']
    mod.__settings__['lsoda_mxstep'] = config_solver.config['lsoda_mxstep']

    # =====REAL DATA PREPARATION=====
    # Remove fixed_species from params. We do it only once.
    params = remove_fixed(mod.parameters, mod.fixed_species, compartment=config_model.config['compartment'])

    if dir_data is not None:
        config_model.config['real_data'] = True
        mod.sim_start = config_model.config['sim_start']
        mod.sim_end = config_model.config['sim_end']
        mod.sim_points = config_model.config['sim_points']
        x_obs = np.load(os.path.join(dir_data, file_data))
    else:
        config_model.config['real_data'] = False
        x_obs, t = generate_data(mod, params, sim_start=config_model.config['sim_start'], sim_end=config_model.config['sim_end'],
                                 sim_points=config_model.config['sim_points'], noise=config_model.config['noise'])

        real_params = read_real_parameters(mod, params)
        real_params_array = dict_to_array(real_params, params)

        np.save(os.path.join(dir_results, exp_name + 'x_obs.npy'), x_obs)
        np.save(os.path.join(dir_results, exp_name + 't.npy'), t)
        np.save(os.path.join(dir_results, exp_name + 'real_params_array.npy'), real_params_array)

        json.dump(real_params, open(os.path.join(dir_results, exp_name + 'real_params.json'), "w"))
        json.dump(params, open(os.path.join(dir_results, exp_name + 'params.json'), "w"))

    pickle.dump(mod, open(os.path.join(dir_results, exp_name + 'mod.pkl'), "wb"))

    # =======EXPERIMENT=======
    # dump, just in case, configs
    pickle.dump(config_method.config, open(os.path.join(dir_results, exp_name + 'config_method.pkl'), "wb"))
    pickle.dump(config_model.config, open(os.path.join(dir_results, exp_name + 'config_model.pkl'), "wb"))

    # Init method
    # -get all classes in the file
    classes = [x for x in dir(EA) if isclass(getattr(EA, x))]
    # -check whether the provided name is available
    assert config_method.config['method_name'] in classes, 'Wrong name of the method! Please pick one of the following methods: {}'.format(classes)

    # -initialize the appropriate class
    module = __import__("algorithms.population_optimization_algorithms", fromlist=[config_method.config['method_name']])
    my_class = getattr(module, config_method.config['method_name'])
    method = my_class(config_method.config, config_model.config)

    # Init parameters
    theta = np.random.uniform(low=config_model.config['low'], high=config_model.config['high'], size=(config_method.config['pop_size'], len(params)))
    theta = np.clip(theta, a_min=config_method.config['clip_min'], a_max=config_method.config['clip_max'])
    # Calcute their energy
    E = calculate_fitness(x_obs, theta, mod, params, dist=method.dist, config_model=config_model.config, config_method=config_method.config)

    # -=Start experiment=-
    best_E = [np.min(E)]

    all_E = E
    all_theta = theta

    clock_start = time.time()
    print('START ~~~~~~>')
    g = config_method.config['generations']
    for i in range(g):
        print(f'========> Generation {i+1}/{g}')
        theta, E = method.step(theta, E, x_obs, mod, params)
        if np.min(E) < best_E[-1]:
            best_E.append(np.min(E))
        else:
            best_E.append(best_E[-1])

        all_theta = np.concatenate((all_theta, theta), 0)
        all_E = np.concatenate((all_E, E), 0)
        # SAVING
        np.save(os.path.join(dir_results, exp_name + 'all_theta.npy'), all_theta)
        np.save(os.path.join(dir_results, exp_name + 'all_E.npy'), all_E)
        np.save(os.path.join(dir_results, exp_name + 'best_E.npy'), np.asarray(best_E))

        # early stopping
        if i > config_method.config['patience']:
            if best_E[-config_method.config['patience']] == best_E[-1]:
                break
    print('~~~~~~> END')
    clock_stop = time.time()
    print('Time elapsed: {}'.format(clock_stop - clock_start))
    np.save(os.path.join(dir_results, exp_name + 'time.npy'), np.asarray(clock_stop - clock_start))
