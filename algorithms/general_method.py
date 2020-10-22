import scipy.stats as stats


class GeneralMethod(object):
    def __init__(self, config_method, config_model):
        self.config_method = config_method
        self.config_model = config_model

        self.indices = self.config_model['indices']

        self.dist_name = self.config_method['dist_name']
        self.dist = stats.norm

    def proposal(self, theta, E=None):
        pass

    def step(self, theta, E_old, x_obs, mod, params):
        pass