import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from algorithms.general_method import GeneralMethod
from simulators.ode_simulator import calculate_fitness


class DE(GeneralMethod):
    def __init__(self, config_method, config_model):
        super().__init__(config_method, config_model)
        print('Initialized Differential Evolution (DE).')

    def proposal(self, theta, E=None):
        indices_1 = np.random.permutation(theta.shape[0])
        indices_2 = np.random.permutation(theta.shape[0])
        theta_1 = theta[indices_1]
        theta_2 = theta[indices_2]

        de_noise = self.config_method['gamma'] * (theta_1 - theta_2)

        if self.config_method['best']:
            tht = theta[[np.argmin(E)]]
        else:
            tht = theta

        theta_new = tht + de_noise

        p_1 = np.random.binomial(1, self.config_method['CR'], tht.shape)
        return p_1 * theta_new + (1. - p_1) * tht

    def step(self, theta, E_old, x_obs, mod, params):
        # (1. Generate)
        theta_new = self.proposal(theta, E_old)
        theta_new = np.clip(theta_new, a_min=self.config_method['clip_min'], a_max=self.config_method['clip_max'])

        # (2. Evaluate)
        E_new = calculate_fitness(x_obs, theta_new, mod, params, self.dist, self.config_model, self.config_method)

        # (3. Select)
        theta_cat = np.concatenate((theta, theta_new), 0)
        E_cat = np.concatenate((E_old, E_new), 0)

        indx = np.argsort(E_cat.squeeze())

        return theta_cat[indx[:theta.shape[0]],:], E_cat[indx[:theta.shape[0]],:]


class RevDE(GeneralMethod):
    def __init__(self, config_method, config_model):
        super().__init__(config_method, config_model)
        print('Initialized RevDE.')

        R = np.asarray([[1, self.config_method['gamma'], -self.config_method['gamma']],
                        [-self.config_method['gamma'], 1. - self.config_method['gamma'] ** 2, self.config_method['gamma'] + self.config_method['gamma'] ** 2],
                        [self.config_method['gamma'] + self.config_method['gamma'] ** 2, -self.config_method['gamma'] + self.config_method['gamma'] ** 2 + self.config_method['gamma'] ** 3, 1. - 2. * self.config_method['gamma'] ** 2 - self.config_method['gamma'] ** 3]])

        self.R = np.expand_dims(R, 0) # 1 x 3 x 3

    def proposal(self, theta, E=None):

        theta_0 = np.expand_dims(theta, 1) # B x 1 x D

        indices_1 = np.random.permutation(theta.shape[0])
        indices_2 = np.random.permutation(theta.shape[0])
        theta_1 = np.expand_dims(theta[indices_1], 1)
        theta_2 = np.expand_dims(theta[indices_2], 1)

        tht = np.concatenate((theta_0, theta_1, theta_2), 1) # B x 3 x D

        y = np.matmul(self.R, tht)

        theta_new = np.concatenate((y[:,0], y[:,1], y[:,2]), 0)

        p_1 = np.random.binomial(1, self.config_method['CR'], theta_new.shape)
        return p_1 * theta_new + (1. - p_1) * np.concatenate((tht[:,0], tht[:,1], tht[:,2]), 0)

    def step(self, theta, E_old, x_obs, mod, params):
        # (1. Generate)
        theta_new = self.proposal(theta, E_old)
        theta_new = np.clip(theta_new, a_min=self.config_method['clip_min'], a_max=self.config_method['clip_max'])

        # (2. Evaluate)
        E_new = calculate_fitness(x_obs, theta_new, mod, params, self.dist, self.config_model, self.config_method)

        # (3. Select)
        theta_cat = np.concatenate((theta, theta_new), 0)
        E_cat = np.concatenate((E_old, E_new), 0)

        indx = np.argsort(E_cat.squeeze())

        return theta_cat[indx[:theta.shape[0]],:], E_cat[indx[:theta.shape[0]],:]


class RevDEknn(GeneralMethod):
    def __init__(self, config_method, config_model):
        super().__init__(config_method, config_model)
        print('Initialized RevDE+knn.')

        R = np.asarray([[1, self.config_method['gamma'], -self.config_method['gamma']],
                        [-self.config_method['gamma'], 1. - self.config_method['gamma'] ** 2,
                         self.config_method['gamma'] + self.config_method['gamma'] ** 2],
                        [self.config_method['gamma'] + self.config_method['gamma'] ** 2,
                         -self.config_method['gamma'] + self.config_method['gamma'] ** 2 + self.config_method['gamma'] ** 3,
                         1. - 2. * self.config_method['gamma'] ** 2 - self.config_method['gamma'] ** 3]])

        self.R = np.expand_dims(R, 0)  # 1 x 3 x 3

        self.nn = KNeighborsRegressor(n_neighbors=3)

        self.X = None
        self.E = None

    def proposal(self, theta, E=None):

        if self.X is None:
            self.X = theta
            self.E = E
        else:
            if self.X.shape[0] < 10000:
                self.X = np.concatenate((self.X, theta), 0)
                self.E = np.concatenate((self.E, E), 0)

        self.nn.fit(self.X, self.E)

        theta_0 = np.expand_dims(theta, 1) # B x 1 x D

        indices_1 = np.random.permutation(theta.shape[0])
        indices_2 = np.random.permutation(theta.shape[0])
        theta_1 = np.expand_dims(theta[indices_1], 1)
        theta_2 = np.expand_dims(theta[indices_2], 1)

        tht = np.concatenate((theta_0, theta_1, theta_2), 1) # B x 3 x D

        y = np.matmul(self.R, tht)

        theta_new = np.concatenate((y[:,0], y[:,1], y[:,2]), 0)

        p_1 = np.random.binomial(1, self.config_method['CR'], theta_new.shape)
        theta_new = p_1 * theta_new + (1. - p_1) * np.concatenate((tht[:,0], tht[:,1], tht[:,2]), 0)

        E_pred = self.nn.predict((theta_new))

        ind = np.argsort(E_pred.squeeze())

        return theta_new[ind[:theta.shape[0]]]

    def step(self, theta, E_old, x_obs, mod, params):
        # (1. Generate)
        theta_new = self.proposal(theta, E_old)
        theta_new = np.clip(theta_new, a_min=self.config_method['clip_min'], a_max=self.config_method['clip_max'])

        # (2. Evaluate)
        E_new = calculate_fitness(x_obs, theta_new, mod, params, self.dist, self.config_model, self.config_method)

        # (3. Select)
        theta_cat = np.concatenate((theta, theta_new), 0)
        E_cat = np.concatenate((E_old, E_new), 0)

        indx = np.argsort(E_cat.squeeze())

        return theta_cat[indx[:theta.shape[0]],:], E_cat[indx[:theta.shape[0]],:]


class ES(GeneralMethod):
    def __init__(self, config_method, config_model):
        super().__init__(config_method, config_model)
        print('Initialized ES.')

        self.sigma = config_method['std']
        self.c = 0.817

    def proposal(self, theta, E=None):
        noise = self.sigma * np.random.randn(theta.shape[0], theta.shape[1])

        if self.config_method['best']:
            tht = theta[[np.argmin(E)]]
        else:
            tht = theta

        theta_new = tht + noise

        p_1 = np.random.binomial(1, self.config_method['CR'], tht.shape)
        return p_1 * theta_new + (1. - p_1) * tht

    def step(self, theta, E_old, x_obs, mod, params):
        # (1. Generate)
        theta_new = self.proposal(theta, E_old)
        theta_new = np.clip(theta_new, a_min=self.config_method['clip_min'], a_max=self.config_method['clip_max'])

        # (2. Evaluate)
        E_new = calculate_fitness(x_obs, theta_new, mod, params, self.dist, self.config_model, self.config_method)

        # (3. Select)
        m = (E_new < E_old) * 1.

        if np.mean(m) < 0.2:
            self.sigma = self.sigma * self.c
        elif np.mean(m) > 0.2:
            self.sigma = self.sigma / self.c

        return m * theta_new + (1. - m) * theta, m * E_new + (1. - m) * E_old


class EDA(GeneralMethod):
    def __init__(self, config_method, config_model):
        super().__init__(config_method, config_model)
        print('Initialized EDA.')

    def estimate(self, x):
        m = np.mean(x, 0, keepdims=True)
        z = np.expand_dims(x - m, 2)

        S = np.mean(np.matmul(z, np.transpose(z, [0, 2, 1])), 0)

        L = np.linalg.cholesky(S)

        return m, L

    def proposal(self, theta, E=None):
        # Fit Gaussian:
        # 1) Calculate mean and covariance matrix
        m, L = self.estimate(theta)

        # 2) Generate new points (2 x more than original theta):
        # x_new = mean + epsilon * L
        theta_new = m + np.dot( np.random.randn(theta.shape[0], theta.shape[1]), L )

        return theta_new

    def step(self, theta, E_old, x_obs, mod, params):
        # (1. Generate)
        theta_new = self.proposal(theta, E_old)
        theta_new = np.clip(theta_new, a_min=self.config_method['clip_min'], a_max=self.config_method['clip_max'])

        # (2. Evaluate)
        E_new = calculate_fitness(x_obs, theta_new, mod, params, self.dist, self.config_model, self.config_method)

        # (3. Select)
        theta_cat = np.concatenate((theta, theta_new), 0)
        E_cat = np.concatenate((E_old, E_new), 0)

        indx = np.argsort(E_cat.squeeze())

        return theta_cat[indx[:theta.shape[0]], :], E_cat[indx[:theta.shape[0]], :]


class EDAknn(GeneralMethod):
    def __init__(self, config_method, config_model):
        super().__init__(config_method, config_model)
        print('Initialized EDA+knn.')

        self.nn = KNeighborsRegressor(n_neighbors=3)

        self.X = None
        self.E = None

    def estimate(self, x):
        m = np.mean(x, 0, keepdims=True)
        z = np.expand_dims(x - m, 2)

        S = np.mean(np.matmul(z, np.transpose(z, [0, 2, 1])), 0)

        L = np.linalg.cholesky(S)

        return m, L

    def proposal(self, theta, E=None):
        if self.X is None:
            self.X = theta
            self.E = E
        else:
            if self.X.shape[0] < 10000:
                self.X = np.concatenate((self.X, theta), 0)
                self.E = np.concatenate((self.E, E), 0)

        self.nn.fit(self.X, self.E)

        # Fit Gaussian:
        # 1) Calculate mean and covariance matrix
        m, L = self.estimate(theta)

        # 2) Generate new points (2 x more than original theta):
        # x_new = mean + epsilon * L
        theta_new = m + np.dot( np.random.randn(theta.shape[0] * 5, theta.shape[1]), L )

        E_pred = self.nn.predict((theta_new))

        ind = np.argsort(E_pred.squeeze())

        return theta_new[ind[:theta.shape[0]]]

    def step(self, theta, E_old, x_obs, mod, params):
        # (1. Generate)
        theta_new = self.proposal(theta, E_old)
        theta_new = np.clip(theta_new, a_min=self.config_method['clip_min'], a_max=self.config_method['clip_max'])

        # (2. Evaluate)
        E_new = calculate_fitness(x_obs, theta_new, mod, params, self.dist, self.config_model, self.config_method)

        # (3. Select)
        theta_cat = np.concatenate((theta, theta_new), 0)
        E_cat = np.concatenate((E_old, E_new), 0)

        indx = np.argsort(E_cat.squeeze())

        return theta_cat[indx[:theta.shape[0]], :], E_cat[indx[:theta.shape[0]], :]