import numpy as np
from scipy.integrate import odeint

class Sir_object(object):

    def __init__(self, data_train):
        self.gamma = 1 / 18
        self.sigma = 1 / 5.2


        i_0 = 1.15e-6
        e_0 = 4.3 * i_0
        s_0 = 1 - 1*i_0 - e_0


        self.x_0 = s_0, e_0, i_0
        self.data_train = data_train



    def F_system_eq(self, x, t, R0=1.6):
        """
        Time derivative of the state vector.
            * x is the state vector (array_like)
            * t is time (scalar)
            * R0 is the effective transmission rate, defaulting to a constant
        """
        s, e, i = x

        # New exposure of susceptibles
        beta = R0(t) * self.gamma if callable(R0) else R0 * self.gamma
        ne = beta * s * i

        # Time derivatives
        ds = - ne
        de = ne - self.sigma * e
        di = self.sigma * e - self.gamma * i

        return ds, de, di



    def solve_path(self, R0, t_vec):
        """
        Solve for i(t) and c(t) via numerical integration,
        given the time path for R0.


        """
        G = lambda x, t: self.F_system_eq(x, t, R0)
        s_path, e_path, i_path = odeint(G, self.x_0, t_vec).transpose()

        c_path = 1 - s_path - e_path       # cumulative cases
        c_path = c_path *0.828 * np.max(self.data_train.values.squeeze())/ np.max(c_path)
        return c_path

