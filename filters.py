"""Decoupled (EKF+Bayes) and Coupled (PF) filters."""
import numpy as np
from pomdp import (transition, transition_jacobian, visual_obs, wrap_angle,
                   contact_likelihood, contact_likelihood_decoupled,
                   POMDPParams, Action)


class DecoupledFilter:
    def __init__(self, params):
        self.params = params
        self.x_hat = params.init_mean.copy()
        self.P = params.init_cov.copy()
        self.belief_theta = np.array([1. - params.prior_compatible,
                                       params.prior_compatible])

    def predict(self, action):
        F = transition_jacobian(self.x_hat, action, self.params)
        self.x_hat = transition(self.x_hat, action, self.params, noise=np.zeros(3))
        self.P = F @ self.P @ F.T + self.params.Q

    def update_visual(self, z_v):
        H = visual_obs()
        R = self.params.R_visual
        y = z_v - H @ self.x_hat
        # z_v is (x, phi): wrap the phi component (index 1)
        y[-1] = wrap_angle(y[-1])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ y
        self.x_hat[2] = wrap_angle(self.x_hat[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def update_contact(self, z_c):
        for k in range(self.params.K):
            self.belief_theta[k] *= contact_likelihood_decoupled(
                z_c, k, self.x_hat, self.params)
        s = self.belief_theta.sum()
        if s > 0:
            self.belief_theta /= s
        else:
            self.belief_theta[:] = 1. / self.params.K

    def step(self, action, z_v, z_c=None):
        self.predict(action)
        self.update_visual(z_v)
        if z_c is not None:
            self.update_contact(z_c)

    @property
    def theta_estimate(self):
        return int(np.argmax(self.belief_theta))

    @property
    def theta_confidence(self):
        return float(np.max(self.belief_theta))

    @property
    def x_estimate(self):
        return self.x_hat.copy()

    @property
    def x_covariance(self):
        return self.P.copy()


class CoupledFilter:
    def __init__(self, params, n_particles=1000):
        self.params = params
        self.N = n_particles
        self.thetas = np.random.choice(
            params.K, size=n_particles,
            p=[1. - params.prior_compatible, params.prior_compatible])
        self.xs = np.random.multivariate_normal(
            params.init_mean, params.init_cov, size=n_particles)
        self.xs[:, 2] = wrap_angle(self.xs[:, 2])
        self.weights = np.ones(n_particles) / n_particles

    def predict(self, action):
        noise = np.random.multivariate_normal(
            np.zeros(3), self.params.Q, size=self.N)
        d = self.params.nudge_delta
        phi = self.xs[:, 2]
        if action == Action.NUDGE_X_POS:
            self.xs[:, 0] += d*np.cos(phi)
            self.xs[:, 1] += d*np.sin(phi)
        elif action == Action.NUDGE_X_NEG:
            self.xs[:, 0] -= d*np.cos(phi)
            self.xs[:, 1] -= d*np.sin(phi)
        elif action == Action.NUDGE_Y_POS:
            self.xs[:, 0] -= d*np.sin(phi)
            self.xs[:, 1] += d*np.cos(phi)
        elif action == Action.NUDGE_Y_NEG:
            self.xs[:, 0] += d*np.sin(phi)
            self.xs[:, 1] -= d*np.cos(phi)
        elif action == Action.ROTATE_POS:
            self.xs[:, 2] += self.params.rotate_delta
        elif action == Action.ROTATE_NEG:
            self.xs[:, 2] -= self.params.rotate_delta
        self.xs += noise
        self.xs[:, 2] = wrap_angle(self.xs[:, 2])

    def update_visual(self, z_v):
        H = visual_obs()
        z_pred = (H @ self.xs.T).T  # (N, dim_obs)
        diffs = z_v - z_pred
        # z_v is (x, phi): wrap the phi component (last index)
        diffs[:, -1] = wrap_angle(diffs[:, -1])
        Ri = np.linalg.inv(self.params.R_visual)
        dR = np.linalg.det(self.params.R_visual)
        dim = H.shape[0]
        nc = 1. / np.sqrt((2*np.pi)**dim * dR)
        self.weights *= nc * np.exp(
            -0.5 * np.sum(diffs @ Ri * diffs, axis=1))
        self._norm()

    def update_contact(self, z_c):
        pe = np.sqrt(self.xs[:, 0]**2 + self.xs[:, 1]**2)
        ae = np.abs(wrap_angle(self.xs[:, 2]))
        c = pe + self.params.contact_angle_weight * ae
        a = self.params.contact_alpha
        r = self.params.contact_radius
        sig = 1. / (1. + np.exp(np.clip(-a*(r - c), -500, 500)))
        ps = np.where(self.thetas == 1, sig, self.params.contact_epsilon)
        self.weights *= ps if z_c else (1. - ps)
        self._norm()

    def step(self, action, z_v, z_c=None):
        self.predict(action)
        self.update_visual(z_v)
        if z_c is not None:
            self.update_contact(z_c)
        neff = 1. / np.sum(self.weights**2)
        if neff < self.N / 2:
            self._resample()

    def _norm(self):
        s = self.weights.sum()
        if s > 0:
            self.weights /= s
        else:
            self.weights[:] = 1. / self.N

    def _resample(self):
        N = self.N
        cs = np.cumsum(self.weights)
        pos = (np.random.random() + np.arange(N)) / N
        idx = np.clip(np.searchsorted(cs, pos), 0, N - 1)
        self.thetas = self.thetas[idx]
        self.xs = self.xs[idx].copy()
        self.weights[:] = 1. / N
        # Roughening
        self.xs += np.random.randn(N, 3) * np.sqrt(np.diag(self.params.Q))
        self.xs[:, 2] = wrap_angle(self.xs[:, 2])
        # Tiny theta diversity
        fl = np.random.random(N) < 0.005
        self.thetas[fl] = 1 - self.thetas[fl]


    @property
    def belief_theta(self):
        return np.bincount(self.thetas, weights=self.weights,
                           minlength=self.params.K)

    @property
    def theta_estimate(self):
        return int(np.argmax(self.belief_theta))

    @property
    def theta_confidence(self):
        return float(np.max(self.belief_theta))

    @property
    def x_estimate(self):
        m = np.average(self.xs, weights=self.weights, axis=0)
        m[2] = wrap_angle(m[2])
        return m

    @property
    def x_covariance(self):
        m = self.x_estimate
        d = self.xs - m
        d[:, 2] = wrap_angle(d[:, 2])
        return (d * self.weights[:, None]).T @ d
