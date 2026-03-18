"""
POMDP for partially observable assembly.

Front-facing camera observes (x, phi) with noise. Depth y is NOT observed.
    x   -- horizontal offset (visible from camera)
    phi -- rotation (visible from block face geometry)
    y   -- depth (hidden: no stereo/depth sensor)
"""
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum

class Action(IntEnum):
    NUDGE_X_POS = 0
    NUDGE_X_NEG = 1
    NUDGE_Y_POS = 2
    NUDGE_Y_NEG = 3
    ROTATE_POS = 4
    ROTATE_NEG = 5
    ATTEMPT_ATTACH = 6

@dataclass
class POMDPParams:
    K: int = 2
    prior_compatible: float = 0.5
    nudge_delta: float = 0.1
    rotate_delta: float = 0.15
    process_noise_xy: float = 0.02
    process_noise_phi: float = 0.01
    obs_noise_x: float = 0.05
    obs_noise_phi: float = 0.05
    contact_alpha: float = 12.0
    contact_radius: float = 0.35
    contact_epsilon: float = 0.05
    contact_angle_weight: float = 0.5
    init_mean: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.8, 0.5]))
    init_cov: np.ndarray = field(default_factory=lambda: np.diag([0.25, 0.25, 0.3])**2)

    @property
    def Q(self):
        return np.diag([self.process_noise_xy**2,
                        self.process_noise_xy**2,
                        self.process_noise_phi**2])

    @property
    def R_visual(self):
        """2x2: observes x and phi only."""
        return np.diag([self.obs_noise_x**2,
                        self.obs_noise_phi**2])

def wrap_angle(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi

def transition(x, action, params, noise=None):
    if noise is None:
        noise = np.random.multivariate_normal(np.zeros(3), params.Q)
    xn = x.copy()
    d, phi = params.nudge_delta, x[2]
    if action == Action.NUDGE_X_POS:
        xn[0] += d*np.cos(phi); xn[1] += d*np.sin(phi)
    elif action == Action.NUDGE_X_NEG:
        xn[0] -= d*np.cos(phi); xn[1] -= d*np.sin(phi)
    elif action == Action.NUDGE_Y_POS:
        xn[0] -= d*np.sin(phi); xn[1] += d*np.cos(phi)
    elif action == Action.NUDGE_Y_NEG:
        xn[0] += d*np.sin(phi); xn[1] -= d*np.cos(phi)
    elif action == Action.ROTATE_POS:
        xn[2] += params.rotate_delta
    elif action == Action.ROTATE_NEG:
        xn[2] -= params.rotate_delta
    xn += noise
    xn[2] = wrap_angle(xn[2])
    return xn

def transition_jacobian(x, action, params):
    F = np.eye(3)
    d, phi = params.nudge_delta, x[2]
    if action == Action.NUDGE_X_POS:
        F[0,2] = -d*np.sin(phi); F[1,2] = d*np.cos(phi)
    elif action == Action.NUDGE_X_NEG:
        F[0,2] = d*np.sin(phi); F[1,2] = -d*np.cos(phi)
    elif action == Action.NUDGE_Y_POS:
        F[0,2] = -d*np.cos(phi); F[1,2] = -d*np.sin(phi)
    elif action == Action.NUDGE_Y_NEG:
        F[0,2] = d*np.cos(phi); F[1,2] = d*np.sin(phi)
    return F

def visual_obs():
    """2x3: observes x (row 0) and phi (row 2), NOT y."""
    return np.array([[1., 0., 0.],
                     [0., 0., 1.]])

def observe_visual(x, params, noise=None):
    """Returns 2D observation: (x, phi) + noise."""
    O = visual_obs()
    if noise is None:
        noise = np.random.multivariate_normal(np.zeros(2), params.R_visual)
    o = O @ x + noise
    o[1] = wrap_angle(o[1])  # wrap the phi component
    return o

def contact_success_probability(theta, x, params):
    if theta == 1:
        pe = np.sqrt(x[0]**2 + x[1]**2)
        ae = abs(wrap_angle(x[2]))
        c = pe + params.contact_angle_weight * ae
        v = params.contact_alpha * (params.contact_radius - c)
        return float(1./(1.+np.exp(-np.clip(v, -500, 500))))
    return params.contact_epsilon

def observe_contact(theta, x, params):
    return np.random.random() < contact_success_probability(theta, x, params)

def contact_likelihood(z_c, theta, x, params):
    p = contact_success_probability(theta, x, params)
    return p if z_c else (1.-p)

def contact_likelihood_decoupled(z_c, theta, x_hat, params):
    p = contact_success_probability(theta, x_hat, params)
    return p if z_c else (1.-p)
