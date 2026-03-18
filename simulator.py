import numpy as np, time
from filters import DecoupledFilter, CoupledFilter
from dataclasses import replace
from pomdp import Action, POMDPParams, transition, observe_visual, observe_contact, wrap_angle

def sample_init(params):
    theta = int(np.random.random() < params.prior_compatible)
    x0 = np.random.multivariate_normal(params.init_mean, params.init_cov)
    x0[2] = wrap_angle(x0[2])
    return theta, x0

def policy(xh, t, params):
    if t % 5 == 4:
        return Action.ATTEMPT_ATTACH
    if abs(xh[2]) > 0.2:
        return Action.ROTATE_NEG if xh[2] > 0 else Action.ROTATE_POS
    elif abs(xh[0]) > abs(xh[1]):
        return Action.NUDGE_X_NEG if xh[0] > 0 else Action.NUDGE_X_POS
    else:
        return Action.NUDGE_Y_NEG if xh[1] > 0 else Action.NUDGE_Y_POS

def run_single(params, n_steps, n_particles=1000,
               init_mean_override=None, init_cov_override=None):

    p = params
    if init_mean_override is not None or init_cov_override is not None:
        kw = {}
        if init_mean_override is not None:
            kw["init_mean"] = init_mean_override
        if init_cov_override is not None:
            kw["init_cov"] = init_cov_override
        p = replace(params, **kw)

    tt, x0 = sample_init(p)
    dec = DecoupledFilter(p)
    coup = CoupledFilter(p, n_particles=n_particles)
    
    h = dict(
        theta_true=tt, x_true=[x0.copy()], actions=[],
        dec_x=[dec.x_estimate], dec_bt=[dec.belief_theta.copy()],
        dec_te=[dec.theta_estimate],
        dec_Ptr=[np.trace(dec.x_covariance)],
        dec_phi_std=[np.sqrt(max(dec.x_covariance[2,2], 0))],
        dec_time=[],
        coup_x=[coup.x_estimate], coup_bt=[coup.belief_theta.copy()],
        coup_te=[coup.theta_estimate],
        coup_Ptr=[np.trace(coup.x_covariance)],
        coup_phi_std=[np.sqrt(max(coup.x_covariance[2,2], 0))],
        coup_time=[],
        z_v=[], z_c=[],
    )

    x = x0.copy()
    for t in range(n_steps):
        action = policy(dec.x_estimate, t, params)
        h["actions"].append(action)
        x = transition(x, action, params)
        zv = observe_visual(x, params)
        zc = (observe_contact(tt, x, params)
              if action == Action.ATTEMPT_ATTACH else None)
        h["x_true"].append(x.copy())
        h["z_v"].append(zv)
        h["z_c"].append(zc)
        t0 = time.perf_counter()
        dec.step(action, zv, zc)
        h["dec_time"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        coup.step(action, zv, zc)
        h["coup_time"].append(time.perf_counter() - t0)
        for pf, pref in [(dec, "dec"), (coup, "coup")]:
            h[pref+"_x"].append(pf.x_estimate)
            h[pref+"_bt"].append(pf.belief_theta.copy())
            h[pref+"_te"].append(pf.theta_estimate)
            h[pref+"_Ptr"].append(np.trace(pf.x_covariance))
            h[pref+"_phi_std"].append(np.sqrt(max(pf.x_covariance[2,2], 0)))
    
    # Store final covariances for compounding
    h["dec_final_P"] = dec.x_covariance.copy()
    h["coup_final_P"] = coup.x_covariance.copy()
    return h


def run_sequential(params, n_blocks, steps_per_block, n_particles=1000):

    block_histories = []
    accumulated_cov = np.zeros((3, 3))

    for b in range(n_blocks):
        init_cov_b = params.init_cov + accumulated_cov

        hist = run_single(
            params, steps_per_block, n_particles=n_particles,
            init_mean_override=params.init_mean,
            init_cov_override=init_cov_b,
        )
        block_histories.append(hist)

        # Compound: add average of both filters' final covariance
        # (same ground truth, different estimates)
        avg_final_P = 0.5 * (hist["dec_final_P"] + hist["coup_final_P"])
        accumulated_cov = accumulated_cov + avg_final_P

    return block_histories

def generate_trajectory(params, n_steps):

    tt, x0 = sample_init(params)
    dec = DecoupledFilter(params)  # only used for action selection
    traj = dict(theta_true=tt, x_true=[x0.copy()], actions=[], z_v=[], z_c=[])
    x = x0.copy()
    for t in range(n_steps):
        action = policy(dec.x_estimate, t, params)
        traj["actions"].append(action)
        x = transition(x, action, params)
        zv = observe_visual(x, params)
        zc = (observe_contact(tt, x, params)
              if action == Action.ATTEMPT_ATTACH else None)
        traj["x_true"].append(x.copy())
        traj["z_v"].append(zv)
        traj["z_c"].append(zc)

        # Update the EKF
        dec.step(action, zv, zc)
    return traj


def run_filter_on_trajectory(traj, params, filter_type="decoupled",
                              n_particles=1000):

    from filters import DecoupledFilter, CoupledFilter

    if filter_type == "decoupled":
        filt = DecoupledFilter(params)
    else:
        filt = CoupledFilter(params, n_particles=n_particles)

    tt = traj["theta_true"]
    h = dict(
        x_est=[filt.x_estimate],
        bt=[filt.belief_theta.copy()],
        te=[filt.theta_estimate],
        phi_std=[np.sqrt(max(filt.x_covariance[2, 2], 0))],
        time_per_step=[],
    )

    for t, action in enumerate(traj["actions"]):
        zv = traj["z_v"][t]
        zc = traj["z_c"][t]
        t0 = time.perf_counter()
        filt.step(action, zv, zc)
        h["time_per_step"].append(time.perf_counter() - t0)
        h["x_est"].append(filt.x_estimate)
        h["bt"].append(filt.belief_theta.copy())
        h["te"].append(filt.theta_estimate)
        h["phi_std"].append(np.sqrt(max(filt.x_covariance[2, 2], 0)))

    h["theta_correct"] = int(h["te"][-1] == tt)
    return h
