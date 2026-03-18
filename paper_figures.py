
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pomdp import POMDPParams, wrap_angle
from simulator import run_single, run_sequential
import os

RD = "figures"
os.makedirs(RD, exist_ok=True)

# ======================================================================
# IEEE Style Configuration. # styling done with Claude to make nice figures for paper
# ======================================================================
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Colors: grayscale-friendly with distinct markers
C_DEC = "#2166ac"   # blue
C_COUP = "#b2182b"  # red
C_TRUE = "black"
LS_DEC = "-"
LS_COUP = "--"
MK_DEC = "o"
MK_COUP = "s"

# IEEE column widths
COL1 = 3.5   # single column inches
COL2 = 7.16  # double column inches

def se(vals):
    a = np.array(vals, dtype=float)
    return np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0

def mets(h):
    tt = h["theta_true"]
    xt, dx, cx = np.array(h["x_true"]), np.array(h["dec_x"]), np.array(h["coup_x"])
    de = dx - xt; de[:, 2] = [wrap_angle(e) for e in de[:, 2]]
    ce = cx - xt; ce[:, 2] = [wrap_angle(e) for e in ce[:, 2]]
    return dict(
        dok=int(h["dec_te"][-1] == tt), cok=int(h["coup_te"][-1] == tt),
        dtm=np.mean(h["dec_time"]), ctm=np.mean(h["coup_time"]),
    )


# single trajectory visual
def fig_trajectory():
    print("Fig 1: Single trajectory")
    np.random.seed(42)
    h = run_single(POMDPParams(), 100, n_particles=1000)
    tt = h["theta_true"]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2))

    xt = np.array(h["x_true"])
    dx = np.array(h["dec_x"])
    cx = np.array(h["coup_x"])

    # (a) 2D trajectory
    ax = axes[0]
    ax.plot(xt[:, 0], xt[:, 1], color=C_TRUE, lw=1.2, label="Ground truth")
    ax.plot(dx[:, 0], dx[:, 1], color=C_DEC, ls="--", lw=1.0, label="Decoupled")
    ax.plot(cx[:, 0], cx[:, 1], color=C_COUP, ls=":", lw=1.0, label="Coupled")
    ax.plot(0, 0, "k*", ms=10, zorder=5)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("(a) Trajectory")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal")

    # (b) Hidden depth y
    ax = axes[1]
    ax.plot(xt[:, 1], color=C_TRUE, lw=1.2, label="True $y$")
    ax.plot(dx[:, 1], color=C_DEC, ls="--", lw=1.0, label="Decoupled $\\hat{y}$")
    ax.plot(cx[:, 1], color=C_COUP, ls=":", lw=1.0, label="Coupled $\\hat{y}$")
    ax.set_xlabel("Timestep"); ax.set_ylabel("$y$")
    ax.set_title("(b) Depth (hidden from camera)")
    ax.legend(loc="best", framealpha=0.9)

    # (c) Compatibility belief
    ax = axes[2]
    dbt = np.array(h["dec_bt"]); cbt = np.array(h["coup_bt"])
    ax.plot(dbt[:, 1], color=C_DEC, ls=LS_DEC, lw=1.2, label="Decoupled")
    ax.plot(cbt[:, 1], color=C_COUP, ls=LS_COUP, lw=1.2, label="Coupled")
    ax.axhline(float(tt == 1), color="gray", ls=":", lw=0.8, alpha=0.6)
    for t, zc in enumerate(h["z_c"]):
        if zc is not None:
            ax.plot(t+1, -0.06, "^" if zc else "v",
                    color="forestgreen" if zc else "firebrick",
                    ms=3.5, alpha=0.7)
    ax.set_xlabel("Timestep"); ax.set_ylabel("$p(\\theta = \\mathrm{compatible})$")
    ax.set_title("(c) Compatibility belief")
    # ax.set_ylim(-0.12, 1.08)
    ax.legend(loc="center right", framealpha=0.9)

    plt.tight_layout(w_pad=1.5)
    plt.savefig(f"{RD}/fig1_trajectory.pdf"); plt.savefig(f"{RD}/fig1_trajectory.png")
    plt.close(); print("  -> fig1_trajectory.pdf")


# particle count tradeoff

def fig_particles(nt=30, ns=100):
    print(f"Fig 2: Particle Pareto ({nt} trials)")
    Ns = [50, 100, 200, 500, 1000, 2000]
    p = POMDPParams(obs_noise_x=0.15)
    R = {N: {"a": [], "t": []} for N in Ns}
    da, dt = [], []

    for trial in range(nt):
        for N in Ns:
            np.random.seed(trial * 10000 + N)
            m = mets(run_single(p, ns, n_particles=N))
            R[N]["a"].append(m["cok"]); R[N]["t"].append(m["ctm"])
        np.random.seed(trial * 10000)
        m = mets(run_single(p, ns))
        da.append(m["dok"]); dt.append(m["dtm"])

    accs = [np.mean(R[N]["a"]) for N in Ns]
    acc_e = [se(R[N]["a"]) for N in Ns]
    tms = [np.mean(R[N]["t"]) * 1000 for N in Ns]
    tms_e = [se(R[N]["t"]) * 1000 for N in Ns]

    fig, ax1 = plt.subplots(figsize=(COL1, 2.5))

    # Accuracy on left axis
    ln1 = ax1.errorbar(Ns, accs, yerr=acc_e, fmt=f"{MK_COUP}{LS_COUP}",
                        color=C_COUP, capsize=3, label="Coupled accuracy")
    ax1.axhline(np.mean(da), color=C_DEC, ls="--", lw=1, alpha=0.7)
    ax1.axhspan(np.mean(da) - se(da), np.mean(da) + se(da),
                color=C_DEC, alpha=0.08)
    ax1.set_xlabel("Number of particles")
    ax1.set_ylabel("$\\theta$ classification accuracy")
    # ax1.set_ylim(0.9, 1.02)

    # Time on right axis
    ax2 = ax1.twinx()
    ln2 = ax2.errorbar(Ns, tms, yerr=tms_e, fmt=f"{MK_DEC}-",
                        color="gray", capsize=3, label="Compute (ms/update)")
    ax2.set_ylabel("Time per update (ms)")

    # Combined legend
    lns = [ln1, ln2]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", framealpha=0.9)
    ax1.annotate("Decoupled baseline", xy=(Ns[-1]*0.6, np.mean(da)),
                 fontsize=7, color=C_DEC, style="italic")

    plt.tight_layout()
    plt.savefig(f"{RD}/fig2_particles.pdf"); plt.savefig(f"{RD}/fig2_particles.png")
    plt.close(); print("  -> fig2_particles.pdf")


# pose estimation accuracy (x, y, phi RMSE) across noise levels, with error bars

def fig_pose_rmse(nt=30, ns=100, pf_reps=5):
    print(f"Fig 3: Per-state RMSE ({nt} scenarios, {pf_reps} PF reps each)")
    from simulator import generate_trajectory, run_filter_on_trajectory
    noises = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    R = {n: {"d_x": [], "d_y": [], "d_phi": [],
             "c_x": [], "c_y": [], "c_phi": []} for n in noises}

    for noise in noises:
        print(f"  sigma={noise}")
        p = POMDPParams(obs_noise_x=noise)
        for trial in range(nt):
            np.random.seed(trial)
            traj = generate_trajectory(p, ns)
            xt = np.array(traj["x_true"])

            # EKF - deterministic given observations
            np.random.seed(trial * 100000)
            hd = run_filter_on_trajectory(traj, p, "decoupled")
            dx = np.array(hd["x_est"])
            de = dx[-20:] - xt[-20:]
            de[:, 2] = [wrap_angle(e) for e in de[:, 2]]
            R[noise]["d_x"].append(np.sqrt(np.mean(de[:, 0]**2)))
            R[noise]["d_y"].append(np.sqrt(np.mean(de[:, 1]**2)))
            R[noise]["d_phi"].append(np.sqrt(np.mean(de[:, 2]**2)))

            # PF - multiple reps
            for rep in range(pf_reps):
                np.random.seed(trial * 100000 + rep + 1)
                hc = run_filter_on_trajectory(traj, p, "coupled", n_particles=1000)
                cx = np.array(hc["x_est"])
                ce = cx[-20:] - xt[-20:]
                ce[:, 2] = [wrap_angle(e) for e in ce[:, 2]]
                R[noise]["c_x"].append(np.sqrt(np.mean(ce[:, 0]**2)))
                R[noise]["c_y"].append(np.sqrt(np.mean(ce[:, 1]**2)))
                R[noise]["c_phi"].append(np.sqrt(np.mean(ce[:, 2]**2)))

    # Shared y-axis upper limit across all three panels
    all_vals = []
    for n in noises:
        for k in ["d_x", "c_x", "d_y", "c_y", "d_phi", "c_phi"]:
            all_vals.append(np.mean(R[n][k]) + se(R[n][k]))
    y_max = max(all_vals) * 1.1

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))

    # (a) x RMSE — observed by camera
    ax = axes[0]
    ax.errorbar(noises, [np.mean(R[n]["d_x"]) for n in noises],
                yerr=[se(R[n]["d_x"]) for n in noises],
                fmt=f"{MK_DEC}{LS_DEC}", color=C_DEC, capsize=3, label="Decoupled")
    ax.errorbar(noises, [np.mean(R[n]["c_x"]) for n in noises],
                yerr=[se(R[n]["c_x"]) for n in noises],
                fmt=f"{MK_COUP}{LS_COUP}", color=C_COUP, capsize=3, label="Coupled")
    ax.set_xlabel("$\\sigma_x^{\\mathrm{obs}}$")
    ax.set_ylabel("RMSE")
    ax.set_title("(a) $x$ (observed)")
    ax.legend(framealpha=0.9, fontsize=7)
    ax.set_ylim(0, y_max)

    # (b) y RMSE — hidden from camera
    ax = axes[1]
    ax.errorbar(noises, [np.mean(R[n]["d_y"]) for n in noises],
                yerr=[se(R[n]["d_y"]) for n in noises],
                fmt=f"{MK_DEC}{LS_DEC}", color=C_DEC, capsize=3, label="Decoupled")
    ax.errorbar(noises, [np.mean(R[n]["c_y"]) for n in noises],
                yerr=[se(R[n]["c_y"]) for n in noises],
                fmt=f"{MK_COUP}{LS_COUP}", color=C_COUP, capsize=3, label="Coupled")
    ax.set_xlabel("$\\sigma_x^{\\mathrm{obs}}$")
    ax.set_ylabel("RMSE")
    ax.set_title("(b) $y$ (hidden)")
    ax.legend(framealpha=0.9, fontsize=7)
    ax.set_ylim(0, y_max)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    # (c) phi RMSE — observed by camera
    ax = axes[2]
    ax.errorbar(noises, [np.mean(R[n]["d_phi"]) for n in noises],
                yerr=[se(R[n]["d_phi"]) for n in noises],
                fmt=f"{MK_DEC}{LS_DEC}", color=C_DEC, capsize=3, label="Decoupled")
    ax.errorbar(noises, [np.mean(R[n]["c_phi"]) for n in noises],
                yerr=[se(R[n]["c_phi"]) for n in noises],
                fmt=f"{MK_COUP}{LS_COUP}", color=C_COUP, capsize=3, label="Coupled")
    ax.set_xlabel("$\\sigma_x^{\\mathrm{obs}}$")
    ax.set_ylabel("RMSE (rad)")
    ax.set_title("(c) $\\phi$ (observed)")
    ax.legend(framealpha=0.9, fontsize=7)
    ax.set_ylim(0, y_max)

    plt.tight_layout(w_pad=1.5)
    plt.savefig(f"{RD}/fig3_pose_rmse.pdf"); plt.savefig(f"{RD}/fig3_pose_rmse.png")
    plt.close(); print("  -> fig3_pose_rmse.pdf")

# noise tradeoff: single block accuracy vs noise, and sequential accuracy vs noise across block positions

def fig_noise_tradeoff(nt_single=100, nt_seq=30, max_blocks=5, spb=60):
    print(f"Fig 4: Noise tradeoff (single={nt_single}, seq={nt_seq} trials)")
    noises_single = [0.05, 0.15, 0.3, 0.5]
    noises_seq = [0.05, 0.2, 0.5]

    # --- Single block data ---
    single = {n: {"da": [], "ca": []} for n in noises_single}
    for noise in noises_single:
        print(f"  single sigma={noise}")
        for trial in range(nt_single):
            np.random.seed(trial)
            m = mets(run_single(POMDPParams(obs_noise_x=noise), spb))
            single[noise]["da"].append(m["dok"])
            single[noise]["ca"].append(m["cok"])

    # --- Sequential data ---
    seq_dec = {n: {k: [] for k in range(max_blocks)} for n in noises_seq}
    seq_coup = {n: {k: [] for k in range(max_blocks)} for n in noises_seq}
    for noise in noises_seq:
        print(f"  sequential sigma={noise}")
        for trial in range(nt_seq):
            np.random.seed(trial * 999 + int(noise * 1000))
            blocks = run_sequential(POMDPParams(obs_noise_x=noise),
                                     max_blocks, spb, n_particles=1000)
            for k, bh in enumerate(blocks):
                seq_dec[noise][k].append(int(bh["dec_te"][-1] == bh["theta_true"]))
                seq_coup[noise][k].append(int(bh["coup_te"][-1] == bh["theta_true"]))

    positions = list(range(1, max_blocks + 1))

    fig, ax = plt.subplots(figsize=(COL1, 2.5))

    da_m = np.array([np.mean(single[n]["da"]) for n in noises_single])
    da_e = np.array([1.96 * se(single[n]["da"]) for n in noises_single])
    ca_m = np.array([np.mean(single[n]["ca"]) for n in noises_single])
    ca_e = np.array([1.96 * se(single[n]["ca"]) for n in noises_single])

    ld, = ax.plot(noises_single, da_m, f"{MK_DEC}{LS_DEC}", color=C_DEC, label="Decoupled")
    ax.fill_between(noises_single, np.clip(da_m - da_e, 0, 1),
                    np.clip(da_m + da_e, 0, 1.05),
                    color=C_DEC, alpha=0.15)
    lc, = ax.plot(noises_single, ca_m, f"{MK_COUP}{LS_COUP}", color=C_COUP, label="Coupled")
    ax.fill_between(noises_single, np.clip(ca_m - ca_e, 0, 1),
                    np.clip(ca_m + ca_e, 0, 1.05),
                    color=C_COUP, alpha=0.15)
    ax.set_xlabel("Visual noise ($\\sigma_x$)")
    ax.set_ylabel("$\\theta$ accuracy")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{RD}/fig4a_single_block.pdf")
    plt.savefig(f"{RD}/fig4a_single_block.png")
    plt.close()
    print("  -> fig4a_single_block.pdf")

    sq = 2.4
    fig, axes = plt.subplots(1, 3, figsize=(3 * sq + 1.0, sq))

    for j, (noise, ax) in enumerate(zip(noises_seq, axes)):
        d_acc = np.array([np.mean(seq_dec[noise][k]) for k in range(max_blocks)])
        d_err = np.array([1.96 * se(seq_dec[noise][k]) for k in range(max_blocks)])
        c_acc = np.array([np.mean(seq_coup[noise][k]) for k in range(max_blocks)])
        c_err = np.array([1.96 * se(seq_coup[noise][k]) for k in range(max_blocks)])

        ax.errorbar(positions, d_acc, yerr=d_err,
                    fmt=f"{MK_DEC}{LS_DEC}", color=C_DEC, capsize=3,
                    label="Decoupled")
        ax.errorbar(positions, c_acc, yerr=c_err,
                    fmt=f"{MK_COUP}{LS_COUP}", color=C_COUP, capsize=3,
                    label="Coupled")

        ax.set_xlabel("Block position")
        ax.set_title(f"$\\sigma_x^{{\\mathrm{{obs}}}}$ = {noise}")
        ax.set_xticks(positions)
        ax.set_aspect("auto")
        ax.set_box_aspect(1)
        if j == 0:
            ax.set_ylabel("$\\theta$ accuracy")
            ax.legend(framealpha=0.9, fontsize=7)
        else:
            ax.set_yticklabels([])

    # Share y-axis limits
    all_axes_ylim = [ax.get_ylim() for ax in axes]
    ylo = min(y[0] for y in all_axes_ylim)
    yhi = max(y[1] for y in all_axes_ylim)
    for ax in axes:
        ax.set_ylim(ylo, yhi)

    plt.tight_layout(w_pad=0.5)
    plt.savefig(f"{RD}/fig4b_sequential.pdf")
    plt.savefig(f"{RD}/fig4b_sequential.png")
    plt.close()
    print("  -> fig4b_sequential.pdf")

# computational cost

def fig_compute(nt=30, ns=100):
    print(f"Fig 5: Computational cost ({nt} trials)")
    Ns = [100, 200, 500, 1000, 2000]
    p = POMDPParams(obs_noise_x=0.15)
    R = {N: [] for N in Ns}
    dec_t = []

    for trial in range(nt):
        for N in Ns:
            np.random.seed(trial * 10000 + N)
            m = mets(run_single(p, ns, n_particles=N))
            R[N].append(m["ctm"] * 1000)
        np.random.seed(trial * 10000)
        m = mets(run_single(p, ns))
        dec_t.append(m["dtm"] * 1000)

    fig, ax = plt.subplots(figsize=(COL1, 2.5))

    tms = [np.mean(R[N]) for N in Ns]
    tms_e = [se(R[N]) for N in Ns]

    ax.errorbar(Ns, tms, yerr=tms_e, fmt=f"{MK_COUP}{LS_COUP}",
                color=C_COUP, capsize=3, label="Coupled (PF)")
    ax.axhline(np.mean(dec_t), color=C_DEC, ls="-", lw=1.2,
               label="Decoupled (EKF)")
    ax.axhspan(np.mean(dec_t) - se(dec_t),
               np.mean(dec_t) + se(dec_t),
               color=C_DEC, alpha=0.1)


    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Time per update (ms)")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{RD}/fig5_compute.pdf"); plt.savefig(f"{RD}/fig5_compute.png")
    plt.close(); print("  -> fig5_compute.pdf")


def fig_noise_decomposition(nt=50, ns=100):
    """
    Three panels:
      (a) Sweep obs_noise_x
      (b) Sweep obs_noise_phi
      (c) Sweep all three together
    """
    print(f"Fig 6: Noise decomposition ({nt} trials)")

    def _sweep(param_name, values, fixed_kwargs):
        da_all, ca_all = {}, {}
        for v in values:
            da_all[v], ca_all[v] = [], []
            kw = dict(fixed_kwargs)
            kw[param_name] = v
            for trial in range(nt):
                np.random.seed(trial)
                p = POMDPParams(**kw)
                m = mets(run_single(p, ns, n_particles=1000))
                da_all[v].append(m["dok"])
                ca_all[v].append(m["cok"])
        return da_all, ca_all

    default_x = 0.05
    default_phi_obs = 0.05

    # (a) Sweep obs_noise_x
    vis_levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    print("  (a) Sweep obs_noise_x ...")
    da_vis, ca_vis = _sweep("obs_noise_x", vis_levels,
                             {"obs_noise_phi": default_phi_obs})

    # (b) Sweep obs_noise_phi
    phi_obs_levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    print("  (b) Sweep obs_noise_phi ...")
    da_phi_obs, ca_phi_obs = _sweep("obs_noise_phi", phi_obs_levels,
                                     {"obs_noise_x": default_x})

    # (c) Sweep all together
    all_levels = [(0.02, 0.02), (0.05, 0.05), (0.1, 0.1),
                  (0.2, 0.2), (0.3, 0.3), (0.5, 0.5)]
    print("  (c) Sweep all together ...")
    da_all, ca_all = {}, {}
    for pair in all_levels:
        da_all[pair], ca_all[pair] = [], []
        for trial in range(nt):
            np.random.seed(trial)
            p = POMDPParams(obs_noise_x=pair[0], obs_noise_phi=pair[1])
            m = mets(run_single(p, ns, n_particles=1000))
            da_all[pair].append(m["dok"])
            ca_all[pair].append(m["cok"])

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))

    def _plot_panel(ax, values, da, ca, xlabel, title, fixed_str=None):
        da_m = np.array([np.mean(da[v]) for v in values])
        da_e = np.array([1.96 * se(da[v]) for v in values])
        ca_m = np.array([np.mean(ca[v]) for v in values])
        ca_e = np.array([1.96 * se(ca[v]) for v in values])
        ax.errorbar(range(len(values)), da_m, yerr=da_e,
                    fmt=f"{MK_DEC}{LS_DEC}", color=C_DEC, capsize=3,
                    label="Decoupled")
        ax.errorbar(range(len(values)), ca_m, yerr=ca_e,
                    fmt=f"{MK_COUP}{LS_COUP}", color=C_COUP, capsize=3,
                    label="Coupled")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$\\theta$ accuracy")
        ax.set_title(title)
        ax.legend(framealpha=0.9, fontsize=7)
        # ax.set_ylim(0.78, 1.03)
        if fixed_str:
            ax.text(0.98, 0.03, fixed_str, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7,
                    color="#666666", fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.8))

    _plot_panel(axes[0], vis_levels, da_vis, ca_vis,
                "$\\sigma_x^{\\mathrm{obs}}$",
                "(a) Vary position obs noise",
                f"$\\sigma_\\phi^{{\\mathrm{{obs}}}}$={default_phi_obs} fixed")

    _plot_panel(axes[1], phi_obs_levels, da_phi_obs, ca_phi_obs,
                "$\\sigma_\\phi^{\\mathrm{obs}}$",
                "(b) Vary heading obs noise",
                f"$\\sigma_x^{{\\mathrm{{obs}}}}$={default_x} fixed")

    # (c) All together
    ax = axes[2]
    da_m = np.array([np.mean(da_all[t]) for t in all_levels])
    da_e = np.array([1.96 * se(da_all[t]) for t in all_levels])
    ca_m = np.array([np.mean(ca_all[t]) for t in all_levels])
    ca_e = np.array([1.96 * se(ca_all[t]) for t in all_levels])
    x_idx = list(range(len(all_levels)))
    ax.errorbar(x_idx, da_m, yerr=da_e,
                fmt=f"{MK_DEC}{LS_DEC}", color=C_DEC, capsize=3,
                label="Decoupled")
    ax.errorbar(x_idx, ca_m, yerr=ca_e,
                fmt=f"{MK_COUP}{LS_COUP}", color=C_COUP, capsize=3,
                label="Coupled")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"{t[0]}" for t in all_levels], fontsize=7)
    ax.set_xlabel("$\\sigma_{total}$")
    ax.set_ylabel("$\\theta$ accuracy")
    ax.set_title("(c) Vary both obs noises jointly")
    ax.legend(framealpha=0.9, fontsize=7)
    # ax.set_ylim(0.78, 1.03)

    plt.tight_layout(w_pad=1.5)
    plt.savefig(f"{RD}/fig6_noise_decomposition.pdf")
    plt.savefig(f"{RD}/fig6_noise_decomposition.png")
    plt.close()
    print("  -> fig6_noise_decomposition.pdf")
    
# ======================================================================
if __name__ == "__main__":
    fig_trajectory()
    fig_particles(nt=100, ns=100)
    fig_pose_rmse(nt=100, ns=100)
    fig_noise_tradeoff(nt_single=100, nt_seq=30, max_blocks=5, spb=60)
    fig_compute(nt=100, ns=100)
    fig_noise_decomposition(nt=100, ns=100)
    print(f"\nAll figures saved to ./{RD}/")
