"""
make_figures.py
===============
Generate all publication figures (vector PDF + 300-dpi PNG, colorblind-safe).
Reads CSVs from ../results and ngspice .raw transients from ../data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
import os

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.linewidth": 0.9, "lines.linewidth": 1.3,
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.labelsize": 12, "legend.fontsize": 9.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 120,
})
# Okabe-Ito colorblind-safe palette
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
          purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", black="#000000")
T = 1.0/10000.0


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", name)


def strob(prefix, t0=18e-3):
    d = np.loadtxt(os.path.join(DATA, prefix + ".raw"))
    t = d[:, 0]; iL = d[:, 1]; vout = d[:, 3]; q = d[:, 5]
    return t, iL, vout, q


def strob_samples(t, iL, t0=18e-3):
    mask = t > t0
    ts, iLs = t[mask], iL[mask]
    ks = np.arange(int(np.ceil(ts[0]/T)), int(ts[-1]/T))
    return ks*T, np.interp(ks*T, ts, iLs)


# ---------------------------------------------------------------------
# Fig 1: converter schematic (drawn programmatically)
# ---------------------------------------------------------------------
def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    lw = 1.6
    def wire(x1, y1, x2, y2): ax.plot([x1, x2], [y1, y2], color="k", lw=lw)
    # input source
    ax.add_patch(Circle((1, 3), 0.55, fill=False, lw=lw))
    ax.text(1, 3, r"$V_{\rm in}$", ha="center", va="center", fontsize=11)
    wire(1, 3.55, 1, 5); wire(1, 2.45, 1, 1)
    # inductor (top rail) to node sw
    wire(1, 5, 3, 5)
    ax.text(3.7, 5.45, r"$L$", ha="center", fontsize=12)
    xs = np.linspace(3, 4.4, 60)
    ax.plot(xs, 5+0.16*np.sin((xs-3)/1.4*4*np.pi), color="k", lw=lw)
    wire(4.4, 5, 6.2, 5)            # node 'sw'
    ax.text(6.0, 5.35, "sw", fontsize=9, color=CB["blue"])
    # diode to out
    wire(6.2, 5, 7.2, 5)
    ax.add_patch(Polygon([[7.2, 5.28], [7.2, 4.72], [7.85, 5]], closed=True,
                             fill=False, lw=lw))
    ax.plot([7.85, 7.85], [4.72, 5.28], color="k", lw=lw)
    ax.text(7.5, 5.45, r"$D$", ha="center", fontsize=12)
    wire(7.85, 5, 9.5, 5)          # node 'out'
    ax.text(9.4, 5.35, "out", fontsize=9, color=CB["red"])
    # switch (MOSFET) from sw to ground
    wire(6.2, 5, 6.2, 3.4)
    ax.add_patch(Rectangle((5.9, 2.2), 0.6, 1.2, fill=False, lw=lw))
    ax.text(6.85, 2.8, "S", fontsize=11)
    wire(6.2, 2.2, 6.2, 1)
    ax.annotate("", xy=(5.9, 2.8), xytext=(5.0, 2.8),
                arrowprops=dict(arrowstyle="->", lw=lw))
    ax.text(4.9, 2.8, r"$q(t)$", ha="right", va="center", fontsize=10,
            color=CB["green"])
    # output cap
    wire(9.5, 5, 9.5, 3.4)
    ax.plot([9.15, 9.85], [3.4, 3.4], color="k", lw=lw)
    ax.plot([9.15, 9.85], [3.15, 3.15], color="k", lw=lw)
    ax.text(10.2, 3.3, r"$C$", fontsize=12)
    wire(9.5, 3.15, 9.5, 1)
    # load R
    wire(11, 5, 11, 3.6); wire(9.5, 5, 11, 5)
    ax.add_patch(Rectangle((10.7, 2.4), 0.6, 1.2, fill=False, lw=lw))
    ax.text(11.55, 3.0, r"$R$", fontsize=12)
    wire(11, 2.4, 11, 1)
    # ground rail
    wire(1, 1, 11, 1)
    # controller box
    ax.add_patch(Rectangle((3.6, 0.1)*np.array([1,1]), 0, 0))
    ax.add_patch(FancyBboxPatch((2.0, -0.15), 3.0, 1.0,
                 boxstyle="round,pad=0.05", fill=False, lw=lw,
                 edgecolor=CB["green"]))
    ax.text(3.5, 0.35, "PCMC + slope comp.\nlatch (clk @ $f_s$)",
            ha="center", va="center", fontsize=8.5, color=CB["green"])
    ax.annotate("", xy=(5.0, 2.7), xytext=(4.6, 0.85),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=CB["green"]))
    # current sense arrow
    ax.annotate(r"$i_L\!\rightarrow$", xy=(2.1, 5.32), xytext=(2.1, 5.32),
                fontsize=10)
    ax.annotate("", xy=(1.2, 0.55), xytext=(2.0, 0.35),
                arrowprops=dict(arrowstyle="<-", lw=1.0, color=CB["green"]))
    ax.text(0.6, 0.4, r"$i_L$ sense", fontsize=8, color=CB["green"], ha="left")
    save(fig, "fig1_schematic")


# ---------------------------------------------------------------------
# Fig 2: bifurcation diagram (uncompensated) + period color
# ---------------------------------------------------------------------
def fig_bifurcation(tag="mc0", fname="fig2_bifurcation_uncomp", title=""):
    bif = np.loadtxt(os.path.join(RES, f"bifurcation_{tag}.csv"),
                     delimiter=",", skiprows=1)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(bif[:, 0], bif[:, 1], ".", ms=0.4, color=CB["blue"], alpha=0.55,
            rasterized=True)
    ax.set_xlabel(r"reference current $I_{\rm ref}$ (A)")
    ax.set_ylabel(r"stroboscopic $i_L$ at clock instants (A)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlim(bif[:, 0].min(), bif[:, 0].max())
    save(fig, fname)


# ---------------------------------------------------------------------
# Fig 3: Lyapunov exponent + period
# ---------------------------------------------------------------------
def fig_lyapunov():
    s = np.loadtxt(os.path.join(RES, "summary_mc0.csv"),
                   delimiter=",", skiprows=1)
    Iref, per, lle = s[:, 0], s[:, 1], s[:, 2]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.plot(Iref, lle, "-", color=CB["red"], lw=1.4)
    ax.fill_between(Iref, 0, lle, where=lle > 0, color=CB["red"], alpha=0.18,
                    label="chaotic ($\\lambda_1>0$)")
    ax.set_xlabel(r"reference current $I_{\rm ref}$ (A)")
    ax.set_ylabel(r"largest Lyapunov exponent $\lambda_1$ (nat/cycle)")
    ax.legend(loc="upper left")
    ax.set_xlim(Iref.min(), Iref.max())
    save(fig, "fig3_lyapunov")


# ---------------------------------------------------------------------
# Fig 4: ngspice time-domain waveforms period-1 / 2 / chaos
# ---------------------------------------------------------------------
def fig_timedomain():
    cases = [("boost_test", "(a) $I_{\\rm ref}=0.8$ A: period-1", CB["blue"]),
             ("boost_p2",   "(b) $I_{\\rm ref}=1.2$ A: period-2", CB["green"]),
             ("boost_chaos","(c) $I_{\\rm ref}=2.4$ A: chaos",    CB["red"])]
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=True)
    for ax, (pfx, lbl, c) in zip(axs, cases):
        t, iL, vout, q = strob(pfx)
        m = (t > 30e-3) & (t < 35e-3)
        ax.plot(t[m]*1e3, iL[m], color=c, lw=1.0)
        ax.set_ylabel(r"$i_L$ (A)")
        ax.text(0.015, 0.86, lbl, transform=ax.transAxes, fontsize=10,
                va="top")
        ax.grid(alpha=0.25)
    axs[-1].set_xlabel("time (ms)")
    save(fig, "fig4_timedomain")


# ---------------------------------------------------------------------
# Fig 5: Poincare / phase plots (iL vs vout sampled) for three regimes
# ---------------------------------------------------------------------
def fig_poincare():
    cases = [("boost_test", "period-1", CB["blue"], "o"),
             ("boost_p2",   "period-2", CB["green"], "s"),
             ("boost_chaos","chaos",    CB["red"], ".")]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for pfx, lbl, c, mk in cases:
        d = np.loadtxt(os.path.join(DATA, pfx + ".raw"))
        t, iL, vout = d[:, 0], d[:, 1], d[:, 3]
        mask = t > 20e-3
        ts, iLs, vs = t[mask], iL[mask], vout[mask]
        ks = np.arange(int(np.ceil(ts[0]/T)), int(ts[-1]/T))
        siL = np.interp(ks*T, ts, iLs)
        sv = np.interp(ks*T, ts, vs)
        ax.scatter(sv, siL, s=(14 if mk != "." else 7), marker=mk,
                   color=c, alpha=0.7, label=lbl, edgecolors="none")
    ax.set_xlabel(r"output voltage $v_C$ (V)")
    ax.set_ylabel(r"inductor current $i_L$ (A)")
    ax.set_title("Poincar\\'e section (sampled at clock instants)", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "fig5_poincare")


# ---------------------------------------------------------------------
# Fig 6: controlled vs uncontrolled bifurcation comparison
# ---------------------------------------------------------------------
def fig_control_compare():
    b0 = np.loadtxt(os.path.join(RES, "bifurcation_mc0.csv"),
                    delimiter=",", skiprows=1)
    b1 = np.loadtxt(os.path.join(RES, "bifurcation_mcStab.csv"),
                    delimiter=",", skiprows=1)
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    axs[0].plot(b0[:, 0], b0[:, 1], ".", ms=0.4, color=CB["red"], alpha=0.5,
                rasterized=True)
    axs[0].text(0.02, 0.92, "(a) uncompensated ($m_c=0$)",
                transform=axs[0].transAxes, va="top", fontsize=10)
    axs[1].plot(b1[:, 0], b1[:, 1], ".", ms=0.4, color=CB["blue"], alpha=0.5,
                rasterized=True)
    axs[1].text(0.02, 0.92, "(b) with slope compensation",
                transform=axs[1].transAxes, va="top", fontsize=10)
    for ax in axs:
        ax.set_ylabel(r"$i_L$ (A)")
    axs[1].set_xlabel(r"reference current $I_{\rm ref}$ (A)")
    save(fig, "fig6_control_compare")


# ---------------------------------------------------------------------
# Fig 7: stability boundary (design curve) onset Iref vs mc
# ---------------------------------------------------------------------
def fig_stability_boundary():
    sb = np.loadtxt(os.path.join(RES, "stability_boundary.csv"),
                    delimiter=",", skiprows=1)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    valid = ~np.isnan(sb[:, 1])
    ax.plot(sb[valid, 0], sb[valid, 1], "o-", color=CB["purple"], ms=5)
    ax.set_xlabel(r"compensating ramp slope $m_c$ (A/s)")
    ax.set_ylabel(r"period-doubling onset $I_{\rm ref}^{*}$ (A)")
    ax.set_title("Stable-operation boundary vs slope compensation", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "fig7_stability_boundary")


# ---------------------------------------------------------------------
# Fig 8: map vs ngspice agreement (overlay stroboscopic levels)
# ---------------------------------------------------------------------
def fig_map_vs_spice():
    # map summary
    s = np.loadtxt(os.path.join(RES, "summary_mc0.csv"),
                   delimiter=",", skiprows=1)
    # ngspice stroboscopic distinct levels for the three runs
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bif = np.loadtxt(os.path.join(RES, "bifurcation_mc0.csv"),
                     delimiter=",", skiprows=1)
    ax.plot(bif[:, 0], bif[:, 1], ".", ms=0.4, color="0.7", alpha=0.5,
            rasterized=True, label="map (discrete model)")
    cols = {"boost_test": (0.8, CB["blue"]), "boost_p2": (1.2, CB["green"]),
            "boost_chaos": (2.4, CB["red"])}
    first = True
    for pfx, (Iref, c) in cols.items():
        d = np.loadtxt(os.path.join(DATA, pfx + ".raw"))
        t, iL = d[:, 0], d[:, 1]
        _, siL = strob_samples(t, iL)
        xs = np.full(len(siL[-40:]), Iref)
        ax.plot(xs, siL[-40:], "x", ms=6, color=c, mew=1.4,
                label=("ngspice circuit" if first else None))
        first = False
    ax.set_xlabel(r"reference current $I_{\rm ref}$ (A)")
    ax.set_ylabel(r"stroboscopic $i_L$ (A)")
    ax.legend(loc="upper left")
    ax.set_xlim(0.6, 3.0)
    save(fig, "fig8_map_vs_spice")


if __name__ == "__main__":
    fig_schematic()
    fig_bifurcation()
    fig_lyapunov()
    fig_timedomain()
    fig_poincare()
    fig_control_compare()
    fig_stability_boundary()
    fig_map_vs_spice()
    print("ALL FIGURES DONE")
