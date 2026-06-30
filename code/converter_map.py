"""
converter_map.py
================
Exact cycle-by-cycle (stroboscopic) iterated map of a peak current-mode
controlled DC-DC boost converter, with optional ramp (slope) compensation.

Continuous-conduction-mode (CCM) boost converter:
    State x = [i_L (inductor current), v_C (output capacitor voltage)].

    Switch ON  (subinterval D*T): diode off,
        L di/dt = Vin
        C dv/dt = -v/R
    Switch OFF (subinterval (1-D)*T): diode on,
        L di/dt = Vin - v
        C dv/dt =  i - v/R

Peak current-mode control:
    - Clock turns the switch ON at the start of every period T.
    - Switch turns OFF when  i_L(t) + m_c*(t - nT) >= I_ref
      where m_c >= 0 is the artificial compensating ramp slope (slope
      compensation). m_c = 0 is the uncompensated case.
    - If the threshold is never reached within the period the switch stays
      ON for the whole period (current-limit saturation -> border collision).

The map integrates the two affine LTI subsystems *exactly* with matrix
exponentials, so it is not a small-ripple approximation: it is the true
Poincare (stroboscopic) map sampled at the clock instants t = nT.

All component values are SI. Reference current I_ref (A) is the natural
bifurcation parameter; input voltage Vin (V) is an alternative.

Author: generated for F. Gul, RTEU AI&IoT Lab.
"""

import numpy as np
from scipy.linalg import expm

# ----------------------------------------------------------------------
# Nominal converter parameters (a deliberately fast, lightly-damped design
# that exhibits the classical period-doubling cascade in current mode).
# ----------------------------------------------------------------------
PARAMS = dict(
    L=1.0e-3,      # H   inductor
    C=4.0e-6,      # F   output capacitor
    R=20.0,        # ohm load
    Vin=10.0,      # V   input voltage
    T=1.0/10000.0, # s   switching period (fs = 10 kHz)
    Iref=2.0,      # A   peak current reference (bifurcation parameter)
    mc=0.0,        # A/s artificial ramp slope (slope compensation)
)


def _matrices(p):
    """Return (A_on, A_off, b) affine subsystem matrices for state [iL, vC]."""
    L, C, R, Vin = p["L"], p["C"], p["R"], p["Vin"]
    # ON: switch closed, diode blocked
    A_on = np.array([[0.0, 0.0],
                     [0.0, -1.0/(R*C)]])
    # OFF: diode conducts
    A_off = np.array([[0.0, -1.0/L],
                      [1.0/C, -1.0/(R*C)]])
    b = np.array([Vin/L, 0.0])     # input vector (same in both subintervals)
    return A_on, A_off, b


def _flow(A, b, x0, t):
    """Exact solution of xdot = A x + b for duration t, start x0.
    x(t) = expm(A t) x0 + A^{-1}(expm(A t) - I) b   (A invertible here)."""
    if t <= 0:
        return x0.copy()
    Phi = expm(A * t)
    # A is invertible for both subsystems (check: A_off det = 1/(LC) != 0;
    # A_on is singular in iL row -> handle separately)
    # General robust approach: augmented matrix exponential.
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    E = expm(M * t)
    Phi = E[:n, :n]
    g = E[:n, n]
    return Phi @ x0 + g


def _on_slope(p):
    """di/dt during ON subinterval = Vin/L (constant)."""
    return p["Vin"] / p["L"]


def switching_instant(p, x0):
    """Time within the period at which i_L + mc*t reaches Iref during ON.
    Returns (d, hit) where d in (0,T] is the on-time and hit=True if the
    comparator tripped before the clock; if the current never reaches the
    threshold, returns (T, False) -> switch stays on whole cycle."""
    Iref, mc, T = p["Iref"], p["mc"], p["T"]
    iL0 = x0[0]
    s = _on_slope(p)            # current rise slope
    # i_L(t) ~ iL0 + s t (vC barely affects iL in ON since A_on row for iL is 0)
    # comparator: iL0 + s t + mc t = Iref  -> but compensation ramp is
    # *subtracted* from the reference equivalently: trip when
    #   iL0 + s t >= Iref - mc t  => iL0 + (s+mc) t >= Iref
    denom = s + mc
    if denom <= 0:
        return T, False
    d = (Iref - iL0) / denom
    if d <= 0.0:
        # already above threshold at clock -> minimal on-time (immediate)
        return 1e-12, True
    if d >= T:
        return T, False
    return d, True


def poincare_map(x0, p):
    """One clock-period stroboscopic map x_{n} -> x_{n+1}."""
    A_on, A_off, b = _matrices(p)
    d, hit = switching_instant(p, x0)
    # ON phase for duration d
    x1 = _flow(A_on, b, x0, d)
    # OFF phase for remaining (T-d); if not hit, off time = 0
    toff = p["T"] - d
    x2 = _flow(A_off, b, x1, toff)
    return x2


def iterate(p, x0=None, n=4000, n_transient=2000):
    """Iterate the map; return steady-state samples (after transient)."""
    if x0 is None:
        x0 = np.array([0.5, p["Vin"]])  # start: some current, vC=Vin
    x = x0.copy()
    traj = np.empty((n, 2))
    for k in range(n_transient):
        x = poincare_map(x, p)
    for k in range(n):
        x = poincare_map(x, p)
        traj[k] = x
    return traj


def jacobian(x0, p, eps=1e-7):
    """Numerical Jacobian of the Poincare map at x0 (2x2)."""
    J = np.zeros((2, 2))
    f0 = poincare_map(x0, p)
    for j in range(2):
        dx = np.zeros(2); dx[j] = eps * max(1.0, abs(x0[j]))
        fj = poincare_map(x0 + dx, p)
        J[:, j] = (fj - f0) / dx[j]
    return J


def lyapunov_exponents(p, x0=None, n=6000, n_transient=2000):
    """Largest Lyapunov exponent (per iteration) of the discrete map via
    QR-based product of Jacobians along the orbit. Returns both exponents."""
    if x0 is None:
        x0 = np.array([0.5, p["Vin"]])
    x = x0.copy()
    for _ in range(n_transient):
        x = poincare_map(x, p)
    Q = np.eye(2)
    lyap = np.zeros(2)
    for k in range(n):
        J = jacobian(x, p)
        Z = J @ Q
        Q, Rmat = np.linalg.qr(Z)
        d = np.diag(Rmat).copy()
        # fix signs so diagonal positive
        s = np.sign(d); s[s == 0] = 1
        Q = Q * s
        d = np.abs(d)
        d[d < 1e-300] = 1e-300
        lyap += np.log(d)
        x = poincare_map(x, p)
    lyap /= n
    return np.sort(lyap)[::-1]   # largest first, units: per iteration (nat/cycle)


if __name__ == "__main__":
    p = dict(PARAMS)
    for Iref in [1.0, 2.0, 3.0, 4.0]:
        p["Iref"] = Iref
        tr = iterate(p, n=200, n_transient=2000)
        iL = tr[:, 0]
        uniq = np.unique(np.round(iL[-40:], 4))
        le = lyapunov_exponents(p, n=2000, n_transient=1500)
        print(f"Iref={Iref:.1f}  distinct iL levels~{len(uniq):2d}  "
              f"iL range=[{iL.min():.3f},{iL.max():.3f}]  LLE={le[0]:+.4f}/cyc")
