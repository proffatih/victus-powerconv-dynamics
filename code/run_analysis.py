"""
run_analysis.py
===============
Produces all numerical results (CSV) for the manuscript:
  1. Bifurcation diagram of i_L vs Iref (uncompensated, mc=0)
  2. Largest Lyapunov exponent vs Iref
  3. Period classification vs Iref
  4. Duty cycle at the fixed point vs Iref (to locate the D=0.5 boundary)
  5. Bifurcation diagram with slope compensation (mc = mc*)  -> stable extension
  6. Stability boundary (onset of period-2) vs mc : "design curve"
  7. Poincare / time series sample points for representative cases
Outputs go to ../results/*.csv
"""
import numpy as np
import os
from converter_map import (PARAMS, poincare_map, iterate, lyapunov_exponents,
                           jacobian, switching_instant)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)


def period_of(iL, tol=1e-4, maxper=16):
    last = iL[-2*maxper:]
    for per in range(1, maxper+1):
        if all(abs(last[k]-last[k-per]) < tol for k in range(per, len(last))):
            return per
    return 99


def duty_at_fixed_point(p, n_settle=4000):
    """Mean on-time fraction d/T at steady operation."""
    x = np.array([0.5, p["Vin"]])
    for _ in range(n_settle):
        x = poincare_map(x, p)
    ds = []
    for _ in range(64):
        d, _hit = switching_instant(p, x)
        ds.append(d/p["T"])
        x = poincare_map(x, p)
    return float(np.mean(ds))


def fixed_point_jacobian_eig(p):
    """Eigenvalues of the Jacobian at the period-1 fixed point (if it exists),
    found by a few Newton-like settle iterations."""
    x = np.array([0.5, p["Vin"]])
    for _ in range(6000):
        x = poincare_map(x, p)
    J = jacobian(x, p)
    return np.linalg.eigvals(J)


# ---------------------------------------------------------------
# 1-4. Uncompensated sweep over Iref
# ---------------------------------------------------------------
def bifurcation_sweep(mc, Iref_vals, tag, n_plot=120, n_trans=4000):
    p = dict(PARAMS); p["mc"] = mc
    bif_rows = []     # (Iref, iL_sample)
    summ_rows = []    # (Iref, period, LLE, eig1mag, eig2mag, duty)
    for Iref in Iref_vals:
        p["Iref"] = Iref
        tr = iterate(p, n=n_plot, n_transient=n_trans)
        iL = tr[:, 0]
        per = period_of(iL)
        le = lyapunov_exponents(p, n=1800, n_transient=1500)[0]
        eig = fixed_point_jacobian_eig(p)
        duty = duty_at_fixed_point(p)
        for v in iL[-n_plot:]:
            bif_rows.append((Iref, v))
        summ_rows.append((Iref, per, le, abs(eig[0]), abs(eig[1]), duty))
    np.savetxt(os.path.join(RES, f"bifurcation_{tag}.csv"),
               np.array(bif_rows), delimiter=",",
               header="Iref,iL", comments="")
    np.savetxt(os.path.join(RES, f"summary_{tag}.csv"),
               np.array(summ_rows), delimiter=",",
               header="Iref,period,LLE_per_cycle,eig1_mag,eig2_mag,duty", comments="")
    return np.array(summ_rows)


# ---------------------------------------------------------------
# 6. Stability boundary: minimum Iref for period-2 onset vs mc
#    (the compensating ramp pushes the onset to higher Iref)
# ---------------------------------------------------------------
def stability_boundary(mc_vals, Iref_scan):
    p = dict(PARAMS)
    rows = []
    for mc in mc_vals:
        p["mc"] = mc
        onset = np.nan
        for Iref in Iref_scan:
            p["Iref"] = Iref
            tr = iterate(p, n=80, n_transient=4000)
            if period_of(tr[:, 0]) > 1:
                onset = Iref
                break
        rows.append((mc, onset))
    np.savetxt(os.path.join(RES, "stability_boundary.csv"),
               np.array(rows), delimiter=",",
               header="mc_A_per_s,Iref_onset", comments="")
    return np.array(rows)


def critical_slope_theory(p):
    """Classical slope-compensation criterion for current-mode control.
    For stability the compensating ramp slope must satisfy
        mc > (m2 - m1)/2   where m1 = up-slope (Vin/L),
        m2 = down-slope magnitude = (Vo - Vin)/L.
    Returns (m1, m2, mc_crit) at the nominal operating point."""
    Vin, L = p["Vin"], p["L"]
    # boost steady output Vo = Vin/(1-D); use measured duty
    duty = duty_at_fixed_point(p)
    Vo = Vin/(1.0-duty)
    m1 = Vin/L
    m2 = (Vo - Vin)/L
    mc_crit = 0.5*(m2 - m1)
    return m1, m2, mc_crit, duty, Vo


if __name__ == "__main__":
    Iref_fine = np.arange(0.60, 3.001, 0.01)

    print("== Uncompensated sweep (mc=0) ==")
    s0 = bifurcation_sweep(0.0, Iref_fine, "mc0")
    # locate onset of period-2 and the duty there
    onset_idx = np.argmax(s0[:, 1] > 1)
    print(f"  period-doubling onset at Iref={s0[onset_idx,0]:.2f}, "
          f"duty there={s0[onset_idx,5]:.3f}")

    print("== Critical slope theory at a chaotic operating point ==")
    pchaos = dict(PARAMS); pchaos["Iref"] = 2.4
    m1, m2, mc_crit, duty, Vo = critical_slope_theory(pchaos)
    print(f"  Iref=2.4: duty={duty:.3f} Vo={Vo:.2f}V m1={m1:.1f} m2={m2:.1f} "
          f"mc_crit={mc_crit:.1f} A/s")

    # choose compensation a bit above critical for the chaotic point
    mc_use = round(1.2*mc_crit, -2)   # round to hundreds
    print(f"== Compensated sweep (mc={mc_use:.0f} A/s) ==")
    s1 = bifurcation_sweep(mc_use, Iref_fine, "mcStab")
    onset1 = s1[np.argmax(s1[:, 1] > 1), 0] if np.any(s1[:, 1] > 1) else np.nan
    print(f"  compensated period-doubling onset at Iref={onset1:.2f}")

    with open(os.path.join(RES, "compensation_used.txt"), "w") as f:
        f.write(f"mc_use={mc_use}\nmc_crit={mc_crit}\nVo={Vo}\nduty={duty}\n"
                f"m1={m1}\nm2={m2}\n")

    print("== Stability boundary vs mc ==")
    mc_vals = np.arange(0.0, mc_use*1.4+1, max(50, mc_use/12))
    sb = stability_boundary(mc_vals, np.arange(0.7, 3.001, 0.02))
    for mc, onset in sb:
        print(f"  mc={mc:8.1f} A/s  -> period-2 onset Iref={onset}")

    print("DONE. CSVs in results/")
