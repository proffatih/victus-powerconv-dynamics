"""
make_spice.py
=============
Generate an ngspice netlist for a peak current-mode controlled boost
converter and run it headless. The control loop is:

  * A 10 kHz clock generates short SET pulses at the start of every period.
  * An SR latch (built from two behavioural NOR gates with a tiny RC for a
    well-defined memory) holds the gate signal Q.
  * The switch turns ON when the latch is SET (clock pulse).
  * A comparator RESETs the latch when the *sensed* inductor current
    i_L*Rsense reaches the compensated reference  (Iref - mc*t_cyc)*Rsense,
    i.e. peak current-mode control with a compensating ramp of slope mc.

The compensating sawtooth ramp (slope mc, reset every clock period) is
generated from the same clock.

This is a genuine transient circuit simulation (real switching device model
via a voltage-controlled switch with small Ron); it is independent of the
discrete map and is used to confirm the map's period-1/2/chaos predictions.

Usage: python3 make_spice.py Iref mc outprefix [tstop]
"""
import subprocess, sys, os, tempfile

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CODE, "..", "data")
os.makedirs(DATA, exist_ok=True)

# Component values must match converter_map.PARAMS
L = 1.0e-3
C = 4.0e-6
R = 20.0
VIN = 10.0
FS = 10000.0
T = 1.0/FS
RSENSE = 1.0          # current-sense gain (V/A) -> sensed voltage = iL*Rsense


def netlist(Iref, mc, prefix, tstop):
    # compensating ramp amplitude over one period: mc*T  (A) -> *Rsense volts
    ramp_amp = mc * T * RSENSE
    raw = os.path.join(DATA, f"{prefix}.raw")
    return f"""* Peak current-mode boost converter  (Iref={Iref} A, mc={mc} A/s)
.title CMC boost Iref={Iref} mc={mc}

* ---- power stage ----
Vin  in   0   dc {VIN}
Lind in   sw  {L} ic=0
* ideal-ish MOSFET as voltage-controlled switch (gate node = q)
Sw   sw   0   q 0  swmod
Dout sw   out dmod
Cout out  0   {C} ic=0
Rload out 0   {R}

.model swmod sw(vt=0.5 vh=0.05 ron=0.01 roff=1e6)
.model dmod  d(is=1e-14 n=1 rs=0.01)

* ---- current sense: sensed voltage = i(Lind)*Rsense ----
Bsense isns 0 v = i(Lind)*{RSENSE}

* ---- clock: 10 kHz, narrow SET pulse (2% duty) ----
Vclk clk 0 PULSE(0 1 0 1n 1n {0.02*T} {T})

* ---- compensating sawtooth ramp, reset each period (slope mc) ----
* integrate (1) but reset on clock: approximate with a sawtooth via PULSE
Vramp rmp 0 PULSE(0 {ramp_amp} 0 {0.98*T} 1n 1n {T})

* ---- compensated reference: (Iref)*Rsense - ramp ----
Bref ref 0 v = {Iref*RSENSE} - v(rmp)

* ---- comparator: RESET high when sensed current exceeds reference ----
Bcmp rst 0 v = ( v(isns) > v(ref) ) ? 1 : 0

* ---- SR latch (set=clk, reset=rst), Q=q -> gate ----
* Q = NOR(reset, Qbar); Qbar = NOR(set, Q). Implement with behavioural
* sources + small RC memory to give a clean bistable.
Bq    qd  0 v = ( (v(rst) > 0.5) || (v(qbar) > 0.5) ) ? 0 : 1
Rq    qd  q 100
Cq    q   0 1n ic=0
Bqb   qbd 0 v = ( (v(clk) > 0.5) || (v(q) > 0.5) ) ? 0 : 1
Rqb   qbd qbar 100
Cqb   qbar 0 1n ic=0

.ic v(q)=1 v(qbar)=0

.control
set noaskquit
tran 50n {tstop} uic
* sample inductor current at clock instants (just before each SET) and
* write the full transient too
wrdata {raw} i(Lind) v(out) v(q)
.endc
.end
"""


def run(Iref, mc, prefix, tstop):
    nl = netlist(Iref, mc, prefix, tstop)
    netpath = os.path.join(DATA, f"{prefix}.cir")
    with open(netpath, "w") as f:
        f.write(nl)
    r = subprocess.run(["ngspice", "-b", netpath],
                       capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout, r.stderr, os.path.join(DATA, f"{prefix}.raw")


if __name__ == "__main__":
    Iref = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    mc   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    prefix = sys.argv[3] if len(sys.argv) > 3 else f"boost_I{Iref}_m{mc}"
    tstop = float(sys.argv[4]) if len(sys.argv) > 4 else 60e-3
    rc, out, err, raw = run(Iref, mc, prefix, tstop)
    print("rc", rc)
    print(out[-1500:])
    if err.strip():
        print("STDERR", err[-800:])
    print("RAW:", raw, "exists:", os.path.exists(raw))
