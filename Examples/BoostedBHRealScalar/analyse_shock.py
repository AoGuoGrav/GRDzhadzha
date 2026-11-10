"""Accretion, density and Mach-number analysis of a BoostedBH scalar run.

usage:
    python analyse_shock.py RUNDIR/params.txt [--out DIR] [--maps-every N]
                            [--maps-steps 2400,4800] [--no-hdf5]
                            [--shock auto|on|off] [--avg-from T]
                            [--evo-times 600,1200,1800]

All physical and grid parameters are read from the params file, so this works
for the real (phi, Pi) and the complex (phi_Re, phi_Im, Pi_Re, Pi_Im) examples
and for a time-dependent BH velocity (bh_velocity -> bh_velocity_final ramp).
The BH velocity of each snapshot is used to go to the BH frame, so the Mach
number M = |u| / c_s is the instantaneous one.

Regime: the healing length xi = 1/(mu c_s0) is compared with r_sg = 2M/c_s0^2.
For xi/r_sg < 0.3 (fluid regime, e.g. mu M = 1) the upstream shock of Boudon et
al. is searched for. Above that (dispersive regime, e.g. mu M = 0.1) quantum
pressure smears it out, the supersonic flow instead carries Bogoliubov-Cherenkov
ripples standing upstream of the BH with wavelength lambda_B(v), and the shock
columns are left empty (--shock on forces the search, --shock off skips it).

Outputs (in --out, default RUNDIR/analysis):
  timeseries.png / timeseries.txt   accretion rate, fluxes, energy balance
  accretion_vs_mach.png             accretion rate vs instantaneous M0(t)
  shock_vs_time.png / .txt          upstream shock position vs t and M0(t), and
                                    the measured vs predicted ripple wavelength
  maps_<step>.png, axis_<step>.png  rho/rho0 and Mach maps on the z=0 plane,
                                    near zone (+-3.5 r_sg)
  maps_full_<step>.png              the same maps on the whole domain (level 0)
  evolution.png                     whole-domain rho/rho0 (top) and Mach
                                    (bottom) at a few times, left to right
  subsonic.png                      the same at 12 times while the BH is
                                    still subsonic (two strips)
  cone_formation.png                the same at 12 times from M0 = 0.9 to a
                                    quarter ramp after the ramp ends (two
                                    strips), when the run crosses Mach 1
  late_average.png / .npz           level-0 maps averaged over the snapshots at
                                    t >= --avg-from (default: the last 2/3 of
                                    the constant-velocity phase)
  cone_profile.png / .txt           downstream <rho/rho0>(theta) in radial shells:
                                    measured vs predicted Mach-cone angle
  axis_profiles.npz                 rho/rho0 and Mach along the upstream axis

Single snapshots alias the 2 omega0 oscillation of rho unless the plot cadence
is a multiple of pi/omega0; the script says when it is not and which
plot_interval would sample a fixed phase.

Density: local oscillator energy rho = 1/2 Pi^2 + V (plus |.|^2 for complex),
without gradient energy and metric factors, so it is unreliable for r < ~10M.
Velocity: v_DM = -grad(theta)/omega with theta = arg(phi - i Pi/omega) (real)
or arg(Phi) (complex), in the DM frame; BH frame u = v_DM - v_BH(t) x_hat.
"""
import argparse
import glob
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

F_STAR = 0.66  # Boudon et al. radial accretion constant (real field)
FLUID_MAX = 0.3  # xi / r_sg below which the flow is fluid-like (shocks)


def read_params(fn):
    params = {}
    for line in open(fn):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, val = line.split("=", 1)
            params[key.strip()] = val.split()
    return params


class Run:
    def __init__(self, params_file):
        self.params = read_params(params_file)
        self.rundir = os.path.dirname(os.path.abspath(params_file))
        out = self.text("output_path", "")
        base = out if os.path.isabs(out) else os.path.join(self.rundir, out)
        self.hdf5_dir = os.path.join(base, self.text("hdf5_subpath", "hdf5"))
        self.data_dir = os.path.join(base, self.text("data_subpath", "data"))
        self.plot_prefix = self.text("plot_prefix", "BoostedBHp_")

        self.mu = self.num("scalar_mass", 0.2)
        self.phi0 = self.num("scalar_amplitude", 0.1)
        self.lam = self.num("scalar_lambda", 0.0)
        self.M = self.num("bh_mass", 1.0)
        self.v_i = self.num("bh_velocity", 0.0)
        self.v_f = self.num("bh_velocity_final", self.v_i)
        self.ramp_start = self.num("ramp_start", 0.0)
        self.ramp_duration = self.num("ramp_duration", 0.0)
        if self.lam <= 0:
            raise SystemExit("scalar_lambda must be > 0 for a sound speed")
        self.field = None  # set from the plot files or the data

    def text(self, key, default):
        return self.params[key][0].strip('"') if key in self.params else default

    def num(self, key, default):
        return float(self.params[key][0]) if key in self.params else default

    def set_field(self, field):
        self.field = field
        mu, lam, phi0 = self.mu, self.lam, self.phi0
        x0 = lam * phi0**2 / mu**2
        if field == "real":
            self.rho0 = 0.5 * mu**2 * phi0**2 + 0.25 * lam * phi0**4
            self.cs0 = float(self.cs_real(x0))
            self.omega0 = mu * math.sqrt(1 + 0.75 * x0)
        else:
            self.rho0 = 0.5 * (2 + x0) * mu**2 * phi0**2 + 0.25 * lam * phi0**4
            self.cs0 = math.sqrt(x0 / (2 + 3 * x0))
            self.omega0 = mu * math.sqrt(1 + x0)
        self.rsg = 2 * self.M / self.cs0**2
        self.Mdot_rad = 12 * math.pi * F_STAR * self.rho0 * self.M**2 / self.cs0**2
        self.M_t = (3 * F_STAR * self.cs0) ** (-1.0 / 3.0)
        self.xi = 1 / (mu * self.cs0)  # healing length
        self.fluid = self.xi / self.rsg < FLUID_MAX
        # coupling g of the linearised (rotating-wave) KG equation around the
        # homogeneous oscillation, (k^2 - W^2)^2 + 2 g (k^2 - W^2) = 4 omega0^2 W^2,
        # fixed so that its k -> 0 sound speed g / (2 omega0^2 + g) is c_s0^2
        self.g_bog = 2 * self.omega0**2 * self.cs0**2 / (1 - self.cs0**2)

    def regime(self):
        kind = "fluid" if self.fluid else "dispersive"
        return f"{kind} regime: xi = 1/(mu c_s0) = {self.xi:.1f} M, xi/r_sg = {self.xi / self.rsg:.2f}"

    def lambda_B(self, v):
        """wavelength (on the grid, i.e. in the DM frame) of the Bogoliubov
        ripples that stand still in the frame of a BH moving at v: phase
        velocity W/k = v on the axis. Exists only for v > c_s0, else nan"""
        v = float(v)
        q = 4 * self.omega0**2 * v**2 - 2 * self.g_bog * (1 - v**2)
        return 2 * math.pi * (1 - v**2) / math.sqrt(q) if q > 0 else np.nan

    def mach_angle(self, v):
        """Mach half-angle asin(c_s0/v) in degrees, nan when subsonic. The grid
        measures lengths in the DM rest frame (the boosted metric is length-
        contracted along x and the far DM is at rest on it), where sound moves
        isotropically at c_s0, so this is exact on the maps; the same cone in
        the BH frame would be narrower, asin(gamma_s c_s0 / (gamma_v v))"""
        v = float(v)
        return math.degrees(math.asin(self.cs0 / v)) if v > self.cs0 else np.nan

    @staticmethod
    def cs_real(x):
        """sound speed of the real field: nonrelativistic sqrt(3x/8) with the
        relativistic correction fitted from 1D runs (valid for c_NR < ~0.3)"""
        cnr = np.sqrt(3 * x / 8)
        return cnr * (1 - 1.4 * cnr**2)

    def v_bh(self, t):
        """the C^2 velocity ramp of SimulationParameters::bh_velocity_at"""
        t = np.asarray(t, dtype=float)
        if self.ramp_duration <= 0:
            return np.full_like(t, self.v_i)
        s = np.clip((t - self.ramp_start) / self.ramp_duration, 0, 1)
        return self.v_i + (self.v_f - self.v_i) * s**3 * (10 - 15 * s + 6 * s**2)

    def Mdot_HL(self, v):
        return 4 * math.pi * self.rho0 * self.M**2 / np.asarray(v) ** 3

    def derived(self, c, dx, v_bh):
        """rho/rho0 and BH-frame Mach number on a 2D (y, x) patch"""
        mu, lam = self.mu, self.lam
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.field == "real":
                phi, Pi = c["phi"], c["Pi"]
                rho = 0.5 * Pi**2 + 0.5 * mu**2 * phi**2 + 0.25 * lam * phi**4
                A2 = (np.sqrt(mu**4 + 4 * lam * rho) - mu**2) / lam
                om = np.sqrt(mu**2 + 0.75 * lam * A2)
                z = phi - 1j * Pi / om
                cs = self.cs_real(lam * A2 / mu**2)
            else:
                z = c["phi_Re"] + 1j * c["phi_Im"]
                A2 = np.abs(z) ** 2
                rho = (
                    0.5 * (c["Pi_Re"] ** 2 + c["Pi_Im"] ** 2 + mu**2 * A2)
                    + 0.25 * lam * A2**2
                )
                om = np.sqrt(mu**2 + lam * A2)
                xl = lam * A2 / mu**2
                cs = np.sqrt(xl / (2 + 3 * xl))
            dzy, dzx = np.gradient(z, dx)
            az2 = np.abs(z) ** 2
            ux = -np.imag(np.conj(z) * dzx) / az2 / om - v_bh
            uy = -np.imag(np.conj(z) * dzy) / az2 / om
            mach = np.hypot(ux, uy) / cs
        return rho / self.rho0, mach


# ---------------------------------------------------------------- time series
def running_mean(a, w):
    """centred boxcar mean, nan where the window does not fit: padding an
    oscillating signal with its end values puts spikes at both ends"""
    w = max(1, int(w))
    out = np.full(len(a), np.nan)
    if len(a) >= w:
        out[w // 2 : w // 2 + len(a) - w + 1] = np.convolve(
            a, np.ones(w) / w, mode="valid"
        )
    return out


def smooth_edge(a, w):
    """centred boxcar mean padded with the end values (for spatial profiles)"""
    w = max(1, int(w))
    pad = np.pad(a, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def timeseries(run, out):
    E = np.loadtxt(os.path.join(run.data_dir, "EnergyIntegrals.dat"))
    F = np.loadtxt(os.path.join(run.data_dir, "FluxIntegrals.dat"))
    n = min(len(E), len(F))
    E, F = E[:n], F[:n]
    t = E[:, 0]
    Esh, Psh, Psrc = E[:, 1], E[:, 2], E[:, 3]
    v = E[:, 4] if E.shape[1] > 4 else run.v_bh(t)
    FE_in, FP_in, FE_out, FP_out = F[:, 1], F[:, 2], F[:, 3], F[:, 4]
    radii = run.params.get("extraction_radii", ["?", "?"])
    dt = t[1] - t[0]
    M0 = v / run.cs0

    # average over a few oscillation half-periods, and over ~r_sg/(4 c_s0)
    w_osc = round(4 * math.pi / run.omega0 / dt)
    w_long = round(max(4 * math.pi / run.omega0, 0.25 * run.rsg / run.cs0) / dt)
    Mdot = running_mean(-FE_in, w_osc)
    Mdot_long = running_mean(-FE_in, w_long)

    # energy balance in the shell: dE/dt = F(r_in) - F(r_out), outward fluxes
    dF = FE_in - FE_out
    cum = np.concatenate([[0], np.cumsum(0.5 * (dF[1:] + dF[:-1]) * dt)])
    resid = (Esh - Esh[0]) - cum

    with open(os.path.join(out, "timeseries.txt"), "w") as f:
        f.write(
            f"# {run.field}: rho0={run.rho0:.6g} cs0={run.cs0:.4f} "
            f"rsg={run.rsg:.1f} Mdot_rad(F*={F_STAR})={run.Mdot_rad:.3f} "
            f"M_t={run.M_t:.3f}\n"
            f"# {run.regime()}\n"
            f"# energy-balance residual at end: {resid[-1]:.4g} "
            f"({resid[-1] / Esh[0]:.2e} of E0)\n"
            "# t  v_BH  M0  Mdot(r_in)  inflow(r_out)  Pflux(r_in)  Psource"
            f"   (window-averaged; r_in={radii[0]}, r_out={radii[1]})\n"
        )
        win = max(50.0, round(t[-1] / 40 / 50) * 50)
        for a in np.arange(0, t[-1], win):
            m = (t >= a) & (t < a + win)
            if m.any():
                f.write(
                    "%8.1f %7.4f %6.3f %10.3f %10.3f %10.3f %10.3f\n"
                    % (
                        t[m].mean(),
                        v[m].mean(),
                        M0[m].mean(),
                        -FE_in[m].mean(),
                        -FE_out[m].mean(),
                        FP_in[m].mean(),
                        Psrc[m].mean(),
                    )
                )

    fig, ax = plt.subplots(5, 1, figsize=(8, 13), sharex=True)
    ax[0].plot(t, M0, color="k")
    ax[0].axhline(1, color="C3", ls=":")
    ax[0].axhline(run.M_t, color="C2", ls=":", label=f"M_t = {run.M_t:.2f}")
    ax[0].set_ylabel("M0 = v_BH / c_s0")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, -FE_in, lw=0.3, color="0.75", label="raw")
    ax[1].plot(t, Mdot, color="C0", label="running mean")
    ax[1].axhline(run.Mdot_rad, ls="--", color="C3", label="Mdot_rad (F*=0.66)")
    ax[1].plot(t, run.Mdot_HL(v), ls=":", color="C2", label="Mdot_HL(v_BH)")
    ax[1].set_ylim(0, 2.5 * max(run.Mdot_rad, np.nanmax(Mdot)))
    ax[1].set_ylabel(f"energy inflow at r={radii[0]}")
    ax[1].legend(fontsize=8)
    ax[2].plot(t, -FE_out, lw=0.3, color="0.75")
    ax[2].plot(t, running_mean(-FE_out, w_osc), color="C1")
    ax[2].set_ylabel(f"energy inflow at r={radii[1]}")
    ax[3].plot(t, running_mean(FP_in, w_osc), label=f"x-mom flux at r={radii[0]}")
    ax[3].plot(t, running_mean(Psrc, w_osc), label="x-mom source in shell")
    ax[3].set_ylabel("momentum")
    ax[3].legend(fontsize=8)
    ax[4].plot(t, resid / Esh[0], color="C4")
    ax[4].set_ylabel("energy-balance resid / E0")
    ax[4].set_xlabel("t [M]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{run.field} field, c_s0 = {run.cs0:.3f}, r_sg = {run.rsg:.0f} M")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timeseries.png"), dpi=110)
    plt.close(fig)

    if run.ramp_duration > 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(M0, Mdot_long, color="C0", label="Mdot(r_in), running mean")
        mgrid = np.linspace(max(M0.min(), 0.3), M0.max(), 200)
        ax.plot(mgrid, run.Mdot_HL(mgrid * run.cs0), ":", color="C2", label="Mdot_HL")
        ax.axhline(run.Mdot_rad, ls="--", color="C3", label="Mdot_rad (F*=0.66)")
        ax.axvline(1, color="0.5", lw=0.8)
        ax.axvline(run.M_t, color="C2", lw=0.8, ls="--", label="M_t")
        ax.set_ylim(0, 2.5 * max(run.Mdot_rad, np.nanmax(Mdot_long)))
        ax.set_xlabel("instantaneous M0(t) = v_BH(t) / c_s0")
        ax.set_ylabel("accretion rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "accretion_vs_mach.png"), dpi=110)
        plt.close(fig)
    print(f"time series: residual/E0 at end = {resid[-1] / Esh[0]:.2e}")


# ------------------------------------------------------------------ plot files
def read_layer(fn, row_only=False):
    """patches of the z = dx/2 layer (and only the y = dx/2 row if row_only)"""
    import h5py

    f = h5py.File(fn, "r")
    ncomp = int(f.attrs["num_components"])
    names = []
    for i in range(ncomp):
        a = f.attrs["component_%d" % i]
        names.append(a.decode() if isinstance(a, bytes) else str(a))
    t, patches = None, []
    for lev in range(int(f.attrs["num_levels"])):
        g = f["level_%d" % lev]
        dx = float(g.attrs["dx"])
        t = float(g.attrs["time"])
        if lev == 0:
            dom = g.attrs["prob_domain"]
            x_extent = (dom[3] + 1) * dx
        boxes = g["boxes"][:]
        offs = g["data:offsets=0"][:]
        dset = g["data:datatype=0"]
        gh = np.zeros(3, int)
        if "data_attributes" in g and "outputGhost" in g["data_attributes"].attrs:
            gh = np.array(list(g["data_attributes"].attrs["outputGhost"]), int)
        for b, bx in enumerate(boxes):
            lo = np.array([bx["lo_i"], bx["lo_j"], bx["lo_k"]])
            hi = np.array([bx["hi_i"], bx["hi_j"], bx["hi_k"]])
            if lo[2] != 0 or (row_only and lo[1] != 0):
                continue
            nn = hi - lo + 1 + 2 * gh
            arr = dset[offs[b] : offs[b + 1]].reshape(ncomp, nn[2], nn[1], nn[0])
            sl = (slice(gh[1], nn[1] - gh[1]), slice(gh[0], nn[0] - gh[0]))
            c = {nm: arr[i, gh[2]][sl] for i, nm in enumerate(names)}
            xs = (np.arange(lo[0], hi[0] + 1) + 0.5) * dx
            ys = (np.arange(lo[1], hi[1] + 1) + 0.5) * dx
            patches.append((lev, dx, xs, ys, c))
    f.close()
    return t, patches, names, x_extent


def axis_profile(run, patches, xc, v_bh):
    """rho/rho0 and Mach along y = dx/2 (finest level wins), vs x - xc"""
    best = {}
    for lev, dx, xs, ys, c in patches:
        if ys[0] > dx:
            continue
        rr, mm = run.derived(c, dx, v_bh)
        for i, x in enumerate(xs):
            key = round(x, 6)
            if key not in best or best[key][0] < lev:
                best[key] = (lev, rr[0, i], mm[0, i])
    x = np.array(sorted(best))
    lev = np.array([best[k][0] for k in x])
    keep = np.ones(len(x), bool)
    for L in range(lev.max()):  # drop coarse points inside finer levels
        fine = x[lev > L]
        keep &= ~((lev == L) & (x > fine.min()) & (x < fine.max()))
    rr = np.array([best[k][1] for k in x])
    mm = np.array([best[k][2] for k in x])
    return x[keep] - xc, rr[keep], mm[keep]


def shock_position(run, d, rr, mm, threshold, v_bh=None, find_shock=True):
    """upstream shock from profiles smoothed over one ripple wavelength.
    The ripple wavelength is measured on the upstream axis: in the fluid
    regime near the shock (0.8-2.2 r_sg, wavelengths < 60 M, as before); in
    the dispersive regime from 0.8 r_sg over at least four predicted
    wavelengths lambda_B(v_bh), with a zero-padded FFT"""
    ok = np.isfinite(rr) & np.isfinite(mm) & (np.abs(d) > 6)
    xu = np.arange(10, d[ok].max(), 0.25)
    R = np.interp(xu, d[ok], rr[ok])
    Mc = np.interp(xu, d[ok], mm[ok])
    lam_B = run.lambda_B(v_bh) if v_bh is not None else np.nan
    long_band = not run.fluid and np.isfinite(lam_B)
    if long_band:  # kept off the upstream boundary (lambda_B -> inf as v -> c_s0)
        hi = max(2.2 * run.rsg, 0.8 * run.rsg + 4 * lam_B)
        band = (xu > 0.8 * run.rsg) & (xu < min(hi, 0.9 * xu.max()))
    else:
        band = (xu > 0.8 * run.rsg) & (xu < 2.2 * run.rsg)
    nb = band.sum()
    lam_rip = np.nan
    # (dispersive and subsonic: no standing ripples, nothing to measure)
    if (run.fluid or long_band) and nb > 16 and np.ptp(R[band]) > 1e-3:
        y = (R[band] - R[band].mean()) * np.hanning(nb)
        nfft = 8 * nb if long_band else nb  # zero padding: interpolated peak
        k = np.fft.rfftfreq(nfft, 0.25)
        P = np.abs(np.fft.rfft(y, nfft)) ** 2
        sel = k > (2 / (0.25 * nb) if long_band else 1 / 60)
        lam_rip = 1 / k[sel][np.argmax(P[sel])]
    far = xu > 0.85 * xu.max()
    m_far = np.median(Mc[far])
    r_rho = r_mach = np.nan
    # no shock for a subsonic upstream (only smooth compression), and none
    # looked for in the dispersive regime, where the upstream ripples would
    # trigger the threshold
    if find_shock and m_far > 1:
        w = int(round(max(lam_rip if np.isfinite(lam_rip) else math.pi * run.xi, 8) / 0.25))
        Rs = smooth_edge(R, w)
        Ms = smooth_edge(Mc, w)
        above = np.where(Rs > threshold)[0]
        sub = np.where(Ms < 1)[0]
        r_rho = xu[above.max()] if len(above) else np.nan
        r_mach = xu[sub.max()] if len(sub) else np.nan
    return r_rho, r_mach, lam_rip, m_far, np.median(R[far])


def map_figure(run, t, patches, xc, v_bh):
    """rho/rho0 and Mach maps on the z = 0 plane (1300 x 560 px at dpi 100)"""
    M0 = v_bh / run.cs0
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, which in zip(axs, ("rho", "mach")):
        kw = (
            dict(cmap="viridis", vmin=0.6, vmax=2.0)
            if which == "rho"
            else dict(cmap="RdBu_r", vmin=0.0, vmax=2.0)
        )
        for lev, dx, xs, ys, c in sorted(patches, key=lambda p: p[0]):
            rr, mm = run.derived(c, dx, v_bh)
            X = (np.append(xs - dx / 2, xs[-1] + dx / 2) - xc) / run.rsg
            Y = np.append(ys - dx / 2, ys[-1] + dx / 2) / run.rsg
            for sgn in (1, -1):
                ax.pcolormesh(X, sgn * Y, rr if which == "rho" else mm, **kw)
        if M0 > 1:
            thc = math.asin(1 / M0)
            r = np.linspace(0, 4, 50)
            for sgn in (1, -1):
                ax.plot(-r * math.cos(thc), sgn * r * math.sin(thc), "w--", lw=1)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.set_xlabel("(x - x_BH) / r_sg   [flow comes from +x]")
        ax.set_ylabel("y / r_sg")
        sm = plt.cm.ScalarMappable(
            cmap=kw["cmap"], norm=plt.Normalize(kw["vmin"], kw["vmax"])
        )
        fig.colorbar(
            sm, ax=ax, shrink=0.8, label="rho/rho0" if which == "rho" else "Mach"
        )
    fig.suptitle(
        f"{run.field}  t = {t:.0f} M,  v_BH = {v_bh:.4f},  M0 = {M0:.2f}  "
        f"(c_s0 = {run.cs0:.3f}, r_sg = {run.rsg:.0f} M); "
        "dashed: Mach cone",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_map(run, t, patches, xc, v_bh, out, step):
    fig = map_figure(run, t, patches, xc, v_bh)
    fig.savefig(os.path.join(out, f"maps_{step:06d}.png"), dpi=100)
    plt.close(fig)


# ------------------------------------------------- whole domain and the cone
def level0_grid(run, patches, xc, v_bh):
    """rho/rho0 and Mach on the z = 0 plane of level 0, where Chombo has
    averaged the finer levels down: (x - xc, y, R[y, x], M[y, x])"""
    p0 = [p for p in patches if p[0] == 0]
    dx = p0[0][1]
    nx = int(round(max(p[2][-1] for p in p0) / dx + 0.5))
    ny = int(round(max(p[3][-1] for p in p0) / dx + 0.5))
    R = np.full((ny, nx), np.nan)
    M = np.full((ny, nx), np.nan)
    for _, dx, xs, ys, c in p0:
        rr, mm = run.derived(c, dx, v_bh)
        i0, j0 = int(round(xs[0] / dx - 0.5)), int(round(ys[0] / dx - 0.5))
        R[j0 : j0 + len(ys), i0 : i0 + len(xs)] = rr
        M[j0 : j0 + len(ys), i0 : i0 + len(xs)] = mm
    return (np.arange(nx) + 0.5) * dx - xc, (np.arange(ny) + 0.5) * dx, R, M


MACH_NORM = (0.3, 1.0, 2.5)  # (min, white, max) of the whole-domain Mach maps


def mach_kw(center=MACH_NORM):
    """RdBu_r Mach colour map with white at center[1] (M = 1): blue is
    subsonic, red supersonic"""
    from matplotlib.colors import TwoSlopeNorm

    return dict(cmap="RdBu_r", norm=TwoSlopeNorm(center[1], center[0], center[2]))


def draw_cone(ax, run, v_bh, length):
    """the Mach cone asin(c_s0/v) from the BH downstream, dashed"""
    th = run.mach_angle(v_bh)
    if np.isfinite(th):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        for sgn in (1, -1):
            ax.plot([0, -length * c], [0, sgn * length * s], "w--", lw=1)


def full_map_figure(run, x, y, R, M, v_bh, title, rho_range=(0.8, 1.4)):
    """level-0 rho/rho0 and Mach maps of the whole z = 0 plane (mirrored in y)"""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    kws = (dict(cmap="viridis", vmin=rho_range[0], vmax=rho_range[1]), mach_kw())
    labs = ("rho/rho0", "Mach (BH frame), white: M = 1")
    for ax, A, kw, lab in zip(axs, (R, M), kws, labs):
        for sgn in (1, -1):
            im = ax.pcolormesh(x, sgn * y, A, shading="auto", **kw)
        fig.colorbar(im, ax=ax, shrink=0.8, label=lab)
        draw_cone(ax, run, v_bh, max(-x[0], x[-1]))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-y[-1], y[-1])
        ax.set_aspect("equal")
        ax.set_xlabel("x - x_BH [M]   [flow comes from +x]")
        ax.set_ylabel("y [M]")
    fig.suptitle(
        title + "; dashed: Mach cone asin(c_s0/v)", fontsize=9
    )
    fig.tight_layout()
    return fig


def cone_profile(run, out, x, y, R, v_bh):
    """downstream <rho/rho0>(theta), theta measured from the -x axis, in three
    radial shells: angle of the density peak (the smeared Mach cone of the
    linear wake) and where the excess over rho0 falls to half of the peak's"""
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    th = np.degrees(np.arctan2(Y, -X))
    rmax = min(-x[0], y[-1])
    edges = np.array([0.25, 0.47, 0.68, 0.9]) * rmax
    bins = np.arange(0, 92.5, 2.5)
    thc = 0.5 * (bins[1:] + bins[:-1])
    pred = run.mach_angle(v_bh)
    cols, summary = [thc], []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r0, r1 in zip(edges[:-1], edges[1:]):
        shell = (r > r0) & (r < r1) & np.isfinite(R)
        prof = np.array(
            [
                R[shell & (th >= a) & (th < b)].mean()
                if (shell & (th >= a) & (th < b)).any()
                else np.nan
                for a, b in zip(bins[:-1], bins[1:])
            ]
        )
        cols.append(prof)
        ps = smooth_edge(np.nan_to_num(prof, nan=1.0), 3)
        inner = thc < 80
        i_pk = int(np.argmax(np.where(inner, ps, -np.inf)))
        half = 1 + 0.5 * (ps[i_pk] - 1)
        beyond = np.where((np.arange(len(ps)) > i_pk) & (ps < half))[0]
        th_half = thc[beyond[0]] if len(beyond) else np.nan
        summary.append((r0, r1, thc[i_pk], ps[i_pk], th_half))
        ax.plot(thc, prof, label=f"r = {r0:.0f}-{r1:.0f} M")
    ax.axvline(pred, color="k", ls="--", lw=0.8)
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_xlabel("angle from the downstream axis [deg]")
    ax.set_ylabel("<rho/rho0>")
    ax.set_title(
        f"v_BH = {v_bh:.4f}: Mach angle asin(c_s0/v) = {pred:.1f} deg (dashed)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cone_profile.png"), dpi=110)
    plt.close(fig)
    head = (
        f"{run.field}: v_BH = {v_bh:.4f}, M0 = {v_bh / run.cs0:.3f}, Mach angle "
        f"asin(c_s0/v) = {pred:.2f} deg (DM rest frame = grid lengths)\n"
        + "".join(
            f"shell r = {a:.0f}-{b:.0f} M: density peak at {p:.1f} deg "
            f"(<rho/rho0> = {h:.3f}), half of the excess at {q:.1f} deg\n"
            for a, b, p, h, q in summary
        )
        + "theta  <rho/rho0> per shell"
    )
    np.savetxt(os.path.join(out, "cone_profile.txt"), np.array(cols).T, fmt="%8.3f", header=head)
    for a, b, p, h, q in summary:
        print(
            f"cone: r = {a:.0f}-{b:.0f} M, density peak at {p:.1f} deg, half excess "
            f"at {q:.1f} deg (Mach angle {pred:.1f} deg)"
        )


def late_average(run, out, snaps, avg_from):
    """level-0 maps averaged over the snapshots at t >= avg_from (default: the
    last 2/3 of the constant-velocity phase after the ramp), which removes the
    aliased 2 omega0 oscillation and the transients, and the cone profile"""
    t_last = snaps[-1][1]
    t_end = run.ramp_start + run.ramp_duration if run.ramp_duration > 0 else 0.0
    if avg_from is None:
        if t_last <= t_end:
            print("late average: skipped, the velocity ramp has not finished")
            return
        avg_from = t_end + (t_last - t_end) / 3
    use = [(fn, t) for fn, t in snaps if t >= avg_from]
    if len(use) < 2:
        print(f"late average: skipped, < 2 snapshots at t >= {avg_from:.0f}")
        return
    vs = run.v_bh([t for _, t in use])
    if np.ptp(vs) > 1e-3 * np.abs(vs).max():
        print("late average: warning, v_BH changes over the averaging window")
    accR = accM = None
    for fn, _ in use:
        t, patches, _, x_extent = read_layer(fn)
        xc = run.num("center", 0.5 * x_extent)
        x, y, R, M = level0_grid(run, patches, xc, float(run.v_bh(t)))
        accR = R if accR is None else accR + R
        accM = M if accM is None else accM + M
    R, M = accR / len(use), accM / len(use)
    v = float(vs[-1])
    ts = np.array([t for _, t in use])
    np.savez(os.path.join(out, "late_average.npz"), x=x, y=y, rho=R, mach=M, t=ts, v=v)
    title = (
        f"{run.field}: level 0 averaged over {len(use)} snapshots, "
        f"t = {ts[0]:.0f}-{ts[-1]:.0f} M, v_BH = {v:.4f}, M0 = {v / run.cs0:.2f}"
    )
    fig = full_map_figure(run, x, y, R, M, v, title, rho_range=(0.85, 1.3))
    fig.savefig(os.path.join(out, "late_average.png"), dpi=90)
    plt.close(fig)
    if v > run.cs0:
        cone_profile(run, out, x, y, R, v)


def evolution_figure(
    run, out, snaps, evo_times=None, ncols=6, per_row=6, name="evolution.png",
    label="", mach_center=None, rho_contours=None,
):
    """one figure of whole-domain (level 0) maps at a few times, left to right:
    rho/rho0 above Mach for each time, wrapped into strips of per_row times.
    The given times, or ncols times evenly spaced up to the last snapshot
    (nearest snapshots). mach_center = (vmin, 1, vmax) of the Mach scale
    (default MACH_NORM, white at M = 1); rho_contours adds white rho/rho0
    contours (shape of a saturated core)"""
    ts = np.array([t for _, t in snaps])
    if evo_times is None or len(evo_times) == 0:
        evo_times = ts[-1] * np.arange(1, ncols + 1) / ncols
    idx = sorted({int(np.argmin(np.abs(ts - tt))) for tt in evo_times})
    n = len(idx)
    nstrip, ncol = -(-n // per_row), min(n, per_row)
    kws = (dict(cmap="viridis", vmin=0.8, vmax=1.4), mach_kw(mach_center or MACH_NORM))
    fig, axs = plt.subplots(
        2 * nstrip, ncol, figsize=(2.9 * ncol + 1.3, 3.2 * 2 * nstrip),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    ims = [None, None]
    for k, i in enumerate(idx):
        s, c = divmod(k, per_row)
        t, patches, _, x_extent = read_layer(snaps[i][0])
        xc = run.num("center", 0.5 * x_extent)
        v = float(run.v_bh(t))
        x, y, R, M = level0_grid(run, patches, xc, v)
        for row, A in enumerate((R, M)):
            ax = axs[2 * s + row, c]
            for sgn in (1, -1):
                ims[row] = ax.pcolormesh(x, sgn * y, A, shading="auto", **kws[row])
                if row == 0 and rho_contours:
                    ax.contour(x, sgn * y, A, levels=rho_contours, colors="w", linewidths=0.5)
            draw_cone(ax, run, v, max(-x[0], x[-1]))
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(-y[-1], y[-1])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
        axs[2 * s, c].set_title(
            f"t = {t:.0f} M\nv = {v:.3f}, M0 = {v / run.cs0:.2f}", fontsize=9
        )
    for k in range(n, nstrip * per_row):  # empty slots of the last strip
        s, c = divmod(k, per_row)
        axs[2 * s, c].set_visible(False)
        axs[2 * s + 1, c].set_visible(False)
    for r in range(2 * nstrip):
        axs[r, 0].set_ylabel(("rho/rho0" if r % 2 == 0 else "Mach") + "\ny [M]", fontsize=9)
    for c in range(ncol):
        axs[-1, c].set_xlabel("x - x_BH [M]", fontsize=9)
    for s in range(nstrip):  # one bar per row: bars over non-adjacent rows overlap
        fig.colorbar(ims[0], ax=axs[2 * s, :], shrink=0.9, label="rho/rho0")
        fig.colorbar(
            ims[1], ax=axs[2 * s + 1, :], shrink=0.9,
            label="Mach (BH frame), white: M = 1",
        )
    extra = ""
    if rho_contours:
        extra += "; white lines: rho/rho0 = " + ", ".join(f"{c:g}" for c in rho_contours)
    fig.suptitle(
        f"{label}{run.field}: whole domain (level 0, z = 0), flow from +x; "
        f"dashed: Mach cone asin(c_s0/v); c_s0 = {run.cs0:.3f}{extra}",
        fontsize=10,
    )
    fig.savefig(os.path.join(out, name), dpi=110)
    plt.close(fig)
    print(f"{name}: t =", ", ".join(f"{ts[i]:.0f}" for i in idx))


def formation_times(run, t_last, n=12):
    """n times from M0 = 0.9 (just before the BH goes supersonic) to a
    quarter of the ramp duration after the ramp ends, or None if the run
    does not cross Mach 1"""
    if run.ramp_duration <= 0 or not (run.v_i < run.cs0 < run.v_f):
        return None
    tt = np.linspace(run.ramp_start, run.ramp_start + run.ramp_duration, 2001)
    t0 = tt[np.argmax(run.v_bh(tt) >= max(0.9 * run.cs0, run.v_i))]
    t1 = min(t_last, run.ramp_start + 1.25 * run.ramp_duration)
    return np.linspace(t0, t1, n) if t1 > t0 else None


def subsonic_times(run, ts, n=12):
    """n times from the first snapshot after t = 0 to the last snapshot
    where the BH is still subsonic, or None if there are fewer than two"""
    ts = np.asarray(ts)
    sub = ts[(ts > 0) & (run.v_bh(ts) < run.cs0)]
    return np.linspace(sub[0], sub[-1], n) if len(sub) > 1 else None


def cadence_note(run, steps, times):
    """warn when the plot cadence aliases the 2 omega0 oscillation of rho"""
    if len(times) < 2:
        return ""
    dt_plot = float(np.median(np.diff(times)))
    P = math.pi / run.omega0
    frac = (dt_plot / P) % 1
    if min(frac, 1 - frac) < 0.1:
        return ""
    i = int(np.argmax(np.asarray(steps) > 0))
    dt0 = times[i] / steps[i]
    n0 = round(dt_plot / P)  # stay close to the present cadence
    best = min(
        (abs(round(n * P / dt0) * dt0 / P - n), round(n * P / dt0), n)
        for n in range(max(1, n0 - 2), n0 + 3)
    )
    return (
        f"plot files every {dt_plot:g} M = {dt_plot / P:.2f} periods of rho "
        f"(pi/omega0 = {P:.1f} M), so single snapshots alias its 2 omega0 "
        f"oscillation; plot_interval = {best[1]} ({best[2]} periods) would "
        f"sample a fixed phase"
    )


def snapshots(
    run, out, maps_every, maps_steps, threshold, shock="auto", avg_from=None,
    evo_times=None,
):
    files = sorted(
        glob.glob(os.path.join(run.hdf5_dir, run.plot_prefix + "*.3d.hdf5"))
    )
    if not files:
        print("no plot files found in", run.hdf5_dir)
        return
    rows, profiles, snaps = [], {}, []
    find_shock = None
    for n, fn in enumerate(files):
        step = int(fn.split("_")[-1].split(".")[0])
        t, patches, names, x_extent = read_layer(fn, row_only=True)
        if run.field is None:
            run.set_field("complex" if "phi_Re" in names else "real")
        if find_shock is None:
            find_shock = shock == "on" or (shock == "auto" and run.fluid)
            print(run.regime() + ("" if find_shock else ": no shock search"))
        snaps.append((fn, t))
        xc = run.num("center", 0.5 * x_extent)
        v = float(run.v_bh(t))
        d, rr, mm = axis_profile(run, patches, xc, v)
        profiles[f"s{step}"] = np.array([d, rr, mm])
        r_rho, r_mach, lam_rip, m_far, rho_far = shock_position(
            run, d, rr, mm, threshold, v, find_shock
        )
        rows.append(
            (step, t, v, v / run.cs0, r_rho, r_mach, lam_rip, m_far, rho_far, run.lambda_B(v))
        )
        last = n == len(files) - 1
        if (maps_steps and step in maps_steps) or (
            not maps_steps and maps_every > 0 and (n % maps_every == 0 or last)
        ):
            t, patches, _, _ = read_layer(fn)
            plot_map(run, t, patches, xc, v, out, step)
            x, y, R, M = level0_grid(run, patches, xc, v)
            fig = full_map_figure(
                run, x, y, R, M, v,
                f"{run.field}  t = {t:.0f} M, v_BH = {v:.4f}, M0 = {v / run.cs0:.2f}, level 0",
            )
            fig.savefig(os.path.join(out, f"maps_full_{step:06d}.png"), dpi=90)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(d / run.rsg, rr, label="rho/rho0")
            ax.plot(d / run.rsg, mm, label="Mach")
            ax.axhline(1, color="0.6", lw=0.6)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(0, 3)
            ax.set_xlabel("(x - x_BH)/r_sg along the axis   [upstream = +]")
            ax.set_title(f"t = {t:.0f} M, M0 = {v / run.cs0:.2f}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"axis_{step:06d}.png"), dpi=100)
            plt.close(fig)
    np.savez(os.path.join(out, "axis_profiles.npz"), **profiles)

    rows = np.array(rows)
    note = cadence_note(run, rows[:, 0], rows[:, 1])
    if note:
        print("note:", note)
    shock_line = (
        f"shock = outermost upstream point where the profile smoothed over "
        f"one ripple wavelength has rho/rho0 > {threshold} (or Mach < 1)"
        if find_shock
        else "no shock search in this regime (--shock on forces it): shock columns are nan"
    )
    np.savetxt(
        os.path.join(out, "shock_vs_time.txt"),
        rows,
        fmt="%7d %8.1f %7.4f %6.3f %8.2f %8.2f %7.2f %7.3f %7.4f %8.2f",
        header=(
            f"{run.field}: cs0={run.cs0:.4f} rsg={run.rsg:.2f} M_t={run.M_t:.3f}\n"
            f"{run.regime()}\n{shock_line}\n"
            "ripple_lambda: measured on the upstream axis; lambda_B: Bogoliubov "
            "ripples standing in the BH frame (nan if subsonic)\n"
            + (f"note: {note}\n" if note else "")
            + "step t v_BH M0 shock_rho shock_mach ripple_lambda Mach_far rho_far lambda_B"
        ),
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if find_shock:
        ax.plot(rows[:, 1], rows[:, 4] / run.rsg, "o-", ms=3, label="shock (<rho>)")
        ax.plot(rows[:, 1], rows[:, 5] / run.rsg, "s", ms=3, alpha=0.6, label="shock (<M>=1)")
        ax.set_ylabel("upstream shock position / r_sg")
    else:
        ax.plot(rows[:, 1], rows[:, 6], "o", ms=3, label="ripple wavelength, upstream axis")
        ax.plot(rows[:, 1], rows[:, 9], color="C1", label="lambda_B(v_BH), Bogoliubov")
        if np.isfinite(rows[:, 9]).any():  # lambda_B -> inf as M0 -> 1
            ax.set_ylim(0, 3 * np.nanmin(rows[:, 9]))
        ax.set_ylabel("wavelength [M]")
        ax.set_title(run.regime() + ": no shock search", fontsize=9)
    ax.set_xlabel("t [M]")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(rows[:, 1], rows[:, 3], color="k", lw=1, label="M0(t)")
    ax2.axhline(1, color="C3", ls=":", lw=0.8)
    ax2.set_ylabel("M0(t)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "shock_vs_time.png"), dpi=110)
    plt.close(fig)
    print(f"snapshots: {len(files)} plot files, last t = {rows[-1, 1]:.0f}")
    late_average(run, out, snaps, avg_from)
    evolution_figure(run, out, snaps, evo_times)
    times = subsonic_times(run, [t for _, t in snaps])
    if times is not None:
        evolution_figure(
            run, out, snaps, times, name="subsonic.png", label="Subsonic phase, ",
            mach_center=(0.3, 1.0, 1.7), rho_contours=[1.1, 1.5, 2, 3],
        )
    times = formation_times(run, snaps[-1][1])
    if times is not None:
        evolution_figure(
            run, out, snaps, times, name="cone_formation.png",
            label="Mach-cone formation, ",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("params", help="params file of the run (in the run dir)")
    ap.add_argument("--out", default=None, help="output dir (RUNDIR/analysis)")
    ap.add_argument("--maps-every", type=int, default=10, help="maps every N files")
    ap.add_argument("--maps-steps", default="", help="comma-separated steps")
    ap.add_argument("--rho-threshold", type=float, default=1.15)
    ap.add_argument("--no-hdf5", action="store_true", help="time series only")
    ap.add_argument(
        "--shock",
        choices=("auto", "on", "off"),
        default="auto",
        help="upstream shock search: auto = only in the fluid regime",
    )
    ap.add_argument(
        "--avg-from",
        type=float,
        default=None,
        help="start time of the late average (default: last 2/3 after the ramp)",
    )
    ap.add_argument(
        "--evo-times",
        default="",
        help="comma-separated times of the evolution.png columns "
        "(default: 6 evenly spaced up to the end)",
    )
    args = ap.parse_args()

    run = Run(args.params)
    out = args.out or os.path.join(run.rundir, "analysis")
    os.makedirs(out, exist_ok=True)
    steps = [int(s) for s in args.maps_steps.split(",") if s]
    evo_times = [float(s) for s in args.evo_times.split(",") if s]
    if not args.no_hdf5:
        snapshots(
            run, out, args.maps_every, steps, args.rho_threshold, args.shock,
            args.avg_from, evo_times,
        )
    if run.field is None:
        # no plot files read: guess the field from the params file
        complex_vars = "phi_Re" in " ".join(run.params.get("plot_vars", []))
        run.set_field("complex" if complex_vars else "real")
    timeseries(run, out)
    print("outputs in", out)


if __name__ == "__main__":
    main()
