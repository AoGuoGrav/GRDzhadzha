"""Accretion, density and Mach-number analysis of a BoostedBH scalar run.

usage:
    python analyse_shock.py RUNDIR/params.txt [--out DIR] [--maps-every N]
                            [--maps-steps 2400,4800] [--no-hdf5]

All physical and grid parameters are read from the params file, so this works
for the real (phi, Pi) and the complex (phi_Re, phi_Im, Pi_Re, Pi_Im) examples
and for a time-dependent BH velocity (bh_velocity -> bh_velocity_final ramp).
The BH velocity of each snapshot is used to go to the BH frame, so the Mach
number M = |u| / c_s is the instantaneous one.

Outputs (in --out, default RUNDIR/analysis):
  timeseries.png / timeseries.txt   accretion rate, fluxes, energy balance
  accretion_vs_mach.png             accretion rate vs instantaneous M0(t)
  shock_vs_time.png / .txt          upstream shock position vs t and M0(t)
  maps_<step>.png, axis_<step>.png  rho/rho0 and Mach maps on the z=0 plane
  axis_profiles.npz                 rho/rho0 and Mach along the upstream axis

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


def shock_position(run, d, rr, mm, threshold):
    """upstream shock from profiles smoothed over one ripple wavelength"""
    ok = np.isfinite(rr) & np.isfinite(mm) & (np.abs(d) > 6)
    xu = np.arange(10, d[ok].max(), 0.25)
    R = np.interp(xu, d[ok], rr[ok])
    Mc = np.interp(xu, d[ok], mm[ok])
    band = (xu > 0.8 * run.rsg) & (xu < 2.2 * run.rsg)
    lam_rip = np.nan
    if band.sum() > 16 and np.ptp(R[band]) > 1e-3:
        y = (R[band] - R[band].mean()) * np.hanning(band.sum())
        k = np.fft.rfftfreq(band.sum(), 0.25)
        P = np.abs(np.fft.rfft(y)) ** 2
        sel = k > 1 / 60
        lam_rip = 1 / k[sel][np.argmax(P[sel])]
    xi = 1 / (run.mu * run.cs0)  # healing length
    w = int(round(max(lam_rip if np.isfinite(lam_rip) else math.pi * xi, 8) / 0.25))
    Rs = running_mean(R, w)
    Ms = running_mean(Mc, w)
    above = np.where(Rs > threshold)[0]
    sub = np.where(Ms < 1)[0]
    r_rho = xu[above.max()] if len(above) else np.nan
    r_mach = xu[sub.max()] if len(sub) else np.nan
    far = xu > 0.85 * xu.max()
    m_far = np.median(Mc[far])
    if m_far <= 1:  # subsonic upstream: no shock, only smooth compression
        r_rho = r_mach = np.nan
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
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="w", lw=0.6, ls=":"))
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
        "dashed: Mach cone, dotted: r = r_sg",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_map(run, t, patches, xc, v_bh, out, step):
    fig = map_figure(run, t, patches, xc, v_bh)
    fig.savefig(os.path.join(out, f"maps_{step:06d}.png"), dpi=100)
    plt.close(fig)


def snapshots(run, out, maps_every, maps_steps, threshold):
    files = sorted(
        glob.glob(os.path.join(run.hdf5_dir, run.plot_prefix + "*.3d.hdf5"))
    )
    if not files:
        print("no plot files found in", run.hdf5_dir)
        return
    rows, profiles = [], {}
    for n, fn in enumerate(files):
        step = int(fn.split("_")[-1].split(".")[0])
        t, patches, names, x_extent = read_layer(fn, row_only=True)
        if run.field is None:
            run.set_field("complex" if "phi_Re" in names else "real")
        xc = run.num("center", 0.5 * x_extent)
        v = float(run.v_bh(t))
        d, rr, mm = axis_profile(run, patches, xc, v)
        profiles[f"s{step}"] = np.array([d, rr, mm])
        r_rho, r_mach, lam_rip, m_far, rho_far = shock_position(
            run, d, rr, mm, threshold
        )
        rows.append((step, t, v, v / run.cs0, r_rho, r_mach, lam_rip, m_far, rho_far))
        last = n == len(files) - 1
        if (maps_steps and step in maps_steps) or (
            not maps_steps and maps_every > 0 and (n % maps_every == 0 or last)
        ):
            t, patches, _, _ = read_layer(fn)
            plot_map(run, t, patches, xc, v, out, step)
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
    np.savetxt(
        os.path.join(out, "shock_vs_time.txt"),
        rows,
        fmt="%7d %8.1f %7.4f %6.3f %8.2f %8.2f %7.2f %7.3f %7.4f",
        header=(
            f"{run.field}: cs0={run.cs0:.4f} rsg={run.rsg:.2f} M_t={run.M_t:.3f}\n"
            f"shock = outermost upstream point where the profile smoothed over "
            f"one ripple wavelength has rho/rho0 > {threshold} (or Mach < 1)\n"
            "step t v_BH M0 shock_rho shock_mach ripple_lambda Mach_far rho_far"
        ),
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rows[:, 1], rows[:, 4] / run.rsg, "o-", ms=3, label="shock (<rho>)")
    ax.plot(rows[:, 1], rows[:, 5] / run.rsg, "s", ms=3, alpha=0.6, label="shock (<M>=1)")
    ax.set_xlabel("t [M]")
    ax.set_ylabel("upstream shock position / r_sg")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("params", help="params file of the run (in the run dir)")
    ap.add_argument("--out", default=None, help="output dir (RUNDIR/analysis)")
    ap.add_argument("--maps-every", type=int, default=10, help="maps every N files")
    ap.add_argument("--maps-steps", default="", help="comma-separated steps")
    ap.add_argument("--rho-threshold", type=float, default=1.15)
    ap.add_argument("--no-hdf5", action="store_true", help="time series only")
    args = ap.parse_args()

    run = Run(args.params)
    out = args.out or os.path.join(run.rundir, "analysis")
    os.makedirs(out, exist_ok=True)
    steps = [int(s) for s in args.maps_steps.split(",") if s]
    if not args.no_hdf5:
        snapshots(run, out, args.maps_every, steps, args.rho_threshold)
    if run.field is None:
        # no plot files read: guess the field from the params file
        complex_vars = "phi_Re" in " ".join(run.params.get("plot_vars", []))
        run.set_field("complex" if complex_vars else "real")
    timeseries(run, out)
    print("outputs in", out)


if __name__ == "__main__":
    main()
