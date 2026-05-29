import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── System parameters ─────────────────────────────────────────────────────────
m, l, g = 1.0, 1.0, 9.8
q0 = np.pi / 2   # initial angle
p0 = 0.0         # initial momentum
h  = 0.1
N  = 4000

# ── Hamiltonian: H = p²/(2ml²) - mgl·cos(q) ─────────────────────────────────
def hamiltonian(q, p):
    return p**2 / (2 * m * l**2) - m * g * l * np.cos(q)

# ── Equations of motion: dq/dt = ∂H/∂p,  dp/dt = -∂H/∂q ─────────────────────
def derivatives(q, p):
    dqdt = p / (m * l**2)
    dpdt = -m * g * l * np.sin(q)
    return dqdt, dpdt

# ── Integrators ───────────────────────────────────────────────────────────────
def integrate_explicit_euler(q0, p0, h, N):
    q, p = q0, p0
    qs, ps, Hs = [q], [p], [hamiltonian(q, p)]
    for _ in range(N):
        dq, dp = derivatives(q, p)
        q = q + h * dq
        p = p + h * dp
        qs.append(q); ps.append(p); Hs.append(hamiltonian(q, p))
    return np.array(qs), np.array(ps), np.array(Hs)

def integrate_symplectic_euler(q0, p0, h, N):
    q, p = q0, p0
    qs, ps, Hs = [q], [p], [hamiltonian(q, p)]
    for _ in range(N):
        _, dp = derivatives(q, p)
        p = p + h * dp          # update p first
        dq, _ = derivatives(q, p)
        q = q + h * dq          # then q with new p
        qs.append(q); ps.append(p); Hs.append(hamiltonian(q, p))
    return np.array(qs), np.array(ps), np.array(Hs)

# ── Simulation ────────────────────────────────────────────────────────────────
print("Simulating Explicit Euler...")
qs_ee, ps_ee, Hs_ee = integrate_explicit_euler(q0, p0, h, N)
print("Simulating Symplectic Euler...")
qs_se, ps_se, Hs_se = integrate_symplectic_euler(q0, p0, h, N)

H0_ee = Hs_ee[0]
H0_se = Hs_se[0]
err_ee = np.abs(Hs_ee[1:] - H0_ee) / np.abs(H0_ee)
err_se = np.abs(Hs_se[1:] - H0_se) / np.abs(H0_se)
t = np.arange(1, N + 1) * h

# ── Figure ────────────────────────────────────────────────────────────────────
COLOR_EE = "#c0392b"   # red  — Explicit Euler
COLOR_SE = "#1a7a40"   # green — Symplectic Euler

fig = plt.figure(figsize=(11, 7))
fig.patch.set_facecolor("white")

suptitle = (
    f"Simple pendulum — $q_0=\\pi/2$, $p_0=0$,  "
    f"$h={h}$,  $N={N}$ steps"
)
fig.suptitle(suptitle, fontsize=11, y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.10, right=0.97, top=0.91, bottom=0.09)

ax_ph_ee = fig.add_subplot(gs[0, 0])
ax_ph_se = fig.add_subplot(gs[0, 1])
ax_en_ee = fig.add_subplot(gs[1, 0])
ax_en_se = fig.add_subplot(gs[1, 1])

# ── Phase portrait ────────────────────────────────────────────────────────────
def plot_phase(ax, qs, ps, color, title, subtitle):
    ax.plot(qs, ps, color=color, lw=0.6, alpha=0.75, ls="--")
    ax.plot(qs[0], ps[0], "ko", ms=5, zorder=5)
    ax.set_xlabel(r"$q$", fontsize=10)
    ax.set_ylabel(r"Phase portrait $(q,\,p)$" + "\n" + r"$p$", fontsize=8, labelpad=2)
    ax.set_title(f"{title}\n({subtitle})", fontsize=10, color=color, pad=4)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

plot_phase(ax_ph_ee, qs_ee, ps_ee, COLOR_EE,
           "Explicit Euler", "non-symplectic, order 1")
plot_phase(ax_ph_se, qs_se, ps_se, COLOR_SE,
           "Symplectic Euler", "symplectic, order 1")

# ── Energy error ──────────────────────────────────────────────────────────────
def plot_energy(ax, t, err, color, title):
    valid = err > 0
    ax.semilogy(t[valid], err[valid], color=color, lw=0.6, alpha=0.8, ls="--")
    ax.set_xlabel("Time $t$", fontsize=10)
    ax.set_ylabel(r"$|H_n - H_0|\,/\,|H_0|$", fontsize=9, labelpad=2)
    ax.set_title(title, fontsize=10, color=color, pad=4)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

plot_energy(ax_en_ee, t, err_ee, COLOR_EE, "Energy error — Explicit Euler")
plot_energy(ax_en_se, t, err_se, COLOR_SE, "Energy error — Symplectic Euler")

plt.savefig("/mnt/user-data/outputs/simple_pendulum_euler.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print("Figure saved.")
plt.show()
