import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parámetros del sistema ────────────────────────────────────────────────────
m, l, g = 1.0, 1.0, 9.8
q0 = np.array([np.pi / 2, 0.0])
p0 = np.array([0.0, 0.0])
h  = 0.05
N  = 4000

# ── Hamiltoniano ──────────────────────────────────────────────────────────────
def hamiltonian(q, p):
    q1, q2 = q
    p1, p2 = p
    dq  = q1 - q2
    den = 2 * m * l**2 * (1 - 0.5 * np.cos(dq)**2)
    T   = (m * l**2 * (p1**2 + 0.5*p2**2 - p1*p2*np.cos(dq))) / den
    V   = -m * g * l * (2*np.cos(q1) + np.cos(q2))
    return T + V

# ── Derivadas (gradiente del Hamiltoniano) ────────────────────────────────────
def derivatives(q, p):
    q1, q2 = q
    p1, p2 = p
    dq  = q1 - q2
    cd, sd = np.cos(dq), np.sin(dq)
    den = 2 - cd**2

    vq1 = (2*p1 - cd*p2) / (m * l**2 * den)
    vq2 = (-cd*p1*2 + 2*p2*2 - 2*p1*cd) / (2 * m * l**2 * den)   # ∂H/∂p2
    # Corrección limpia:
    vq1 = (2*p1 - p2*cd) / (m * l**2 * den)
    vq2 = (4*p2 - 2*p1*cd) / (2 * m * l**2 * den)

    cross = (p1*p2*sd) / den - (2*p1 - p2*cd)*(2*p2 - 2*p1*cd)*sd / den**2
    dp1 = -cross - 2*m*g*l*np.sin(q1)
    dp2 =  cross -   m*g*l*np.sin(q2)

    return np.array([vq1, vq2]), np.array([dp1, dp2])

# ── Integradores ──────────────────────────────────────────────────────────────
def integrate_explicit_euler(q0, p0, h, N):
    q, p = q0.copy(), p0.copy()
    qs, ps, Hs = [q.copy()], [p.copy()], [hamiltonian(q, p)]
    for _ in range(N):
        dq, dp = derivatives(q, p)
        q = q + h * dq
        p = p + h * dp
        qs.append(q.copy()); ps.append(p.copy()); Hs.append(hamiltonian(q, p))
    return np.array(qs), np.array(ps), np.array(Hs)

def integrate_symplectic_euler(q0, p0, h, N):
    q, p = q0.copy(), p0.copy()
    qs, ps, Hs = [q.copy()], [p.copy()], [hamiltonian(q, p)]
    for _ in range(N):
        _, dp = derivatives(q, p)
        p = p + h * dp          # actualizar p primero
        dq, _ = derivatives(q, p)
        q = q + h * dq          # luego q con el p nuevo
        qs.append(q.copy()); ps.append(p.copy()); Hs.append(hamiltonian(q, p))
    return np.array(qs), np.array(ps), np.array(Hs)

# ── Simulación ────────────────────────────────────────────────────────────────
print("Simulando Euler explícito...")
qs_ee, ps_ee, Hs_ee = integrate_explicit_euler(q0, p0, h, N)
print("Simulando Euler simpléctico...")
qs_se, ps_se, Hs_se = integrate_symplectic_euler(q0, p0, h, N)

H0_ee = Hs_ee[0]
H0_se = Hs_se[0]
err_ee = np.abs(Hs_ee[1:] - H0_ee) / np.abs(H0_ee)
err_se = np.abs(Hs_se[1:] - H0_se) / np.abs(H0_se)
t = np.arange(1, N+1) * h

# ── Figura ────────────────────────────────────────────────────────────────────
COLOR_EE = "#c0392b"   # rojo — Explicit Euler
COLOR_SE = "#1a7a40"   # verde — Symplectic Euler

fig = plt.figure(figsize=(11, 7))
fig.patch.set_facecolor("white")

suptitle = (
    f"Péndulo doble — $q_0=(\\pi/2,\\,0)$, $p_0=(0,\\,0)$, "
    f"$h={h}$, $N={N}$ pasos"
)
fig.suptitle(suptitle, fontsize=11, y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32,
                       left=0.09, right=0.97, top=0.91, bottom=0.09)

ax_ph_ee = fig.add_subplot(gs[0, 0])
ax_ph_se = fig.add_subplot(gs[0, 1])
ax_en_ee = fig.add_subplot(gs[1, 0])
ax_en_se = fig.add_subplot(gs[1, 1])

# ── Retrato de fase ───────────────────────────────────────────────────────────
def plot_phase(ax, qs, ps, color, title, subtitle):
    ax.plot(qs[:, 0], ps[:, 0], color=color, lw=0.5, alpha=0.75, ls="--")
    ax.plot(qs[0, 0], ps[0, 0], "ko", ms=5, zorder=5)
    ax.set_xlabel(r"$q_1$", fontsize=10)
    ax.set_ylabel(r"$p_1$", fontsize=10, labelpad=2)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, color=color, pad=4)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

plot_phase(ax_ph_ee, qs_ee, ps_ee, COLOR_EE,
           "Euler explícito", "(no simpléctico, orden 1)")
plot_phase(ax_ph_se, qs_se, ps_se, COLOR_SE,
           "Euler simpléctico", "(simpléctico, orden 1)")

# ── Error de energía ──────────────────────────────────────────────────────────
def plot_energy(ax, t, err, color, title):
    ax.semilogy(t, err, color=color, lw=0.5, alpha=0.8, ls="--")
    ax.set_xlabel("Tiempo $t$", fontsize=10)
    ax.set_ylabel(r"$|H_n - H_0|$", fontsize=9, labelpad=2)
    ax.set_title(title, fontsize=10, color=color, pad=4)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

plot_energy(ax_en_ee, t, err_ee, COLOR_EE, "Error de energía — Euler explícito")
plot_energy(ax_en_se, t, err_se, COLOR_SE, "Error de energía — Euler simpléctico")

# ── Etiqueta de retrato de fase (eje y compartido) ────────────────────────────
for ax in [ax_ph_ee, ax_ph_se]:
    ax.set_ylabel(r"Retrato de fase$(q_1,\,p_1)$" + "\n" + r"$p_1$",
                  fontsize=8, labelpad=2)

plt.savefig("/mnt/user-data/outputs/double_pendulum_euler.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print("Figura guardada.")
plt.show()
