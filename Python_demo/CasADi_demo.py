import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import time

# -------------------------
# helper: safe scalar
# -------------------------


def to_scalar(x):
    """Convert CasADi / NumPy / Python numeric types to plain Python float safely."""
    if isinstance(x, (float, int)):
        return float(x)
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    x = np.asarray(x)
    if x.size == 1:
        return float(x.reshape(-1)[0])
    raise TypeError(f"Cannot convert {type(x)} with shape {getattr(x, 'shape', None)} to scalar.")


# =========================================================
# 1) System (mass-spring-damper)
# =========================================================
m = 1.0
k = 1.0
b = 0.5

A = np.array([[0.0, 1.0],
              [-k/m, -b/m]])
B = np.array([[0.0],
              [1.0/m]])

Ts = 0.1  # consistent with typical textbook discretization
# 工程注释：
# m(kg)：质量；k(N/m)：弹簧刚度；b(N·s/m)：阻尼系数
# 连续时间线性化的二阶系统写成状态空间 (x1=位置 x, x2=速度 xdot)
# A,B 为连续时间系统矩阵，单位为 1/s（A）和 1/s^2（B 中带 m^-1），
# Ts 为采样周期，单位：秒。离散化需保持系统稳定性与数值精度。


def c2d_zoh(A, B, Ts):
    """ZOH discretization using augmented matrix exponential via eigen-decomposition (small matrices)."""
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n+m, n+m))
    M[:n, :n] = A
    M[:n, n:] = B
    eigvals, eigvecs = np.linalg.eig(M)
    expM = eigvecs @ np.diag(np.exp(eigvals * Ts)) @ np.linalg.inv(eigvecs)
    Ad = np.real(expM[:n, :n])
    Bd = np.real(expM[:n, n:])
    return Ad, Bd

# 工程注释：
# 使用 ZOH（零阶保持）离散化连续系统，通过求增广矩阵指数获得精确离散解。
# 适用于小维度系统（此处 n=2），优点是数值稳定且与理论采样保持一致。


Ad, Bd = c2d_zoh(A, B, Ts)

nx, nu = 2, 1

# =========================================================
# 2) Reference: square wave on position x1
# =========================================================


def ref_square(step, amp=0.5, period_s=2.0):
    period_steps = int(round(period_s / Ts))
    half = period_steps // 2
    return amp if (step % period_steps) < half else -amp

# 工程注释：参考信号为位置 x1 的方波，幅值 amp 单位为位置（m），
# period_s 为周期（秒），内部将其转换为步数以匹配采样周期 Ts。


# =========================================================
# 3) Constraints & weights
# =========================================================
x1_max = 1.0
x2_max = 2.0
u_max = 3.0

# MPC weights (tracking)
Q_mpc = np.diag([50.0, 1.0])
R_mpc = np.diag([0.2])
Qf_mpc = Q_mpc.copy()

# LQR weights (tube feedback) - can be different to reduce conservatism
Q_lqr = np.diag([10.0, 1.0])
R_lqr = np.diag([1.0])

N = 30  # 控制预测步数

# 工程注释（约束与权重）：
# x1_max, x2_max 分别是位置与速度的允许最大绝对值（状态约束，单位：m / m/s），
# u_max 为控制输入限幅（单位：相应输入单位，例如 N 或归一化后的无量纲数）。
# Q_mpc / R_mpc / Qf_mpc：MPC 中状态/输入/终端的二次权重矩阵，用于权衡跟踪精度与控制努力。
# Q_lqr / R_lqr：用于设计内环稳态反馈 K_tube，使误差收敛（可与 MPC 权重不同以降低保守性）。
# N：控制预测步数，应足够长以捕获系统动态但不宜过长以致求解慢。

# =========================================================
# 4) Discrete LQR (dlqr): returns K for u = -Kx
# =========================================================


def dlqr(Ad, Bd, Q, R, iters=5000, tol=1e-12):
    P = Q.copy()
    for _ in range(iters):
        S = R + Bd.T @ P @ Bd
        K = np.linalg.solve(S, (Bd.T @ P @ Ad))
        Pn = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K + Q
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    S = R + Bd.T @ P @ Bd
    K = np.linalg.solve(S, (Bd.T @ P @ Ad))
    return K, P


K_lqr, _ = dlqr(Ad, Bd, Q_lqr, R_lqr)  # u = -K_lqr x
K_tube = -K_lqr                         # u = v + K_tube (x - z)
Acl = Ad + Bd @ K_tube

# 工程注释（离散 LQR 与内环）：
# dlqr 返回离散时间下的最优线性反馈增益 K，使得 u = -K x 最小化二次代价。
# Tube-MPC 中，选择 K_tube 作为稳态收敛的 内环反馈，使误差动态 Acl = Ad + Bd*K_tube 收敛。
# rho(Acl) < 1 是必要的稳定性检查；若不满足需调整 Q_lqr/R_lqr 或减小采样周期 Ts。
rho = max(abs(np.linalg.eigvals(Acl)))
print("rho(Acl) =", rho)
if rho >= 1.0:
    raise RuntimeError("Acl unstable; adjust Q_lqr/R_lqr or discretization.")

# =========================================================
# 5) Input disturbance model & tube tightening
#    True plant: x_{k+1} = Ad x_k + Bd (u_k + d_k)
#    with |d_k| <= dmax (scalar)
#
#    Error dynamics: e_{k+1} = (Ad + Bd K_tube) e_k + Bd d_k = Acl e_k + Bd d_k
# =========================================================
dmax_init = 0.2  # unknown force bound (tune); larger => more robust but more conservative


def compute_emax_box(Acl, wmax, M=80):
    """
    Conservative componentwise bound on |e| for:
      e_{k+1} = Acl e_k + w_k, |w_k|<=wmax (box in state)
    emax <= sum_{i=0}^{M-1} |Acl|^i wmax
    """
    Aabs = np.abs(Acl)
    emax = np.zeros_like(wmax, dtype=float)
    Ai = np.eye(Aabs.shape[0])
    for _ in range(M):
        emax += (np.abs(Ai) @ wmax)
        Ai = Ai @ Aabs
        if np.any(~np.isfinite(emax)) or np.max(emax) > 1e6:
            raise RuntimeError("e_max exploded; bounds too large or Aabs not contracting.")
    return emax


def compute_emax_input_dist(Acl, Bd, dmax, M=120):
    """
    Tight componentwise bound for input disturbance:
        e_{k+1} = Acl e_k + Bd d_k, |d_k|<=dmax (scalar)
    emax <= sum_{i=0}^{M-1} |Acl^i Bd| * dmax
    """
    emax = np.zeros((Acl.shape[0],), dtype=float)
    Ai = np.eye(Acl.shape[0])
    for _ in range(M):
        col = (Ai @ Bd).reshape(-1)         # (nx,)
        emax += np.abs(col) * float(dmax)  # componentwise
        Ai = Ai @ Acl
        if np.any(~np.isfinite(emax)) or np.max(emax) > 1e6:
            raise RuntimeError("e_max exploded; check stability or bounds.")
    return emax


def tighten_or_shrink_dmax(Acl, K_tube, Bd, dmax0, xmax, u_max, M=120):
    d = float(dmax0)
    for _ in range(14):
        emax = compute_emax_input_dist(Acl, Bd, d, M=M)
        zmax = xmax - emax
        Ku_worst = float(np.sum(np.abs(K_tube[0, :]) * emax))
        vmax = u_max - Ku_worst
        if np.all(zmax > 1e-6) and vmax > 1e-6:
            w_eq = np.abs(Bd).reshape(-1) * d  # just for printing
            return d, w_eq, emax, zmax, vmax, Ku_worst
        d *= 0.5
    raise RuntimeError("Unable to find feasible tightening by shrinking dmax.")

# 工程注释（扰动与收缩逻辑）：
# 目标：给定输入扰动上界 dmax，计算由内环反馈引起的状态误差上界 e_max，
# 并基于该误差“收紧”名义轨迹的状态约束（zmax = xmax - emax）和输入约束（vmax = u_max - |K|*emax），
# 以保证对真实系统（存在扰动）施加控制时仍满足原始约束。
# tighten_or_shrink_dmax 会在初始 dmax 无法满足收紧条件时逐步将 d 减半，直至找到可行的收紧。


xmax = np.array([x1_max, x2_max])
dmax_eff, w_eq, e_max, zmax, vmax, Ku_worst = tighten_or_shrink_dmax(Acl, K_tube, Bd, dmax_init, xmax, u_max, M=80)

print("\nTube tightening (input disturbance):")
print("  dmax_eff =", dmax_eff)
print("  w_eq     =", w_eq, "  (|Bd|*dmax_eff)")
print("  e_max    =", e_max)
print("  zmax     =", zmax)
print("  Ku_worst =", Ku_worst, "=> vmax =", vmax)
print("  K_lqr (u=-Kx) =", K_lqr)
print("  K_tube (u=v+K(x-z)) =", K_tube)

# =========================================================
# 6) Build Standard MPC (Opti)
# =========================================================


def build_standard_mpc():
    opti = ca.Opti()

    X = opti.variable(nx, N+1)
    U = opti.variable(nu, N)
    X0 = opti.parameter(nx, 1)
    Rref = opti.parameter(1, N)

    cost = 0
    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]

        e = ca.vertcat(xk[0] - Rref[0, k], xk[1])
        cost += ca.mtimes([e.T, Q_mpc, e]) + ca.mtimes([uk.T, R_mpc, uk])

        x_next = ca.mtimes(Ad, xk) + ca.mtimes(Bd, uk)
        opti.subject_to(X[:, k+1] == x_next)

        opti.subject_to(X[0, k] <= x1_max)
        opti.subject_to(X[0, k] >= -x1_max)
        opti.subject_to(X[1, k] <= x2_max)
        opti.subject_to(X[1, k] >= -x2_max)
        opti.subject_to(U[0, k] <= u_max)
        opti.subject_to(U[0, k] >= -u_max)

    eN = ca.vertcat(X[0, N] - Rref[0, N-1], X[1, N])
    cost += ca.mtimes([eN.T, Qf_mpc, eN])

    opti.minimize(cost)
    opti.subject_to(X[:, 0] == X0)

    opti.set_initial(X, 0)
    opti.set_initial(U, 0)

    opti.solver("ipopt", {"print_time": 0}, {"print_level": 0})
    return opti, X0, Rref, U, X

# 工程注释（标准 MPC）：
# 构建基于 CasADi Opti 的标准预测控制问题：
# - 预测变量 X(状态)、U(输入)，参数为初值 X0 和参考轨迹 Rref；
# - 代价为加权的跟踪误差与控制努力（时间累加），终端项使用 Qf_mpc；
# - 加入硬约束：状态逐步限制在 [-xmax, xmax]，输入在 [-u_max, u_max]，
# - 求解器选择 IPOPT（非线性规划求解器），为可扩展到更复杂动力学保留灵活性。

# =========================================================
# 7) Build Tube-MPC (nominal Z,V with tightened constraints)
# =========================================================


def build_tube_mpc():
    opti = ca.Opti()

    Z = opti.variable(nx, N+1)
    V = opti.variable(nu, N)
    Z0 = opti.parameter(nx, 1)
    Rref = opti.parameter(1, N)

    cost = 0
    for k in range(N):
        zk = Z[:, k]
        vk = V[:, k]

        e = ca.vertcat(zk[0] - Rref[0, k], zk[1])
        cost += ca.mtimes([e.T, Q_mpc, e]) + ca.mtimes([vk.T, R_mpc, vk])

        # nominal dynamics (critical)
        z_next = ca.mtimes(Ad, zk) + ca.mtimes(Bd, vk)
        opti.subject_to(Z[:, k+1] == z_next)

        # tightened constraints
        opti.subject_to(Z[0, k] <= zmax[0])
        opti.subject_to(Z[0, k] >= -zmax[0])
        opti.subject_to(Z[1, k] <= zmax[1])
        opti.subject_to(Z[1, k] >= -zmax[1])
        opti.subject_to(V[0, k] <= vmax)
        opti.subject_to(V[0, k] >= -vmax)

    eN = ca.vertcat(Z[0, N] - Rref[0, N-1], Z[1, N])
    cost += ca.mtimes([eN.T, Qf_mpc, eN])

    opti.minimize(cost)
    opti.subject_to(Z[:, 0] == Z0)

    opti.set_initial(Z, 0)
    opti.set_initial(V, 0)

    opti.solver("ipopt", {"print_time": 0}, {"print_level": 0})
    return opti, Z0, Rref, V, Z

# 工程注释（Tube-MPC）：
# 管道 MPC 在名义系统上求解 Z（状态轨迹）和 V（名义输入），
# 所有约束都使用在步骤 5 计算的 "收紧" 值（zmax, vmax），以保证当实际存在扰动并由内环反馈 K_tube 修正时，
# 真实系统 x = z + e 仍满足原始约束。控制实施为 u = v + K_tube*(x - z)。


std_opti, std_X0, std_Rref, std_U, std_X = build_standard_mpc()
tub_opti, tub_Z0, tub_Rref, tub_V, tub_Z = build_tube_mpc()

# =========================================================
# 8) Closed-loop simulation
# =========================================================
Tsim = 200

x_std = np.zeros((2, 1))
x_tub = np.zeros((2, 1))
z_nom = x_tub.copy()

rng = np.random.default_rng(2)

t_solve_std = np.zeros(Tsim)   # Standard MPC solve time [s]
t_solve_tub = np.zeros(Tsim)   # Tube MPC solve time [s]

log = {
    "r": [],
    "x1_std": [], "x2_std": [], "u_std": [],
    "x1_tub": [], "x2_tub": [], "u_tub": [],
    "v_tub": []
}

# Warm-start buffers
U_prev = np.zeros((1, N))
V_prev = np.zeros((1, N))
X_prev = np.zeros((2, N+1))
Z_prev = np.zeros((2, N+1))

# 工程注释（仿真与热启动）：
# - 使用随机种子 rng 固定仿真可复现性；
# - Warm-start 缓存上一步的解（U_prev, V_prev, X_prev, Z_prev）用于加速下一步求解，
#   这在实时 MPC 中是常见的实践以减少求解时间和迭代次数。

for t in range(Tsim):
    r_h = np.array([ref_square(t + k) for k in range(N)]).reshape(1, -1)
    r_now = ref_square(t)

    # ---------- Standard MPC ----------
    std_opti.set_value(std_X0, x_std)
    std_opti.set_value(std_Rref, r_h)

    std_opti.set_initial(std_U, U_prev)
    std_opti.set_initial(std_X, X_prev)

    t0 = time.perf_counter()
    sol_std = std_opti.solve()
    t_solve_std[t] = time.perf_counter() - t0

    u0_std = to_scalar(sol_std.value(std_U[0, 0]))

    U_sol = np.asarray(sol_std.value(std_U)).reshape(1, -1)
    X_sol = np.asarray(sol_std.value(std_X))

    U_prev = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])
    X_prev = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])

# 工程注释（标准 MPC 步）：
# - 将当前状态和参考序列作为参数设置给 Opti 问题并使用上次解做热启动；
# - 记录求解时间用于评估在线可行性；
# - 提取第一步控制 u0_std 作为实际施加到真值系统的输入（在真实系统中会加入 actuation 与传感延迟考虑）。

    # ---------- Tube MPC ----------
    tub_opti.set_value(tub_Z0, z_nom)
    tub_opti.set_value(tub_Rref, r_h)

    tub_opti.set_initial(tub_V, V_prev)
    tub_opti.set_initial(tub_Z, Z_prev)

    t0 = time.perf_counter()
    sol_tub = tub_opti.solve()
    t_solve_tub[t] = time.perf_counter() - t0

    v0 = to_scalar(sol_tub.value(tub_V[0, 0]))

    V_sol = np.asarray(sol_tub.value(tub_V)).reshape(1, -1)
    Z_sol = np.asarray(sol_tub.value(tub_Z))

    V_prev = np.hstack([V_sol[:, 1:], V_sol[:, -1:]])
    Z_prev = np.hstack([Z_sol[:, 1:], Z_sol[:, -1:]])

    # tube feedback: u = v + K_tube (x - z)
    e = (x_tub - z_nom).reshape(2,)
    u0_tub = float(v0 + (K_tube @ e.reshape(2, 1))[0, 0])
    u0_tub = np.clip(u0_tub, -u_max, u_max)

    # nominal update
    z_nom = Ad @ z_nom + Bd * v0

    # ---------- True plant with INPUT disturbance ----------
    d_std = rng.uniform(-1, 1) * dmax_eff
    d_tub = rng.uniform(-1, 1) * dmax_eff

    x_std = Ad @ x_std + Bd * (u0_std + d_std)
    x_tub = Ad @ x_tub + Bd * (u0_tub + d_tub)

    # log
    log["r"].append(r_now)

    log["x1_std"].append(x_std[0, 0])
    log["x2_std"].append(x_std[1, 0])
    log["u_std"].append(u0_std)

    log["x1_tub"].append(x_tub[0, 0])
    log["x2_tub"].append(x_tub[1, 0])
    log["u_tub"].append(u0_tub)
    log["v_tub"].append(v0)

# 工程注释（闭环仿真流程说明）：
# - 每步构造未来 N 步的参考轨迹 r_h 并分别求解标准 MPC 和管道 MPC；
# - 标准 MPC 直接输出控制 u，而管道 MPC 输出名义输入 v，再加上内环反馈生成实际 u：
#   u = v + K_tube*(x - z)，该反馈用于抑制扰动导致的偏差；
# - 在仿真中向两个系统注入随机输入扰动 d_std / d_tub（幅值受 dmax_eff 控制），用于验证鲁棒性；
# - 记录状态、输入与求解时间用于后续性能比较。

# =========================================================
# 9) Plot
# =========================================================
tt = np.arange(Tsim) * Ts

plt.figure()
plt.plot(tt, log["r"], label="ref (square)")
plt.plot(tt, log["x1_std"], label="x1 standard MPC")
plt.plot(tt, log["x1_tub"], label="x1 Tube-MPC (input-dist)")
plt.xlabel("time [s]")
plt.ylabel("position x1")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(tt, log["u_std"], label="u standard MPC")
plt.plot(tt, log["u_tub"], label="u Tube-MPC")
plt.plot(tt, log["v_tub"], "--", label="v (nominal) Tube-MPC")
plt.xlabel("time [s]")
plt.ylabel("input")
plt.grid(True)
plt.legend()

# 画每步求解耗时曲线
plt.figure()
plt.plot(tt, 1e3*t_solve_std, label="Standard MPC solve [ms]")
plt.plot(tt, 1e3*t_solve_tub, label="Tube MPC solve [ms]")
plt.xlabel("time [s]")
plt.ylabel("solve time [ms]")
plt.grid(True)
plt.legend()
plt.show()


def summarize_times(name, arr):
    arr_ms = 1e3 * np.asarray(arr)
    print(f"\n{name} solve time statistics:")
    print(f"  mean   = {arr_ms.mean():.3f} ms")  # 所有数据的平均值
    print(f"  median = {np.median(arr_ms):.3f} ms")  # 50% 的数据 ≤ 这个值
    print(f"  p90    = {np.percentile(arr_ms, 90):.3f} ms")  # 90% 的数据 ≤ 这个值
    print(f"  p99    = {np.percentile(arr_ms, 99):.3f} ms")  # 99% 的数据 ≤ 这个值
    print(f"  max    = {arr_ms.max():.3f} ms")  # 所有数据的最大值


# 第 0 步通常更慢（Ipopt 初始化、JIT 编译、内存分配）。你可以同时统计“去掉前 5 步的平均值”
# summarize_times("Standard MPC (skip first 5)", t_solve_std[5:])
# summarize_times("Tube MPC (skip first 5)", t_solve_tub[5:])
summarize_times("Standard MPC", t_solve_std)
summarize_times("Tube MPC", t_solve_tub)
print(f"\nTube/Standard mean ratio = {(t_solve_tub.mean() / t_solve_std.mean()):.3f}x")
