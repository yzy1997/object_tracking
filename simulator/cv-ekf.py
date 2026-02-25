"""
无人机轨迹仿真（2D真值） + 3D观测 z=[x, y, 距离rho] + CV-EKF追踪 + 轨迹图&误差图（保存到 simulator/imgs/）

========================
运动方案（总时长约 100 s）
========================
采样周期：
- dt = 0.5 s  （你可改为 0.2/1.0）

状态与坐标：
- 目标在XY平面运动，目标高度 z0 固定（仅用于距离rho计算）：z0 = 100 m
- 传感器（观测点/雷达）位置 sensor_pos = (100, 0, 1000) m  （按你要求移到该位置附近）
- 真值初始位置 (x0,y0) = (0,0)，初始航向沿 +x
- 初始速度 v0 = 5 m/s

5段运动（匀速→匀加速→左转→匀减速→右转），时长分配为 100 s：
1) 段1 匀速直线（CV）
   - T1 = 25 s, v = 5 m/s, a = 0, 距离≈125 m

2) 段2 匀加速直线（CA）
   - T2 = 20 s, a2 = +0.5 m/s^2
   - v: 5 -> 15 m/s, 距离≈200 m

3) 段3 左转圆弧（恒速转弯）
   - T3 = 15 s, R3 = 80 m, v = 15 m/s
   - omega3 = +v/R = +0.1875 rad/s
   - 转角≈161.2 deg, 弧长≈225 m

4) 段4 匀减速直线（CD）
   - T4 = 20 s, a4 = -0.4 m/s^2
   - v: 15 -> 7 m/s, 距离≈220 m

5) 段5 右转圆弧（恒速转弯）
   - T5 = 20 s, R5 = 120 m, v = 7 m/s
   - omega5 = -v/R = -0.05833 rad/s
   - 转角≈-66.8 deg, 弧长≈140 m

========================
观测模型（3D数据）
========================
观测 z = [x_meas, y_meas, rho_meas]^T
- x_meas, y_meas：带噪声的平面位置
- rho_meas：传感器到目标的距离（含高度差）

噪声强度（距离相关标定）：
- sigma_xy(r)  = sigma0_xy  + k_xy  * r
- sigma_rho(r) = sigma0_rho + k_rho * r
Rk = diag([sigma_xy^2, sigma_xy^2, sigma_rho^2])

默认：
- sigma0_xy=1.5 m, k_xy=0.0012
- sigma0_rho=2.5 m, k_rho=0.0020

========================
滤波器：CV-EKF
========================
状态：x=[px, py, vx, vy]^T
预测：CV线性模型 + 白加速度过程噪声Q(q)
更新：z=[px, py, rho(px,py)] -> EKF更新

输出图片：
- simulator/imgs/cv_kf_tracking.png 标题：“CV-KF追踪仿真图”
- simulator/imgs/rmse_cn.png        标题：“跟踪误差RMSE（米）”
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def simulate_truth_2d_piecewise(
    dt=0.5,
    z0=100.0,
    x0=0.0, y0=0.0, v0=5.0, psi0=0.0,
    T1=25.0,
    T2=20.0, a2=+0.5,
    T3=15.0, R3=80.0,
    T4=20.0, a4=-0.4,
    T5=20.0, R5=120.0,
):
    T_total = T1 + T2 + T3 + T4 + T5
    N = int(np.floor(T_total / dt)) + 1
    t = np.arange(N) * dt

    x = np.zeros(N)
    y = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    psi = np.zeros(N)
    vmag = np.zeros(N)

    x[0], y[0] = x0, y0
    psi[0] = psi0
    vmag[0] = v0
    vx[0] = v0 * np.cos(psi0)
    vy[0] = v0 * np.sin(psi0)

    edges = np.cumsum([0.0, T1, T2, T3, T4, T5])

    # for reporting
    v1 = v0
    v2_end = v1 + a2 * T2
    v3 = v2_end
    omega3 = +v3 / R3
    dtheta3 = omega3 * T3
    v4_start = v3
    v4_end = v4_start + a4 * T4
    v5 = v4_end
    omega5 = -v5 / R5
    dtheta5 = omega5 * T5

    for k in range(1, N):
        ti = t[k - 1]

        v_prev = vmag[k - 1]
        psi_prev = psi[k - 1]

        if ti < edges[1]:
            v_new = v_prev
            psi_new = psi_prev
        elif ti < edges[2]:
            v_new = max(v_prev + a2 * dt, 0.0)
            psi_new = psi_prev
        elif ti < edges[3]:
            v_new = v3
            psi_new = psi_prev + omega3 * dt
        elif ti < edges[4]:
            v_new = max(v_prev + a4 * dt, 0.0)
            psi_new = psi_prev
        else:
            v_new = max(v5, 1e-6)
            psi_new = psi_prev + omega5 * dt

        vx[k] = v_new * np.cos(psi_new)
        vy[k] = v_new * np.sin(psi_new)
        x[k] = x[k - 1] + vx[k] * dt
        y[k] = y[k - 1] + vy[k] * dt
        psi[k] = psi_new
        vmag[k] = v_new

    params = {
        "dt": dt,
        "T_total": T_total,
        "z0": z0,
        "segments": {
            "seg1": {"type": "CV", "T": T1, "v": v1, "a": 0.0},
            "seg2": {"type": "CA", "T": T2, "a": a2, "v_start": v1, "v_end": v2_end},
            "seg3": {"type": "Left Turn", "T": T3, "R": R3, "v": v3, "omega": omega3,
                     "dtheta_rad": dtheta3, "dtheta_deg": np.degrees(dtheta3)},
            "seg4": {"type": "CD", "T": T4, "a": a4, "v_start": v4_start, "v_end": v4_end},
            "seg5": {"type": "Right Turn", "T": T5, "R": R5, "v": v5, "omega": omega5,
                     "dtheta_rad": dtheta5, "dtheta_deg": np.degrees(dtheta5)},
        }
    }

    return {"t": t, "x": x, "y": y, "vx": vx, "vy": vy, "psi": psi, "vmag": vmag, "z0": z0, "params": params}


def make_measurements_xy_rho(
    truth,
    sensor_pos=(100.0, 0.0, 1000.0),
    sigma0_xy=1.5, k_xy=0.0012,
    sigma0_rho=2.5, k_rho=0.0020,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(0)

    xs, ys, zs = sensor_pos
    x = truth["x"]
    y = truth["y"]
    z0 = truth["z0"]

    dx = x - xs
    dy = y - ys
    dz = z0 - zs
    rho_true = np.sqrt(dx * dx + dy * dy + dz * dz)

    sigma_xy = sigma0_xy + k_xy * rho_true
    sigma_rho = sigma0_rho + k_rho * rho_true

    z = np.zeros((len(x), 3))
    z[:, 0] = x + rng.normal(0.0, sigma_xy)
    z[:, 1] = y + rng.normal(0.0, sigma_xy)
    z[:, 2] = rho_true + rng.normal(0.0, sigma_rho)

    R_list = np.zeros((len(x), 3, 3))
    for i in range(len(x)):
        R_list[i] = np.diag([sigma_xy[i] ** 2, sigma_xy[i] ** 2, sigma_rho[i] ** 2])

    return {
        "z": z,
        "R_list": R_list,
        "sensor_pos": np.array(sensor_pos, dtype=float),
        "rho_true": rho_true,
        "sigma_xy": sigma_xy,
        "sigma_rho": sigma_rho,
    }


class CVEKF2D:
    def __init__(self, dt, q=0.8):
        self.dt = float(dt)
        self.q = float(q)
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0
        self.I = np.eye(4)

    def F(self):
        dt = self.dt
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    def Q(self):
        dt = self.dt
        q = self.q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                             [0, dt4 / 4, 0, dt3 / 2],
                             [dt3 / 2, 0, dt2, 0],
                             [0, dt3 / 2, 0, dt2]], dtype=float)

    @staticmethod
    def h(x, sensor_pos, z0):
        px, py, vx, vy = x
        xs, ys, zs = sensor_pos
        dx = px - xs
        dy = py - ys
        dz = z0 - zs
        rho = np.sqrt(dx * dx + dy * dy + dz * dz)
        return np.array([px, py, rho], dtype=float)

    @staticmethod
    def H_jac(x, sensor_pos, z0):
        px, py, vx, vy = x
        xs, ys, zs = sensor_pos
        dx = px - xs
        dy = py - ys
        dz = z0 - zs
        rho = np.sqrt(dx * dx + dy * dy + dz * dz)
        rho = max(rho, 1e-9)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [dx / rho, dy / rho, 0, 0]
        ], dtype=float)

    def predict(self):
        F = self.F()
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q()

    def update(self, z, R, sensor_pos, z0):
        z = np.asarray(z, dtype=float).reshape(3)
        H = self.H_jac(self.x, sensor_pos, z0)
        zhat = self.h(self.x, sensor_pos, z0)
        y = z - zhat

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

    def step(self, z, R, sensor_pos, z0):
        self.predict()
        self.update(z, R, sensor_pos, z0)
        return self.x.copy(), self.P.copy()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def rmse_series(truth_xy, est_xy):
    e = est_xy - truth_xy
    err = np.sqrt(np.sum(e * e, axis=1))
    rmse = np.sqrt(np.mean(err ** 2))
    return err, rmse


def set_reasonable_limits(ax, x_all, y_all, pad_ratio=0.12):
    xmin, xmax = np.min(x_all), np.max(x_all)
    ymin, ymax = np.min(y_all), np.max(y_all)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    ax.set_xlim(xmin - dx * pad_ratio, xmax + dx * pad_ratio)
    ax.set_ylim(ymin - dy * pad_ratio, ymax + dy * pad_ratio)
    ax.set_aspect("equal", adjustable="box")


def print_motion_params(params):
    segs = params["segments"]
    print("======== 运动参数汇总（用于核查） ========")
    print(f"dt = {params['dt']} s, 总时长 T_total = {params['T_total']} s, z0 = {params['z0']} m")
    s1 = segs["seg1"]
    print(f"[段1] 匀速直线: T={s1['T']}s, v={s1['v']:.3f}m/s, a={s1['a']:.3f}m/s^2, 距离≈{s1['v']*s1['T']:.1f}m")
    s2 = segs["seg2"]
    L2 = s2["v_start"]*s2["T"] + 0.5*s2["a"]*s2["T"]**2
    print(f"[段2] 匀加速直线: T={s2['T']}s, a={s2['a']:.3f}m/s^2, v: {s2['v_start']:.3f}->{s2['v_end']:.3f}m/s, 距离≈{L2:.1f}m")
    s3 = segs["seg3"]
    print(f"[段3] 左转圆弧: T={s3['T']}s, R={s3['R']}m, v={s3['v']:.3f}m/s, "
          f"omega={s3['omega']:.6f}rad/s, 转角≈{s3['dtheta_deg']:.1f}deg, 弧长≈{s3['v']*s3['T']:.1f}m")
    s4 = segs["seg4"]
    L4 = s4["v_start"]*s4["T"] + 0.5*s4["a"]*s4["T"]**2
    print(f"[段4] 匀减速直线: T={s4['T']}s, a={s4['a']:.3f}m/s^2, v: {s4['v_start']:.3f}->{s4['v_end']:.3f}m/s, 距离≈{L4:.1f}m")
    s5 = segs["seg5"]
    print(f"[段5] 右转圆弧: T={s5['T']}s, R={s5['R']}m, v={s5['v']:.3f}m/s, "
          f"omega={s5['omega']:.6f}rad/s, 转角≈{s5['dtheta_deg']:.1f}deg, 弧长≈{s5['v']*s5['T']:.1f}m")
    print("=======================================")


def main():
    dt = 0.5
    z0 = 100.0

    # motion (100s)
    T1 = 25.0
    T2, a2 = 20.0, +0.5
    T3, R3 = 15.0, 80.0
    T4, a4 = 20.0, -0.4
    T5, R5 = 20.0, 120.0

    x0, y0, v0, psi0 = 0.0, 0.0, 5.0, 0.0

    # sensor moved near (100,0,1000)
    sensor_pos = (100.0, 0.0, 1000.0)

    # noise
    sigma0_xy, k_xy = 1.5, 0.0012
    sigma0_rho, k_rho = 2.5, 0.0020

    # filter
    q = 0.8

    # outputs
    out_dir = os.path.join("simulator", "imgs")
    ensure_dir(out_dir)
    save_path_track = os.path.join(out_dir, "cv_ekf_tracking.png")
    save_path_rmse = os.path.join(out_dir, "rmse_cv_ekf.png")

    rng = np.random.default_rng(42)

    truth = simulate_truth_2d_piecewise(
        dt=dt, z0=z0,
        x0=x0, y0=y0, v0=v0, psi0=psi0,
        T1=T1, T2=T2, a2=a2, T3=T3, R3=R3,
        T4=T4, a4=a4, T5=T5, R5=R5,
    )
    print_motion_params(truth["params"])

    meas = make_measurements_xy_rho(
        truth,
        sensor_pos=sensor_pos,
        sigma0_xy=sigma0_xy, k_xy=k_xy,
        sigma0_rho=sigma0_rho, k_rho=k_rho,
        rng=rng
    )
    z = meas["z"]
    R_list = meas["R_list"]

    ekf = CVEKF2D(dt=dt, q=q)
    ekf.x = np.array([z[0, 0], z[0, 1], 0.0, 0.0], dtype=float)
    ekf.P = np.diag([60.0**2, 60.0**2, 30.0**2, 30.0**2])

    N = len(truth["t"])
    xhat = np.zeros((N, 4))
    for i in range(N):
        xi, _ = ekf.step(z[i], R_list[i], meas["sensor_pos"], truth["z0"])
        xhat[i] = xi

    truth_xy = np.column_stack([truth["x"], truth["y"]])
    est_xy = xhat[:, :2]
    err, rmse_val = rmse_series(truth_xy, est_xy)
    print(f"位置误差RMSE（欧氏距离, XY）= {rmse_val:.3f} m")

    # plot 1
    fig1, ax1 = plt.subplots(figsize=(9.5, 7.5))
    ax1.plot(truth["x"], truth["y"], "k-", linewidth=2.0, label="Truth trajectory (Truth)")
    ax1.plot(z[:, 0], z[:, 1], "rx", markersize=4, alpha=0.65, label="Measurements (x, y)")
    ax1.plot(est_xy[:, 0], est_xy[:, 1], "b--", linewidth=2.0, label="Track trajectory (CV-KF)")

    xs, ys, _ = sensor_pos
    ax1.scatter([xs], [ys], c="g", s=70, marker="^", label="Sensor")

    x_all = np.concatenate([truth["x"], z[:, 0], est_xy[:, 0], np.array([xs])])
    y_all = np.concatenate([truth["y"], z[:, 1], est_xy[:, 1], np.array([ys])])
    set_reasonable_limits(ax1, x_all, y_all, pad_ratio=0.12)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("CV-KF Tracking Simulation")
    ax1.grid(True, linestyle=":", linewidth=0.8)
    ax1.legend(loc="best")

    fig1.tight_layout()
    fig1.savefig(save_path_track, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"已保存轨迹图到: {save_path_track}")

    # plot 2
    fig2, ax2 = plt.subplots(figsize=(10.5, 4.8))
    ax2.plot(truth["t"], err, "m-", linewidth=2.0, label="|Position Error| (m)")
    ax2.axhline(rmse_val, color="k", linestyle="--", linewidth=1.5, label=f"RMSE = {rmse_val:.2f} m")
    ax2.set_xlabel("time(s)")
    ax2.set_ylabel("error(m)")
    ax2.set_title("Tracking RMSE (m)")
    ax2.grid(True, linestyle=":", linewidth=0.8)
    ax2.legend(loc="best")

    fig2.tight_layout()
    fig2.savefig(save_path_rmse, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"已保存RMSE图到: {save_path_rmse}")


if __name__ == "__main__":
    main()