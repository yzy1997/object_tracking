"""
UAV 2D Truth Simulation + 3D Measurements z=[x, y, range rho] + CV-KF tracking (x,y only) + Plots
Figures saved to: simulator/imgs/

========================
Motion Plan (Total = 100 s)
========================
Segment 1: Constant velocity (CV)
- T1 = 25 s, v = 5 m/s, a = 0
- distance ≈ 125 m

Segment 2: Constant acceleration straight (CA)
- T2 = 20 s, a2 = +0.5 m/s^2
- v: 5 -> 15 m/s
- distance ≈ 200 m

Segment 3: Left-turn circular arc (constant-speed turn)
- T3 = 15 s, R3 = 80 m, v = 15 m/s
- omega3 = +v/R3 = +0.1875 rad/s
- turn angle ≈ +161.2 deg, arc length ≈ 225 m

Segment 4: Constant deceleration straight (CD)
- T4 = 20 s, a4 = -0.4 m/s^2
- v: 15 -> 7 m/s
- distance ≈ 220 m

Segment 5: Right-turn circular arc (constant-speed turn)
- T5 = 20 s, R5 = 120 m, v = 7 m/s
- omega5 = -v/R5 = -0.05833 rad/s
- turn angle ≈ -66.8 deg, arc length ≈ 140 m

========================
Measurement Model (simulated)
========================
Sensor position (approx): s = (100, 0, 1000) [m]
Target altitude (fixed): z0 = 100 [m]

We generate 3D measurements:
- z = [x_meas, y_meas, rho_meas]
  where rho = sqrt((x-sx)^2 + (y-sy)^2 + (z0-sz)^2)

Noise (typical at ~1 km):
- x,y noise: sigma_xy = 3.0 m
- range noise: sigma_r = 2.0 m

IMPORTANT:
- The tracker is PURE linear CV-KF and uses ONLY [x_meas, y_meas].
- rho is generated but intentionally ignored (for later EKF/UKF comparisons).

========================
Tracker Model (CV-KF, linear)
========================
State: [x, y, vx, vy]^T
F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
H = [[1,0,0,0],[0,1,0,0]]
Process noise: white acceleration, spectral density q (m^2/s^3)

Outputs:
- Figure 1: "CV-KF Tracking Simulation" (truth vs meas vs KF track)
- Figure 2: "Tracking RMSE (m) = XX.XX" showing XY-only RMSE over time
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Utilities / Configuration
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rmse_xy(truth_xy: np.ndarray, est_xy: np.ndarray) -> np.ndarray:
    """Per-time-step RMSE in XY (i.e., Euclidean position error)."""
    e = est_xy - truth_xy
    return np.sqrt(np.sum(e * e, axis=1))


# -------------------------
# Motion / Truth Simulation
# -------------------------
def simulate_truth(dt=0.5, z0=100.0, seed=0):
    """
    Returns:
      t: (N,)
      truth: (N,4) -> [x, y, vx, vy]
    """
    # Motion plan
    T1, v1 = 25.0, 5.0
    T2, a2 = 20.0, +0.5
    T3, R3, v3 = 15.0, 80.0, 15.0
    omega3 = +v3 / R3
    T4, a4 = 20.0, -0.4
    T5, R5, v5 = 20.0, 120.0, 7.0
    omega5 = -v5 / R5

    # Initial conditions (set a reasonable origin/heading)
    x, y = 0.0, 0.0
    heading = 0.0  # rad, along +x
    vx, vy = v1 * np.cos(heading), v1 * np.sin(heading)

    ts, xs, ys, vxs, vys = [], [], [], [], []
    t = 0.0

    def push():
        ts.append(t)
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)

    # Use semi-implicit update per step for stability & simplicity
    def step_straight(a_long):
        # accelerate along current heading
        nonlocal x, y, vx, vy
        ax = a_long * np.cos(heading)
        ay = a_long * np.sin(heading)
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        vx, vy, x, y = vx_new, vy_new, x_new, y_new

    def step_turn(omega, v_const):
        # constant-speed coordinated turn with given yaw rate omega
        nonlocal x, y, vx, vy, heading
        heading_new = heading + omega * dt
        vx_new = v_const * np.cos(heading_new)
        vy_new = v_const * np.sin(heading_new)
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        heading = heading_new
        vx, vy, x, y = vx_new, vy_new, x_new, y_new

    # Segment 1: CV straight, v=5
    n1 = int(np.round(T1 / dt))
    for _ in range(n1):
        push()
        step_straight(a_long=0.0)
        t += dt

    # Segment 2: CA straight, a=+0.5 (speed 5->15 in 20s)
    n2 = int(np.round(T2 / dt))
    for _ in range(n2):
        push()
        step_straight(a_long=a2)
        t += dt

    # Segment 3: left turn, v=15, omega=+v/R
    n3 = int(np.round(T3 / dt))
    for _ in range(n3):
        push()
        step_turn(omega=omega3, v_const=v3)
        t += dt

    # Segment 4: decel straight, a=-0.4 (speed 15->7 in 20s)
    # Straight along current heading
    n4 = int(np.round(T4 / dt))
    for _ in range(n4):
        push()
        step_straight(a_long=a4)
        t += dt

    # Segment 5: right turn, v=7, omega=-v/R
    n5 = int(np.round(T5 / dt))
    for _ in range(n5):
        push()
        step_turn(omega=omega5, v_const=v5)
        t += dt

    t_arr = np.array(ts)
    truth = np.column_stack([xs, ys, vxs, vys])
    return t_arr, truth


# -------------------------
# Measurement Simulation
# -------------------------
def simulate_measurements(t, truth, sensor_pos=(100.0, 0.0, 1000.0), z0=100.0,
                          sigma_xy=3.0, sigma_r=2.0, seed=1):
    """
    Generate z = [x_meas, y_meas, rho_meas], with independent Gaussian noise.
    Returns:
      meas: (N,3)
    """
    rng = np.random.default_rng(seed)
    sx, sy, sz = sensor_pos
    x = truth[:, 0]
    y = truth[:, 1]

    rho_true = np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z0 - sz) ** 2)

    x_meas = x + rng.normal(0.0, sigma_xy, size=len(t))
    y_meas = y + rng.normal(0.0, sigma_xy, size=len(t))
    rho_meas = rho_true + rng.normal(0.0, sigma_r, size=len(t))

    return np.column_stack([x_meas, y_meas, rho_meas])


# -------------------------
# CV-KF (linear) Tracker
# -------------------------
class CVKalmanFilter2D:
    """
    State: [x, y, vx, vy]
    Measurement used: [x, y] only
    """
    def __init__(self, dt, q=0.6, r_xy=3.0, P0=None):
        self.dt = dt
        self.q = float(q)
        self.r_xy = float(r_xy)

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Continuous white-noise accel -> discrete Q (per axis), block diagonal
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.q
        Q1 = q * np.array([[dt4 / 4, dt3 / 2],
                           [dt3 / 2, dt2]], dtype=float)
        self.Q = np.block([
            [Q1, np.zeros((2, 2))],
            [np.zeros((2, 2)), Q1]
        ])

        self.R = (self.r_xy ** 2) * np.eye(2)

        self.x = np.zeros(4)
        self.P = np.eye(4) if P0 is None else P0.copy()

        self.I = np.eye(4)

    def init_from_measurement(self, z_xy, v0=(0.0, 0.0), P_pos=50.0, P_vel=25.0):
        self.x = np.array([z_xy[0], z_xy[1], v0[0], v0[1]], dtype=float)
        self.P = np.diag([P_pos**2, P_pos**2, P_vel**2, P_vel**2]).astype(float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_xy):
        z = np.asarray(z_xy, dtype=float).reshape(2)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P


def run_kf(t, meas_xyz, dt):
    kf = CVKalmanFilter2D(dt=dt, q=0.6, r_xy=3.0)
    kf.init_from_measurement(meas_xyz[0, :2], v0=(0.0, 0.0), P_pos=60.0, P_vel=30.0)

    est = np.zeros((len(t), 4))
    for k in range(len(t)):
        kf.predict()
        kf.update(meas_xyz[k, :2])
        est[k] = kf.x
    return est


# -------------------------
# Plotting
# -------------------------
def make_plots(t, truth, meas, est, out_dir="simulator/imgs", dpi=220):
    ensure_dir(out_dir)

    truth_xy = truth[:, :2]
    meas_xy = meas[:, :2]
    est_xy = est[:, :2]

    # Figure 1: Trajectories
    fig1, ax1 = plt.subplots(figsize=(8.6, 6.6))
    ax1.plot(truth_xy[:, 0], truth_xy[:, 1], "-", linewidth=2.2, label="Truth trajectory")
    ax1.plot(meas_xy[:, 0], meas_xy[:, 1], "x", markersize=3.5, alpha=0.7, label="Measurements (x,y)")
    ax1.plot(est_xy[:, 0], est_xy[:, 1], "--", linewidth=2.0, label="CV-KF track (x,y only)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("CV-KF Tracking Simulation")
    ax1.grid(True, linestyle=":", linewidth=0.8)
    ax1.legend(loc="best")

    # reasonable view: auto + equal aspect
    ax1.set_aspect("equal", adjustable="datalim")

    save_path_traj = os.path.join(out_dir, "cv_kf_tracking.png")
    fig1.tight_layout()
    fig1.savefig(save_path_traj, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"Saved trajectory figure to: {save_path_traj}")

    # Figure 2: RMSE (XY only) with title including overall RMSE
    e = rmse_xy(truth_xy, est_xy)
    overall = float(np.sqrt(np.mean(e**2)))

    fig2, ax2 = plt.subplots(figsize=(8.6, 4.6))
    ax2.plot(t, e, linewidth=2.0, label="RMSE (x,y only)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position Error (m)")
    ax2.set_title(f"Tracking RMSE (m) = {overall:.2f}")
    ax2.grid(True, linestyle=":", linewidth=0.8)
    ax2.legend(loc="best")

    save_path_rmse = os.path.join(out_dir, "rmse_cv_kf.png")
    fig2.tight_layout()
    fig2.savefig(save_path_rmse, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"Saved RMSE figure to: {save_path_rmse}")


def main():
    # Sampling rate can be adjusted (requirement: can downsample appropriately)
    dt = 0.5  # seconds (0.2 is fine too; 0.5 makes ~200 points for 100s)

    # Sensor and measurement noise
    sensor_pos = (100.0, 0.0, 1000.0)
    z0 = 100.0
    sigma_xy = 3.0
    sigma_r = 2.0

    # Simulate truth & measurements
    t, truth = simulate_truth(dt=dt, z0=z0, seed=0)
    meas = simulate_measurements(t, truth, sensor_pos=sensor_pos, z0=z0,
                                 sigma_xy=sigma_xy, sigma_r=sigma_r, seed=1)

    # Track with PURE linear CV-KF using x,y only
    est = run_kf(t, meas, dt)

    # Plots + save
    make_plots(t, truth, meas, est, out_dir="simulator/imgs", dpi=220)


if __name__ == "__main__":
    main()