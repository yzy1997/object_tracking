import os
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Utilities / Configuration
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rmse_xy(truth_xy: np.ndarray, est_xy: np.ndarray) -> np.ndarray:
    """Per-time-step Euclidean position error in XY."""
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
    # Motion plan (same as before)
    T1, v1 = 25.0, 5.0
    T2, a2 = 20.0, +0.5
    T3, R3, v3 = 15.0, 80.0, 15.0
    omega3 = +v3 / R3
    T4, a4 = 20.0, -0.4
    T5, R5, v5 = 20.0, 120.0, 7.0
    omega5 = -v5 / R5

    # Initial conditions
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

    def step_straight(a_long):
        nonlocal x, y, vx, vy
        ax = a_long * np.cos(heading)
        ay = a_long * np.sin(heading)
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        vx, vy, x, y = vx_new, vy_new, x_new, y_new

    def step_turn(omega, v_const):
        nonlocal x, y, vx, vy, heading
        heading_new = heading + omega * dt
        vx_new = v_const * np.cos(heading_new)
        vy_new = v_const * np.sin(heading_new)
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        heading = heading_new
        vx, vy, x, y = vx_new, vy_new, x_new, y_new

    # Simulate truth motion
    n1 = int(np.round(T1 / dt))
    for _ in range(n1):
        push()
        step_straight(a_long=0.0)
        t += dt

    n2 = int(np.round(T2 / dt))
    for _ in range(n2):
        push()
        step_straight(a_long=a2)
        t += dt

    n3 = int(np.round(T3 / dt))
    for _ in range(n3):
        push()
        step_turn(omega=omega3, v_const=v3)
        t += dt

    n4 = int(np.round(T4 / dt))
    for _ in range(n4):
        push()
        step_straight(a_long=a4)
        t += dt

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
# IMM-KF Tracker (Multiple Models)
# -------------------------
class IMMKalmanFilter:
    """
    State: [x, y, vx, vy]
    Model 1: Constant velocity (CV-KF)
    Model 2: Constant acceleration (CA-KF)
    """
    def __init__(self, dt, q_cv=0.6, q_ca=0.6, r_xy=3.0):
        self.dt = dt
        self.q_cv = q_cv
        self.q_ca = q_ca
        self.r_xy = r_xy

        # State transition matrices for each model
        self.F_cv = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.F_ca = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=float)

        # Measurement matrix (same for both models)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

        # Process noise covariance for each model (4x4 matrices now)
        self.Q_cv = self.q_cv * np.array([[self.dt**4 / 4, self.dt**3 / 2, 0, 0],
                                          [self.dt**3 / 2, self.dt**2, 0, 0],
                                          [0, 0, self.dt**4 / 4, self.dt**3 / 2],
                                          [0, 0, self.dt**3 / 2, self.dt**2]], dtype=float)
        self.Q_ca = self.q_ca * np.array([[self.dt**4 / 4, self.dt**3 / 2, 0, 0],
                                          [self.dt**3 / 2, self.dt**2, 0, 0],
                                          [0, 0, self.dt**4 / 4, self.dt**3 / 2],
                                          [0, 0, self.dt**3 / 2, self.dt**2]], dtype=float)

        # Measurement noise covariance (same for both models)
        self.R = (self.r_xy**2) * np.eye(2)

        # Initial state estimates and covariances
        self.P_cv = np.eye(4) * 500  # Initial covariance for CV-KF
        self.P_ca = np.eye(4) * 500  # Initial covariance for CA-KF

        self.x_cv = np.zeros(4)
        self.x_ca = np.zeros(4)

    def predict(self, model):
        """
        Predict the next state based on the model selected.
        """
        if model == 1:  # CV model
            self.x_cv = self.F_cv @ self.x_cv
            self.P_cv = self.F_cv @ self.P_cv @ self.F_cv.T + self.Q_cv
        else:  # CA model
            self.x_ca = self.F_ca @ self.x_ca
            self.P_ca = self.F_ca @ self.P_ca @ self.F_ca.T + self.Q_ca

    def update(self, z_xy, model):
        """
        Update the state using the measurement z_xy based on the model selected.
        """
        z = np.asarray(z_xy, dtype=float).reshape(2)
        if model == 1:  # CV model
            y = z - (self.H @ self.x_cv)
            S = self.H @ self.P_cv @ self.H.T + self.R
            K = self.P_cv @ self.H.T @ np.linalg.inv(S)
            self.x_cv = self.x_cv + K @ y
            self.P_cv = (np.eye(4) - K @ self.H) @ self.P_cv
        else:  # CA model
            y = z - (self.H @ self.x_ca)
            S = self.H @ self.P_ca @ self.H.T + self.R
            K = self.P_ca @ self.H.T @ np.linalg.inv(S)
            self.x_ca = self.x_ca + K @ y
            self.P_ca = (np.eye(4) - K @ self.H) @ self.P_ca

    def get_state(self, model):
        """
        Return the state estimate for the selected model.
        """
        if model == 1:
            return self.x_cv
        else:
            return self.x_ca

    def get_error_covariance(self, model):
        """
        Return the error covariance matrix for the selected model.
        """
        if model == 1:
            return self.P_cv
        else:
            return self.P_ca


def run_imm_kf(t, meas_xyz, dt):
    imm_kf = IMMKalmanFilter(dt=dt)

    est = np.zeros((len(t), 4))
    for k in range(len(t)):
        # Predict and update for both models
        imm_kf.predict(model=1)
        imm_kf.update(meas_xyz[k, :2], model=1)

        imm_kf.predict(model=2)
        imm_kf.update(meas_xyz[k, :2], model=2)

        # Combine estimates (weighted by likelihood)
        est[k] = (imm_kf.get_state(model=1) + imm_kf.get_state(model=2)) / 2

    return est


# -------------------------
# Plotting
# -------------------------
def make_plots(t, truth, meas, est, sensor_pos=(100.0, 0.0, 1000.0), out_dir="simulator/imgs", dpi=220):
    ensure_dir(out_dir)

    truth_xy = truth[:, :2]
    meas_xy = meas[:, :2]
    est_xy = est[:, :2]

    # Figure 1: Trajectories
    fig1, ax1 = plt.subplots(figsize=(8.6, 6.6))
    ax1.plot(truth_xy[:, 0], truth_xy[:, 1], "-", linewidth=2.2, label="Truth trajectory")
    ax1.plot(meas_xy[:, 0], meas_xy[:, 1], "x", markersize=3.5, alpha=0.7, label="Measurements (x,y)")
    ax1.plot(est_xy[:, 0], est_xy[:, 1], "--", linewidth=2.0, label="IMM-KF track")

    # Sensor marker (green triangle) in XY plane
    sx, sy, _sz = sensor_pos
    ax1.plot([sx], [sy], marker="^", color="green", markersize=10, label="Sensor (x,y)")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("IMM-KF Tracking Simulation")
    ax1.grid(True, linestyle=":", linewidth=0.8)
    ax1.legend(loc="best")
    ax1.set_aspect("equal", adjustable="datalim")

    save_path_traj = os.path.join(out_dir, "imm_kf_tracking.png")
    fig1.tight_layout()
    fig1.savefig(save_path_traj, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"Saved trajectory figure to: {save_path_traj}")

    # Figure 2: Position Error + Running RMSE (XY only)
    e = rmse_xy(truth_xy, est_xy)
    rmse_running = np.sqrt(np.cumsum(e**2) / (np.arange(len(e)) + 1))
    overall = float(np.sqrt(np.mean(e**2)))

    e_mean = float(np.mean(e))
    e_max = float(np.max(e))

    fig2, ax2 = plt.subplots(figsize=(8.6, 4.6))
    ax2.plot(
        t, e, color="tab:purple", linewidth=2.0,
        label=f"Position error (x,y): mean={e_mean:.2f} m, max={e_max:.2f} m"
    )
    ax2.plot(
        t, rmse_running, "k--", linewidth=2.0,
        label=f"Running RMSE: final={overall:.2f} m"
    )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position Error (m)")
    ax2.set_title(f"Tracking RMSE (m) = {overall:.2f}")
    ax2.grid(True, linestyle=":", linewidth=0.8)
    ax2.legend(loc="best")

    save_path_rmse = os.path.join(out_dir, "rmse_imm_kf.png")
    fig2.tight_layout()
    fig2.savefig(save_path_rmse, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"Saved RMSE figure to: {save_path_rmse}")


def main():
    dt = 0.5  # seconds

    sensor_pos = (100.0, 0.0, 1000.0)
    z0 = 100.0
    sigma_xy = 3.0
    sigma_r = 2.0

    t, truth = simulate_truth(dt=dt, z0=z0, seed=0)
    meas = simulate_measurements(
        t, truth, sensor_pos=sensor_pos, z0=z0,
        sigma_xy=sigma_xy, sigma_r=sigma_r, seed=1
    )

    est = run_imm_kf(t, meas, dt)

    make_plots(t, truth, meas, est, sensor_pos=sensor_pos, out_dir="simulator/imgs", dpi=220)


if __name__ == "__main__":
    main()