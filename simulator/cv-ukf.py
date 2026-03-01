import os
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# Function to simulate the 2D motion of the drone
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
            v_new = 15
            omega3 = +v_new / R3
            psi_new = psi_prev + omega3 * dt
        elif ti < edges[4]:
            v_new = max(v_prev + a4 * dt, 0.0)
            psi_new = psi_prev
        else:
            v_new = max(7.0, 1e-6)
            omega5 = -v_new / R5
            psi_new = psi_prev + omega5 * dt

        vx[k] = v_new * np.cos(psi_new)
        vy[k] = v_new * np.sin(psi_new)
        x[k] = x[k - 1] + vx[k] * dt
        y[k] = y[k - 1] + vy[k] * dt
        psi[k] = psi_new
        vmag[k] = v_new

    return {"t": t, "x": x, "y": y, "vx": vx, "vy": vy, "psi": psi, "vmag": vmag, "z0": z0}

# Function to generate measurements with noise
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

# Define the Unscented Kalman Filter (UKF) for CV model
class CVUKF2D:
    def __init__(self, dt, q=0.8):
        self.dt = dt  # Use dt for the time step directly
        self.q = q
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0
        self.I = np.eye(4)

        # Define the sigma points
        points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0)

        # Initialize UnscentedKalmanFilter with required parameters
        self.ukf = UnscentedKalmanFilter(dim_x=4, dim_z=3, fx=self.F, hx=self.h, points=points, dt=self.dt)

    def F(self, state, dt):
        px, py, vx, vy = state
        return np.array([px + vx * dt, py + vy * dt, vx, vy], dtype=float)

    def h(self, state, sensor_pos, z0):
        px, py, vx, vy = state
        xs, ys, zs = sensor_pos
        dx = px - xs
        dy = py - ys
        dz = z0 - zs
        rho = np.sqrt(dx * dx + dy * dy + dz * dz)
        return np.array([px, py, rho], dtype=float)

    def predict(self):
        self.ukf.predict()

    def update(self, z, R, sensor_pos, z0):
        # Pass the sensor_pos and z0 parameters to the update function
        self.ukf.update(z, R, sensor_pos=sensor_pos, z0=z0)

# Function to ensure directories exist
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Function to compute RMSE between truth and estimate
def rmse_series(truth_xy, est_xy):
    e = est_xy - truth_xy
    err = np.sqrt(np.sum(e * e, axis=1))
    rmse = np.sqrt(np.mean(err ** 2))
    return err, rmse

# Function to set reasonable limits for the plot axes
def set_reasonable_limits(ax, x_all, y_all, pad_ratio=0.12):
    xmin, xmax = np.min(x_all), np.max(x_all)
    ymin, ymax = np.min(y_all), np.max(y_all)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    ax.set_xlim(xmin - dx * pad_ratio, xmax + dx * pad_ratio)
    ax.set_ylim(ymin - dy * pad_ratio, ymax + dy * pad_ratio)
    ax.set_aspect("equal", adjustable="box")

# Main function to simulate and use the CV-UKF for tracking
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
    save_path_track = os.path.join(out_dir, "cv_ukf_tracking.png")
    save_path_rmse = os.path.join(out_dir, "rmse_cv_ukf.png")

    rng = np.random.default_rng(42)

    truth = simulate_truth_2d_piecewise(
        dt=dt, z0=z0,
        x0=x0, y0=y0, v0=v0, psi0=psi0,
        T1=T1, T2=T2, a2=a2, T3=T3, R3=R3,
        T4=T4, a4=a4, T5=T5, R5=R5,
    )

    meas = make_measurements_xy_rho(
        truth,
        sensor_pos=sensor_pos,
        sigma0_xy=sigma0_xy, k_xy=k_xy,
        sigma0_rho=sigma0_rho, k_rho=k_rho,
        rng=rng
    )
    z = meas["z"]
    R_list = meas["R_list"]

    # Initialize the UKF filter
    ukf = CVUKF2D(dt=dt, q=q)
    ukf.x = np.array([z[0, 0], z[0, 1], 0.0, 0.0], dtype=float)

    N = len(truth["t"])
    xhat = np.zeros((N, 4))
    for i in range(N):
        ukf.predict()
        ukf.update(z[i], R_list[i], meas["sensor_pos"], truth["z0"])
        xhat[i] = ukf.ukf.x

    truth_xy = np.column_stack([truth["x"], truth["y"]])
    est_xy = xhat[:, :2]
    err, rmse_val = rmse_series(truth_xy, est_xy)
    print(f"位置误差RMSE（欧氏距离, XY）= {rmse_val:.3f} m")

    # plot 1: Track trajectory
    fig1, ax1 = plt.subplots(figsize=(9.5, 7.5))
    ax1.plot(truth["x"], truth["y"], "k-", linewidth=2.0, label="Truth trajectory (Truth)")
    ax1.plot(z[:, 0], z[:, 1], "rx", markersize=4, alpha=0.65, label="Measurements (x, y)")
    ax1.plot(est_xy[:, 0], est_xy[:, 1], "b--", linewidth=2.0, label="Track trajectory (CV-UKF)")

    xs, ys, _ = sensor_pos
    ax1.scatter([xs], [ys], c="g", s=70, marker="^", label="Sensor")

    x_all = np.concatenate([truth["x"], z[:, 0], est_xy[:, 0], np.array([xs])])
    y_all = np.concatenate([truth["y"], z[:, 1], est_xy[:, 1], np.array([ys])])
    set_reasonable_limits(ax1, x_all, y_all, pad_ratio=0.12)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("CV-UKF Tracking Simulation")
    ax1.grid(True, linestyle=":", linewidth=0.8)
    ax1.legend(loc="best")

    fig1.tight_layout()
    fig1.savefig(save_path_track, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"已保存轨迹图到: {save_path_track}")

    # plot 2: RMSE error
    fig2, ax2 = plt.subplots(figsize=(10.5, 4.8))
    ax2.plot(truth["t"], err, color="tab:purple", linestyle="-", linewidth=2.0,
             label=f"Position error: mean={np.mean(err):.2f} m, max={np.max(err):.2f} m")
    ax2.axhline(rmse_val, color="k", linestyle="--", linewidth=2.0,
                label=f"RMSE = {rmse_val:.2f} m")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position Error (m)")
    ax2.set_title(f"Tracking RMSE (m) = {rmse_val:.2f}")
    ax2.grid(True, linestyle=":", linewidth=0.8)
    ax2.legend(loc="best")

    fig2.tight_layout()
    fig2.savefig(save_path_rmse, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"已保存RMSE图到: {save_path_rmse}")


if __name__ == "__main__":
    main()