import numpy as np
import os
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

    z = np.zeros((len(x), 3))  # Make sure z is defined here
    z[:, 0] = x + rng.normal(0.0, sigma_xy)
    z[:, 1] = y + rng.normal(0.0, sigma_xy)
    z[:, 2] = rho_true + rng.normal(0.0, sigma_rho)

    R_list = np.zeros((len(x), 3, 3))
    for i in range(len(x)):
        R_list[i] = np.diag([sigma_xy[i] ** 2, sigma_xy[i] ** 2, sigma_rho[i] ** 2])

    return {
        "z": z,               # Now returning z
        "R_list": R_list,
        "sensor_pos": np.array(sensor_pos, dtype=float),
        "rho_true": rho_true,
        "sigma_xy": sigma_xy,
        "sigma_rho": sigma_rho,
    }

# Define CV-UKF Model (Constant Velocity)
class CVUKF2D:
    def __init__(self, dt, q=0.8):
        self.dt = dt
        self.q = q
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0
        self.I = np.eye(4)

        # Define the sigma points
        points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0)

        # Initialize UnscentedKalmanFilter with required parameters
        self.ukf = UnscentedKalmanFilter(dim_x=4, dim_z=3, fx=self.F, hx=self.h, points=points, dt=self.dt)

    def F(self, state, dt):
        # Define the CV model dynamics
        px, py, vx, vy = state
        return np.array([px + vx * dt, py + vy * dt, vx, vy], dtype=float)

    def h(self, state, sensor_pos, z0):
        # Measurement function for CV model
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
        self.ukf.update(z, R, sensor_pos=sensor_pos, z0=z0)

# Define CT-UKF Model (Constant Turn)
class CTUKF2D:
    def __init__(self, dt, q=0.8):
        self.dt = dt
        self.q = q
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0
        self.I = np.eye(4)

        # Define the sigma points
        points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0)

        # Initialize UnscentedKalmanFilter with required parameters
        self.ukf = UnscentedKalmanFilter(dim_x=4, dim_z=3, fx=self.F, hx=self.h, points=points, dt=self.dt)

    def F(self, state, dt):
        # Define the CT model dynamics (constant turn)
        px, py, v, omega = state  # velocity and angular velocity
        return np.array([px + v * np.cos(omega) * dt, py + v * np.sin(omega) * dt, v, omega], dtype=float)

    def h(self, state, sensor_pos, z0):
        # Measurement function for CT model
        px, py, v, omega = state
        xs, ys, zs = sensor_pos
        dx = px - xs
        dy = py - ys
        dz = z0 - zs
        rho = np.sqrt(dx * dx + dy * dy + dz * dz)
        return np.array([px, py, rho], dtype=float)

    def predict(self):
        self.ukf.predict()

    def update(self, z, R, sensor_pos, z0):
        self.ukf.update(z, R, sensor_pos=sensor_pos, z0=z0)

# Define IMM-UKF Class to manage multiple models
class IMMUKF:
    def __init__(self, dt, q=0.8, models=None):
        self.dt = dt
        self.models = models if models else [CVUKF2D(dt, q), CTUKF2D(dt, q)]  # Two models: CV and CT
        self.N = len(models)
        self.state = np.zeros((self.N, 4))  # States for each model
        self.P = np.array([np.eye(4) * 100.0 for _ in range(self.N)])  # Covariance for each model
        self.weights = np.ones(self.N) / self.N  # Initialize weights equally
        
    def predict(self):
        for model in self.models:
            model.predict()
    
    def update(self, z, R, sensor_pos, z0):
        # Update each model's estimate
        for i, model in enumerate(self.models):
            model.update(z, R, sensor_pos, z0)
            self.state[i] = model.ukf.x

        # Calculate likelihoods and update model weights
        likelihoods = np.array([self.calculate_likelihood(i, z, R, sensor_pos, z0) for i in range(self.N)])
        self.weights = likelihoods / np.sum(likelihoods)

        # Combine state estimates based on weights
        self.combine_estimates()

    def calculate_likelihood(self, model_idx, z, R, sensor_pos, z0):
        # Likelihood calculation based on innovation
        innovation = z - self.models[model_idx].h(self.state[model_idx], sensor_pos, z0)
        S = self.models[model_idx].ukf.S
        return np.exp(-0.5 * innovation.T @ np.linalg.inv(S) @ innovation)

    def combine_estimates(self):
        weighted_sum = np.zeros(4)
        for i in range(self.N):
            weighted_sum += self.weights[i] * self.state[i]
        self.x = weighted_sum

# Function to compute RMSE between truth and estimate
def rmse_series(truth_xy, est_xy):
    e = est_xy - truth_xy
    err = np.sqrt(np.sum(e * e, axis=1))
    rmse = np.sqrt(np.mean(err ** 2))
    return err, rmse

# Function to ensure directories exist
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Main function to simulate and use the IMM-UKF for tracking
def main():
    dt = 0.5
    z0 = 100.0

    # motion parameters (same as before)
    x0, y0, v0, psi0 = 0.0, 0.0, 5.0, 0.0

    # sensor moved near (100,0,1000)
    sensor_pos = (100.0, 0.0, 1000.0)

    # noise parameters (same as before)
    sigma0_xy, k_xy = 1.5, 0.0012
    sigma0_rho, k_rho = 2.5, 0.0020

    # filter parameters
    q = 0.8

    # output directories (same as before)
    out_dir = os.path.join("simulator", "imgs")
    ensure_dir(out_dir)
    save_path_track = os.path.join(out_dir, "imm_ukf_tracking.png")
    save_path_rmse = os.path.join(out_dir, "rmse_imm_ukf.png")

    rng = np.random.default_rng(42)

    truth = simulate_truth_2d_piecewise(
        dt=dt, z0=z0,
        x0=x0, y0=y0, v0=v0, psi0=psi0,
        T1=25.0, T2=20.0, a2=+0.5, T3=15.0, R3=80.0,
        T4=20.0, a4=-0.4, T5=20.0, R5=120.0,
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

    # Initialize IMM-UKF filter with multiple models
    imm_ukf = IMMUKF(dt=dt, q=q, models=[CVUKF2D(dt, q), CTUKF2D(dt, q)])  # Example with 2 models: CV and CT

    # Tracking process
    N = len(truth["t"])
    xhat = np.zeros((N, 4))
    for i in range(N):
        imm_ukf.predict()
        imm_ukf.update(z[i], R_list[i], meas["sensor_pos"], truth["z0"])
        xhat[i] = imm_ukf.x

    truth_xy = np.column_stack([truth["x"], truth["y"]])
    est_xy = xhat[:, :2]
    err, rmse_val = rmse_series(truth_xy, est_xy)
    print(f"位置误差RMSE（欧氏距离, XY）= {rmse_val:.3f} m")

    # plot 1: Track trajectory
    fig1, ax1 = plt.subplots(figsize=(9.5, 7.5))
    ax1.plot(truth["x"], truth["y"], "k-", linewidth=2.0, label="Truth trajectory (Truth)")
    ax1.plot(z[:, 0], z[:, 1], "rx", markersize=4, alpha=0.65, label="Measurements (x, y)")
    ax1.plot(est_xy[:, 0], est_xy[:, 1], "b--", linewidth=2.0, label="Track trajectory (IMM-UKF)")

    ax1.scatter([sensor_pos[0]], [sensor_pos[1]], c="g", s=70, marker="^", label="Sensor")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("IMM-UKF Tracking Simulation")
    ax1.grid(True, linestyle=":", linewidth=0.8)
    ax1.legend(loc="best")

    fig1.tight_layout()
    fig1.savefig(save_path_track, dpi=220, bbox_inches="tight")
    plt.show()

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

if __name__ == "__main__":
    main()