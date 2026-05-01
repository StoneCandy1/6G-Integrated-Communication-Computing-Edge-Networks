import math
import copy
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 6G Integrated Communication-Computing Offloading Game
# Complete simulation code
#
# Main features:
# 1. Multi-user single-edge integrated communication-computing model
# 2. Continuous offloading ratio x_i in [0, 1]
# 3. Utility includes delay, energy, and payment cost
# 4. Best-response dynamics (BRD) for Nash-like equilibrium search
# 5. Baselines:
#       - All Local
#       - All Offloading
#       - Random Offloading
#       - Proposed Game-Theoretic Scheme
# 6. Plots:
#       - Convergence curve
#       - Avg utility vs number of users
#       - Avg delay vs number of users
#       - Avg offloading ratio vs price
#
# You can directly run this file with Python.
# ============================================================


# ============================================================
# Section 1. Global plotting configuration
# ============================================================

# Try to make plots readable.
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 10


# ============================================================
# Section 2. Parameter generation
# ============================================================

def generate_system_parameters(
    N,
    rng,
    B=10e6,
    sigma2=1e-9,
    F_edge=10e9,
    p=1e-9
):
    """
    Generate one random system instance.

    Parameters
    ----------
    N : int
        Number of users.
    rng : np.random.Generator
        Random generator for reproducibility.
    B : float
        System bandwidth in Hz.
    sigma2 : float
        Noise power.
    F_edge : float
        Total edge computing capability in cycles/s.
    p : float
        Edge price per CPU cycle.

    Returns
    -------
    params : dict
        Dictionary containing all user/system parameters.
    """

    # Task input data size D_i: 0.5 ~ 2.0 Mbit
    D = rng.uniform(0.5e6, 2.0e6, size=N)

    # Required CPU cycles C_i: 0.5 ~ 2.0 x 10^9 cycles
    C = rng.uniform(0.5e9, 2.0e9, size=N)

    # Local CPU frequency f_i^loc: 0.5 ~ 1.5 x 10^9 cycles/s
    f_loc = rng.uniform(0.5e9, 1.5e9, size=N)

    # Transmit power P_i: fixed 0.5 W
    P = np.full(N, 0.5)

    # Channel gain h_i
    h = rng.uniform(5e-5, 2e-4, size=N)

    # Local energy coefficient kappa_i
    kappa = np.full(N, 1e-27)

    # Base task utility / reward
    V = np.full(N, 10.0)

    # Utility weights
    alpha = np.full(N, 2.0)   # delay weight
    beta = np.full(N, 1.0)    # energy weight
    gamma = np.full(N, 1.0)   # payment weight

    # Uplink rate R_i
    R = B * np.log2(1.0 + (P * h) / sigma2)

    params = {
        "N": N,
        "B": B,
        "sigma2": sigma2,
        "F_edge": F_edge,
        "p": p,
        "D": D,
        "C": C,
        "f_loc": f_loc,
        "P": P,
        "h": h,
        "kappa": kappa,
        "V": V,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "R": R,
    }
    return params


# ============================================================
# Section 3. Core model functions
# ============================================================

def compute_edge_delay(x, C, F_edge):
    """
    Compute the common edge execution delay under proportional allocation.

    According to the model:
        T_edge = sum_j(x_j C_j) / F_edge

    If nobody offloads, edge delay is set to 0.

    Parameters
    ----------
    x : np.ndarray
        Offloading ratios, shape (N,)
    C : np.ndarray
        CPU cycle requirements, shape (N,)
    F_edge : float
        Total edge computing capability.

    Returns
    -------
    T_edge : float
        Common edge delay.
    """
    total_offloaded_cycles = np.sum(x * C)
    if total_offloaded_cycles <= 0:
        return 0.0
    return total_offloaded_cycles / F_edge


def compute_user_metrics(i, x, params):
    """
    Compute all relevant metrics for user i.

    Returns
    -------
    metrics : dict
        Contains:
        - T_tx
        - T_loc
        - T_edge
        - T_total
        - E_tx
        - E_loc
        - payment
        - utility
    """
    D = params["D"]
    C = params["C"]
    f_loc = params["f_loc"]
    P = params["P"]
    kappa = params["kappa"]
    V = params["V"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    R = params["R"]
    F_edge = params["F_edge"]
    p = params["p"]

    xi = x[i]

    # Transmission delay
    T_tx = (xi * D[i]) / R[i]

    # Local computation delay
    T_loc = ((1.0 - xi) * C[i]) / f_loc[i]

    # Common edge delay caused by all users' offloading
    T_edge = compute_edge_delay(x, C, F_edge)

    # Total completion delay: parallel branches, take the slower one
    T_total = max(T_loc, T_tx + T_edge)

    # Transmission energy
    E_tx = P[i] * T_tx

    # Local computation energy
    E_loc = kappa[i] * (f_loc[i] ** 2) * ((1.0 - xi) * C[i])

    # Payment
    payment = p * xi * C[i]

    # Utility
    utility = (
        V[i]
        - alpha[i] * T_total
        - beta[i] * (E_loc + E_tx)
        - gamma[i] * payment
    )

    return {
        "T_tx": T_tx,
        "T_loc": T_loc,
        "T_edge": T_edge,
        "T_total": T_total,
        "E_tx": E_tx,
        "E_loc": E_loc,
        "payment": payment,
        "utility": utility,
    }


def compute_all_metrics(x, params):
    """
    Compute system-wide metrics for a given strategy profile x.

    Returns
    -------
    results : dict
        Contains per-user arrays and system averages.
    """
    N = params["N"]

    T_tx_all = np.zeros(N)
    T_loc_all = np.zeros(N)
    T_edge_all = np.zeros(N)
    T_total_all = np.zeros(N)
    E_tx_all = np.zeros(N)
    E_loc_all = np.zeros(N)
    pay_all = np.zeros(N)
    U_all = np.zeros(N)

    for i in range(N):
        m = compute_user_metrics(i, x, params)
        T_tx_all[i] = m["T_tx"]
        T_loc_all[i] = m["T_loc"]
        T_edge_all[i] = m["T_edge"]
        T_total_all[i] = m["T_total"]
        E_tx_all[i] = m["E_tx"]
        E_loc_all[i] = m["E_loc"]
        pay_all[i] = m["payment"]
        U_all[i] = m["utility"]

    results = {
        "x": x.copy(),
        "avg_offloading": np.mean(x),
        "avg_delay": np.mean(T_total_all),
        "avg_energy": np.mean(E_tx_all + E_loc_all),
        "avg_utility": np.mean(U_all),
        "social_welfare": np.sum(U_all),
        "user_utilities": U_all,
        "user_delays": T_total_all,
        "user_energies": E_tx_all + E_loc_all,
        "edge_load": np.sum(x * params["C"]),
    }
    return results


# ============================================================
# Section 4. Best-response dynamics
# ============================================================

def best_response_for_user(i, x_current, params, grid):
    """
    Compute the best response of user i by exhaustive search on the grid.

    Parameters
    ----------
    i : int
        User index.
    x_current : np.ndarray
        Current system strategy profile.
    params : dict
        System parameters.
    grid : np.ndarray
        Discretized candidate offloading ratios.

    Returns
    -------
    best_xi : float
        Best response value on the grid.
    best_utility : float
        Corresponding utility.
    """
    best_xi = x_current[i]
    best_utility = -np.inf

    for candidate in grid:
        x_trial = x_current.copy()
        x_trial[i] = candidate
        utility = compute_user_metrics(i, x_trial, params)["utility"]

        if utility > best_utility:
            best_utility = utility
            best_xi = candidate

    return best_xi, best_utility


def run_brd(
    params,
    grid_step=0.01,
    max_iter=200,
    tol=1e-4,
    init_x=None,
    sequential_update=True,
    verbose=False
):
    """
    Run best-response dynamics to obtain a stable offloading profile.

    Parameters
    ----------
    params : dict
        System parameters.
    grid_step : float
        Step size for discretizing [0, 1].
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence threshold.
    init_x : np.ndarray or None
        Initial offloading ratios. If None, initialize at 0.5 for all users.
    sequential_update : bool
        If True, use Gauss-Seidel style update:
            each user's update immediately affects later users in the same round.
        If False, use synchronous update.
    verbose : bool
        Whether to print iteration details.

    Returns
    -------
    history : dict
        Contains:
        - x_list
        - avg_utility_list
        - avg_delay_list
        - avg_offloading_list
        - final_x
        - final_results
        - iterations
    """
    N = params["N"]
    grid = np.round(np.arange(0.0, 1.0 + grid_step / 2, grid_step), 10)

    if init_x is None:
        x = np.full(N, 0.5)
    else:
        x = init_x.copy()

    x_list = [x.copy()]
    avg_utility_list = []
    avg_delay_list = []
    avg_offloading_list = []

    # Record initial metrics
    initial_results = compute_all_metrics(x, params)
    avg_utility_list.append(initial_results["avg_utility"])
    avg_delay_list.append(initial_results["avg_delay"])
    avg_offloading_list.append(initial_results["avg_offloading"])

    for t in range(max_iter):
        x_old = x.copy()

        if sequential_update:
            # Sequential update: later users see earlier updated strategies
            for i in range(N):
                best_xi, _ = best_response_for_user(i, x, params, grid)
                x[i] = best_xi
        else:
            # Synchronous update: all users use x_old
            x_new = x.copy()
            for i in range(N):
                best_xi, _ = best_response_for_user(i, x_old, params, grid)
                x_new[i] = best_xi
            x = x_new

        results = compute_all_metrics(x, params)
        avg_utility_list.append(results["avg_utility"])
        avg_delay_list.append(results["avg_delay"])
        avg_offloading_list.append(results["avg_offloading"])
        x_list.append(x.copy())

        diff = np.linalg.norm(x - x_old)

        if verbose:
            print(
                f"Iter {t+1:03d} | diff={diff:.6f} | "
                f"avg_x={results['avg_offloading']:.4f} | "
                f"avg_U={results['avg_utility']:.4f} | "
                f"avg_T={results['avg_delay']:.4f}"
            )

        if diff < tol:
            break

    final_results = compute_all_metrics(x, params)

    history = {
        "x_list": x_list,
        "avg_utility_list": avg_utility_list,
        "avg_delay_list": avg_delay_list,
        "avg_offloading_list": avg_offloading_list,
        "final_x": x.copy(),
        "final_results": final_results,
        "iterations": len(x_list) - 1,
    }
    return history


# ============================================================
# Section 5. Baseline schemes
# ============================================================

def strategy_all_local(params):
    """All users execute fully locally."""
    return np.zeros(params["N"])


def strategy_all_offloading(params):
    """All users fully offload."""
    return np.ones(params["N"])


def strategy_random(params, rng):
    """Each user selects a random offloading ratio in [0, 1]."""
    return rng.uniform(0.0, 1.0, size=params["N"])


def evaluate_baselines(params, rng, random_trials=20):
    """
    Evaluate all baselines and the proposed scheme.

    Parameters
    ----------
    params : dict
        System parameters.
    rng : np.random.Generator
        Random generator.
    random_trials : int
        Number of random baseline trials, then average them.

    Returns
    -------
    summary : dict
        Metrics for all methods.
    """
    # All Local
    x_local = strategy_all_local(params)
    res_local = compute_all_metrics(x_local, params)

    # All Offloading
    x_off = strategy_all_offloading(params)
    res_off = compute_all_metrics(x_off, params)

    # Random baseline: average over multiple trials
    random_metrics = []
    for _ in range(random_trials):
        x_rand = strategy_random(params, rng)
        random_metrics.append(compute_all_metrics(x_rand, params))

    res_rand = {
        "avg_offloading": np.mean([r["avg_offloading"] for r in random_metrics]),
        "avg_delay": np.mean([r["avg_delay"] for r in random_metrics]),
        "avg_energy": np.mean([r["avg_energy"] for r in random_metrics]),
        "avg_utility": np.mean([r["avg_utility"] for r in random_metrics]),
        "social_welfare": np.mean([r["social_welfare"] for r in random_metrics]),
    }

    # Proposed scheme
    brd_history = run_brd(params)
    res_game = brd_history["final_results"]

    summary = {
        "All Local": res_local,
        "All Offloading": res_off,
        "Random": res_rand,
        "Proposed Game": res_game,
        "BRD History": brd_history,
    }
    return summary


# ============================================================
# Section 6. Repeated experiment helpers
# ============================================================

def run_repeated_experiment_for_N(
    N,
    num_runs=30,
    base_seed=2026,
    p=1e-9,
    F_edge=10e9
):
    """
    Repeat the experiment multiple times for a given number of users N.

    Returns
    -------
    averaged : dict
        Mean metrics of all methods across repeated random instances.
    """
    methods = ["All Local", "All Offloading", "Random", "Proposed Game"]

    collected = {
        m: {
            "avg_utility": [],
            "avg_delay": [],
            "avg_energy": [],
            "avg_offloading": [],
            "social_welfare": [],
        }
        for m in methods
    }

    for run_idx in range(num_runs):
        rng = np.random.default_rng(base_seed + run_idx)
        params = generate_system_parameters(N=N, rng=rng, p=p, F_edge=F_edge)
        summary = evaluate_baselines(params, rng)

        for m in methods:
            collected[m]["avg_utility"].append(summary[m]["avg_utility"])
            collected[m]["avg_delay"].append(summary[m]["avg_delay"])
            collected[m]["avg_energy"].append(summary[m]["avg_energy"])
            collected[m]["avg_offloading"].append(summary[m]["avg_offloading"])
            collected[m]["social_welfare"].append(summary[m]["social_welfare"])

    averaged = {
        m: {k: float(np.mean(v)) for k, v in collected[m].items()}
        for m in methods
    }
    return averaged


def run_price_sweep(
    N=10,
    p_values=None,
    num_runs=20,
    base_seed=5000,
    F_edge=10e9
):
    """
    Sweep price p and record the proposed scheme's average offloading ratio,
    average utility, and average delay.

    Returns
    -------
    result : dict
        Contains lists matched with p_values.
    """
    if p_values is None:
        p_values = np.linspace(1e-10, 8e-9, 10)

    avg_x_list = []
    avg_U_list = []
    avg_T_list = []

    for idx, p in enumerate(p_values):
        avg_x_runs = []
        avg_U_runs = []
        avg_T_runs = []

        for run_idx in range(num_runs):
            rng = np.random.default_rng(base_seed + idx * 1000 + run_idx)
            params = generate_system_parameters(N=N, rng=rng, p=p, F_edge=F_edge)
            brd_history = run_brd(params)
            res = brd_history["final_results"]

            avg_x_runs.append(res["avg_offloading"])
            avg_U_runs.append(res["avg_utility"])
            avg_T_runs.append(res["avg_delay"])

        avg_x_list.append(np.mean(avg_x_runs))
        avg_U_list.append(np.mean(avg_U_runs))
        avg_T_list.append(np.mean(avg_T_runs))

    return {
        "p_values": np.array(p_values),
        "avg_offloading": np.array(avg_x_list),
        "avg_utility": np.array(avg_U_list),
        "avg_delay": np.array(avg_T_list),
    }

def run_edge_capacity_sweep(
    N=10,
    F_edge_values=None,
    num_runs=20,
    base_seed=12000,
    p=1e-9
):
    """
    Sweep edge computing capability F_edge and record the proposed scheme's
    average offloading ratio, average utility, and average delay.

    Parameters
    ----------
    N : int
        Number of users.
    F_edge_values : array-like or None
        Candidate edge computing capability values (cycles/s).
    num_runs : int
        Number of repeated random trials for averaging.
    base_seed : int
        Base seed for reproducibility.
    p : float
        Fixed edge price.

    Returns
    -------
    result : dict
        Contains:
        - F_edge_values
        - avg_offloading
        - avg_utility
        - avg_delay
    """
    if F_edge_values is None:
        F_edge_values = np.array([4e9, 6e9, 8e9, 10e9, 12e9, 14e9, 16e9])

    avg_x_list = []
    avg_U_list = []
    avg_T_list = []

    for idx, F_edge in enumerate(F_edge_values):
        avg_x_runs = []
        avg_U_runs = []
        avg_T_runs = []

        for run_idx in range(num_runs):
            rng = np.random.default_rng(base_seed + idx * 1000 + run_idx)

            params = generate_system_parameters(
                N=N,
                rng=rng,
                p=p,
                F_edge=F_edge
            )

            brd_history = run_brd(params)
            res = brd_history["final_results"]

            avg_x_runs.append(res["avg_offloading"])
            avg_U_runs.append(res["avg_utility"])
            avg_T_runs.append(res["avg_delay"])

        avg_x_list.append(np.mean(avg_x_runs))
        avg_U_list.append(np.mean(avg_U_runs))
        avg_T_list.append(np.mean(avg_T_runs))

    return {
        "F_edge_values": np.array(F_edge_values),
        "avg_offloading": np.array(avg_x_list),
        "avg_utility": np.array(avg_U_list),
        "avg_delay": np.array(avg_T_list),
    }


# ============================================================
# Section 7. Plotting functions
# ============================================================

def plot_convergence(brd_history):
    """
    Plot BRD convergence curves.
    """
    iters = np.arange(len(brd_history["avg_utility_list"]))

    plt.figure(figsize=(5.4, 3.8))
    plt.plot(iters, brd_history["avg_utility_list"], marker="o", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Average Utility")
    plt.title("BRD Convergence of Average Utility")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5.4, 3.8))
    plt.plot(iters, brd_history["avg_offloading_list"], marker="o", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Average Offloading Ratio")
    plt.title("BRD Convergence of Average Offloading Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_metric_vs_users(user_counts, all_results, metric_key, ylabel, title):
    """
    Plot a selected metric versus number of users for all methods.
    """
    methods = ["All Local", "All Offloading", "Random", "Proposed Game"]

    plt.figure(figsize=(5.8, 4.0))
    for method in methods:
        y = [all_results[N][method][metric_key] for N in user_counts]
        plt.plot(user_counts, y, marker="o", label=method)

    plt.xlabel("Number of Users")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_price_sweep(price_result):
    """
    Plot how price affects the proposed scheme.
    """
    p_values = price_result["p_values"]

    plt.figure(figsize=(5.6, 3.9))
    plt.plot(p_values, price_result["avg_offloading"], marker="o")
    plt.xlabel("Edge Price p")
    plt.ylabel("Average Offloading Ratio")
    plt.title("Average Offloading Ratio vs Edge Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5.6, 3.9))
    plt.plot(p_values, price_result["avg_utility"], marker="o")
    plt.xlabel("Edge Price p")
    plt.ylabel("Average Utility")
    plt.title("Average Utility vs Edge Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_edge_capacity_sweep(edge_result):
    """
    Plot how edge computing capability affects the proposed scheme.
    """
    F_values = edge_result["F_edge_values"] / 1e9  # convert to 1e9 cycles/s for readability

    plt.figure(figsize=(5.6, 3.9))
    plt.plot(F_values, edge_result["avg_utility"], marker="o")
    plt.xlabel("Edge Computing Capability F_edge (×1e9 cycles/s)")
    plt.ylabel("Average Utility")
    plt.title("Average Utility vs Edge Computing Capability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5.6, 3.9))
    plt.plot(F_values, edge_result["avg_offloading"], marker="o")
    plt.xlabel("Edge Computing Capability F_edge (×1e9 cycles/s)")
    plt.ylabel("Average Offloading Ratio")
    plt.title("Average Offloading Ratio vs Edge Computing Capability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_delay_vs_price(price_result):
    """
    Plot how edge service price affects average delay.
    """
    p_values = price_result["p_values"]

    plt.figure(figsize=(5.6, 3.9))
    plt.plot(p_values, price_result["avg_delay"], marker="o")
    plt.xlabel("Edge Price p")
    plt.ylabel("Average Delay (s)")
    plt.title("Average Delay vs Edge Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# Section 8. Utility printing helpers
# ============================================================

def print_single_instance_summary(summary):
    """
    Print readable summary for one random instance.
    """
    print("=" * 72)
    print("Single-instance evaluation summary")
    print("=" * 72)

    methods = ["All Local", "All Offloading", "Random", "Proposed Game"]
    for method in methods:
        res = summary[method]
        print(f"\n[{method}]")
        print(f"  Average utility     : {res['avg_utility']:.6f}")
        print(f"  Average delay       : {res['avg_delay']:.6f}")
        print(f"  Average energy      : {res['avg_energy']:.6f}")
        print(f"  Average offloading  : {res['avg_offloading']:.6f}")
        print(f"  Social welfare      : {res['social_welfare']:.6f}")

    brd_history = summary["BRD History"]
    print("\n[BRD details]")
    print(f"  Iterations          : {brd_history['iterations']}")
    print(f"  Final x             : {np.round(brd_history['final_x'], 3)}")
    print("=" * 72)


def print_multi_user_summary(user_counts, all_results):
    """
    Print averaged results across different N values.
    """
    print("\n" + "=" * 72)
    print("Averaged results across different user counts")
    print("=" * 72)

    methods = ["All Local", "All Offloading", "Random", "Proposed Game"]
    for N in user_counts:
        print(f"\nN = {N}")
        for method in methods:
            res = all_results[N][method]
            print(
                f"  {method:15s} | "
                f"avg_U={res['avg_utility']:.4f}, "
                f"avg_T={res['avg_delay']:.4f}, "
                f"avg_x={res['avg_offloading']:.4f}"
            )


# ============================================================
# Section 9. Main experiment pipeline
# ============================================================

def main():
    """
    Main entry of the simulation.

    It performs:
    1. One single-instance experiment + BRD convergence
    2. Repeated experiments under different user counts
    3. Price sensitivity experiment
    """
    # --------------------------------------------------------
    # Part 1. Single-instance experiment
    # --------------------------------------------------------
    seed = 2026
    rng = np.random.default_rng(seed)

    params = generate_system_parameters(
        N=10,
        rng=rng,
        p=1e-9,
        F_edge=10e9
    )

    summary = evaluate_baselines(params, rng)
    print_single_instance_summary(summary)

    # Plot convergence of the proposed method on this instance
    plot_convergence(summary["BRD History"])

    # --------------------------------------------------------
    # Part 2. Repeated experiments for different user counts
    # --------------------------------------------------------
    user_counts = [5, 10, 15, 20, 25]
    all_results = {}

    for N in user_counts:
        all_results[N] = run_repeated_experiment_for_N(
            N=N,
            num_runs=25,
            base_seed=3000 + N,
            p=1e-9,
            F_edge=10e9
        )

    print_multi_user_summary(user_counts, all_results)

    # Plot average utility vs number of users
    plot_metric_vs_users(
        user_counts=user_counts,
        all_results=all_results,
        metric_key="avg_utility",
        ylabel="Average Utility",
        title="Average Utility vs Number of Users"
    )

    # Plot average delay vs number of users
    plot_metric_vs_users(
        user_counts=user_counts,
        all_results=all_results,
        metric_key="avg_delay",
        ylabel="Average Delay (s)",
        title="Average Delay vs Number of Users"
    )

    # --------------------------------------------------------
    # Part 3. Price sensitivity analysis
    # --------------------------------------------------------
    p_values = np.linspace(1e-10, 8e-9, 10)
    price_result = run_price_sweep(
        N=10,
        p_values=p_values,
        num_runs=20,
        base_seed=8000,
        F_edge=10e9
    )

    plot_price_sweep(price_result)

    # --------------------------------------------------------
    # Part 4. Edge computing capability sensitivity analysis
    # --------------------------------------------------------
    F_edge_values = np.array([4e9, 6e9, 8e9, 10e9, 12e9, 14e9, 16e9])

    edge_result = run_edge_capacity_sweep(
        N=10,
        F_edge_values=F_edge_values,
        num_runs=20,
        base_seed=12000,
        p=1e-9
    )

    plot_edge_capacity_sweep(edge_result)

# ============================================================
# Standard Python entry
# ============================================================

if __name__ == "__main__":
    main()