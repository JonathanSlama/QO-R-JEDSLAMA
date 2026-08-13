#!/usr/bin/env python3
"""
===============================================================================
QO+R THREE-BODY PROBLEM TEST
===============================================================================

Tests the QO+R prediction that phase coupling reduces chaos in N-body systems.

THEORETICAL BASIS:
==================
In QO+R, each body has position r_i AND internal phase θ_i.
The phase coupling term:
    -κ/r^ν (1 - cos(Δθ/2))
creates a tendency toward phase synchronization that should:

1. REDUCE Lyapunov exponents (chaos measure)
2. ENHANCE orbital stability for specific configurations
3. PRESERVE energy better than pure Newton

PREDICTION:
==========
λ_QOR ≤ λ_Newton for all configurations
Stability time T_QOR ≥ T_Newton

FALSIFICATION:
=============
If λ_QOR > λ_Newton systematically (>50%), the phase-coherence hypothesis is wrong.

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025
===============================================================================
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import json
import warnings
warnings.filterwarnings('ignore')

# Constants
G = 1.0  # Normalized units
M = 1.0  # Total mass normalized

# QO+R parameters
EPSILON_M = 0.1    # Phase feedback on gravity
KAPPA = 0.5        # Phase coupling strength
NU = 1.5           # Range exponent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


def newton_3body(t, state, masses):
    """Standard Newtonian 3-body equations."""
    n = 3
    pos = state[:2*n].reshape(n, 2)
    vel = state[2*n:].reshape(n, 2)
    
    acc = np.zeros_like(pos)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                if r > 1e-10:
                    acc[i] += G * masses[j] * r_vec / r**3
    
    return np.concatenate([vel.flatten(), acc.flatten()])


def qor_3body(t, state, masses, epsilon_m=EPSILON_M, kappa=KAPPA, nu=NU):
    """QO+R 3-body equations with phase coupling."""
    n = 3
    
    pos = state[:2*n].reshape(n, 2)
    vel = state[2*n:4*n].reshape(n, 2)
    theta = state[4*n:4*n+n]
    omega = state[4*n+n:]
    
    acc = np.zeros_like(pos)
    alpha = np.zeros(n)
    
    Q = np.cos(theta)
    R = np.sin(theta)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                
                if r > 1e-10:
                    delta_theta = theta[i] - theta[j]
                    Xi_ij = 0.5 * (Q[i] * R[j] + Q[j] * R[i])
                    phase_factor = 1 + epsilon_m * Xi_ij * np.cos(delta_theta / 2)
                    
                    acc[i] += G * masses[j] * phase_factor * r_vec / r**3
                    alpha[i] -= (kappa / r**nu) * 0.5 * np.sin(delta_theta / 2)
    
    d_pos = vel.flatten()
    d_vel = acc.flatten()
    d_theta = omega
    d_omega = alpha / (masses * 0.1)
    
    return np.concatenate([d_pos, d_vel, d_theta, d_omega])


def lyapunov_exponent(trajectory, dt):
    """Estimate maximum Lyapunov exponent."""
    n_points = len(trajectory)
    if n_points < 100:
        return np.nan
    
    diffs = []
    for i in range(1, n_points-1):
        d1 = np.linalg.norm(trajectory[i] - trajectory[i-1])
        d2 = np.linalg.norm(trajectory[i+1] - trajectory[i])
        if d1 > 1e-15:
            diffs.append(np.log(d2 / d1 + 1e-15))
    
    if len(diffs) > 10:
        return np.mean(diffs) / dt
    return np.nan


def run_simulation(ic, masses, t_max=50, dt=0.01, model='newton'):
    """Run a 3-body simulation."""
    t_eval = np.arange(0, t_max, dt)
    
    if model == 'newton':
        state0 = ic[:12]
        sol = solve_ivp(
            newton_3body, [0, t_max], state0, args=(masses,),
            t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12
        )
    else:
        theta0 = np.random.uniform(0, 2*np.pi, 3)
        omega0 = np.zeros(3)
        state0 = np.concatenate([ic[:12], theta0, omega0])
        sol = solve_ivp(
            qor_3body, [0, t_max], state0, args=(masses,),
            t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12
        )
    
    trajectory = sol.y.T
    lyap = lyapunov_exponent(trajectory, dt)
    
    return {'lyapunov': lyap, 'success': sol.success}


# Initial conditions
def figure_8():
    return np.array([-0.97000436, 0.24308753, 0.0, 0.0, 0.97000436, -0.24308753,
                     0.4662036850, 0.4323657300, -0.93240737, -0.86473146, 
                     0.4662036850, 0.4323657300])

def lagrange():
    r = 1.0
    pos = np.array([[r*np.cos(0), r*np.sin(0)], 
                    [r*np.cos(2*np.pi/3), r*np.sin(2*np.pi/3)],
                    [r*np.cos(4*np.pi/3), r*np.sin(4*np.pi/3)]])
    omega = np.sqrt(G * M / r**3)
    vel = np.array([[-omega*r*np.sin(0), omega*r*np.cos(0)],
                    [-omega*r*np.sin(2*np.pi/3), omega*r*np.cos(2*np.pi/3)],
                    [-omega*r*np.sin(4*np.pi/3), omega*r*np.cos(4*np.pi/3)]])
    return np.concatenate([pos.flatten(), vel.flatten()])

def random_ic(seed):
    np.random.seed(seed)
    pos = np.random.uniform(-1, 1, (3, 2))
    vel = np.random.uniform(-0.5, 0.5, (3, 2))
    pos -= np.mean(pos, axis=0)
    vel -= np.mean(vel, axis=0)
    return np.concatenate([pos.flatten(), vel.flatten()])


def main():
    print("="*70)
    print(" QO+R THREE-BODY PROBLEM TEST")
    print(" Author: Jonathan Édouard Slama")
    print(" Testing: λ_QOR ≤ λ_Newton (phase-coherence reduces chaos)")
    print("="*70)
    
    masses = np.array([M/3, M/3, M/3])
    
    configs = {
        'Figure-8': figure_8(),
        'Lagrange': lagrange(),
        'Random_1': random_ic(42),
        'Random_2': random_ic(43),
        'Random_3': random_ic(44),
    }
    
    print(f"\n{'Config':<15} {'Model':<8} {'Lyapunov':<12} {'Chaos Reduced?':<15}")
    print("-"*55)
    
    n_reduced = 0
    n_total = 0
    results = {}
    
    for name, ic in configs.items():
        res_n = run_simulation(ic, masses, model='newton')
        res_q = run_simulation(ic, masses, model='qor')
        
        lyap_n = res_n['lyapunov']
        lyap_q = res_q['lyapunov']
        
        results[name] = {'newton': lyap_n, 'qor': lyap_q}
        
        if np.isfinite(lyap_n) and np.isfinite(lyap_q):
            n_total += 1
            reduced = lyap_q <= lyap_n
            if reduced:
                n_reduced += 1
            status = "YES ✓" if reduced else "NO ✗"
        else:
            status = "N/A"
        
        lyap_n_str = f"{lyap_n:.4f}" if np.isfinite(lyap_n) else "N/A"
        lyap_q_str = f"{lyap_q:.4f}" if np.isfinite(lyap_q) else "N/A"
        
        print(f"{name:<15} {'Newton':<8} {lyap_n_str:<12}")
        print(f"{'':<15} {'QO+R':<8} {lyap_q_str:<12} {status}")
    
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    if n_total > 0:
        rate = n_reduced / n_total
        print(f"\nChaos reduction rate: {n_reduced}/{n_total} = {rate*100:.1f}%")
        
        if rate >= 0.7:
            status = "CONFIRMED"
        elif rate >= 0.5:
            status = "CONSISTENT"
        else:
            status = "FALSIFIED"
        
        print(f"\nPREDICTION STATUS: {status}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results['summary'] = {
        'n_total': n_total,
        'n_reduced': n_reduced,
        'rate': n_reduced / n_total if n_total > 0 else 0,
        'status': status
    }
    
    with open(os.path.join(OUTPUT_DIR, 'three_body_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("="*70)
    return results


if __name__ == "__main__":
    main()
