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

TEST CONFIGURATIONS:
===================
1. Figure-8 orbit (Chenciner & Montgomery)
2. Lagrange equilateral triangle
3. Euler collinear configuration
4. Hierarchical triple (binary + distant third)
5. Random initial conditions (chaos comparison)

PREDICTION:
==========
λ_QOR ≤ λ_Newton for all configurations
Stability time T_QOR ≥ T_Newton

FALSIFICATION:
=============
If λ_QOR > λ_Newton systematically, the phase-coherence hypothesis is wrong.

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist
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
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "04_RESULTS", "test_results")


def newton_3body(t, state, masses):
    """
    Standard Newtonian 3-body equations.
    State: [x1,y1,x2,y2,x3,y3, vx1,vy1,vx2,vy2,vx3,vy3]
    """
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
    """
    QO+R 3-body equations with phase coupling.
    State: [x1,y1,x2,y2,x3,y3, vx1,vy1,vx2,vy2,vx3,vy3, θ1,θ2,θ3, ω1,ω2,ω3]
    
    Position dynamics:
    m_i r̈_i = Σ G m_i m_j / r_ij³ [1 + ε_m Ξ_ij cos(Δθ_ij/2)] (r_j - r_i)
    
    Phase dynamics:
    I_i θ̈_i = -Σ κ/r_ij^ν × (1/2) sin(Δθ_ij/2)
    """
    n = 3
    
    # Unpack state
    pos = state[:2*n].reshape(n, 2)
    vel = state[2*n:4*n].reshape(n, 2)
    theta = state[4*n:4*n+n]
    omega = state[4*n+n:]
    
    # Initialize accelerations
    acc = np.zeros_like(pos)
    alpha = np.zeros(n)  # Angular acceleration
    
    # Assume Q, R values (simplified: Q_i = cos(θ_i), R_i = sin(θ_i))
    Q = np.cos(theta)
    R = np.sin(theta)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                
                if r > 1e-10:
                    # Phase difference
                    delta_theta = theta[i] - theta[j]
                    
                    # Cross-coupling factor Ξ_ij = (Q_i R_j + Q_j R_i)/2
                    Xi_ij = 0.5 * (Q[i] * R[j] + Q[j] * R[i])
                    
                    # Phase modulation factor
                    phase_factor = 1 + epsilon_m * Xi_ij * np.cos(delta_theta / 2)
                    
                    # Modified gravitational acceleration
                    acc[i] += G * masses[j] * phase_factor * r_vec / r**3
                    
                    # Phase acceleration (tendency to synchronize)
                    alpha[i] -= (kappa / r**nu) * 0.5 * np.sin(delta_theta / 2)
    
    # Return derivatives
    d_pos = vel.flatten()
    d_vel = acc.flatten()
    d_theta = omega
    d_omega = alpha / (masses * 0.1)  # I_i ~ 0.1 * m_i (moment of inertia estimate)
    
    return np.concatenate([d_pos, d_vel, d_theta, d_omega])


def compute_energy_newton(state, masses):
    """Compute total energy for Newton."""
    n = 3
    pos = state[:2*n].reshape(n, 2)
    vel = state[2*n:4*n].reshape(n, 2)
    
    # Kinetic energy
    T = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)
    
    # Potential energy
    U = 0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[j] - pos[i])
            if r > 1e-10:
                U -= G * masses[i] * masses[j] / r
    
    return T + U


def compute_energy_qor(state, masses, epsilon_m=EPSILON_M, kappa=KAPPA, nu=NU):
    """Compute total energy for QO+R."""
    n = 3
    pos = state[:2*n].reshape(n, 2)
    vel = state[2*n:4*n].reshape(n, 2)
    theta = state[4*n:4*n+n]
    omega = state[4*n+n:]
    
    Q = np.cos(theta)
    R = np.sin(theta)
    
    # Kinetic energy (translational)
    T_trans = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)
    
    # Kinetic energy (rotational/phase)
    I = masses * 0.1
    T_rot = 0.5 * np.sum(I * omega**2)
    
    # Gravitational potential (modified)
    U_grav = 0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[j] - pos[i])
            delta_theta = theta[i] - theta[j]
            Xi_ij = 0.5 * (Q[i] * R[j] + Q[j] * R[i])
            
            if r > 1e-10:
                phase_factor = 1 + epsilon_m * Xi_ij * np.cos(delta_theta / 2)
                U_grav -= G * masses[i] * masses[j] * phase_factor / r
    
    # Phase coupling potential
    U_phase = 0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[j] - pos[i])
            delta_theta = theta[i] - theta[j]
            if r > 1e-10:
                U_phase += (kappa / r**nu) * (1 - np.cos(delta_theta / 2))
    
    return T_trans + T_rot + U_grav + U_phase


def lyapunov_exponent(trajectory, dt):
    """
    Estimate maximum Lyapunov exponent from trajectory divergence.
    Uses crude method based on phase space volume evolution.
    """
    n_points = len(trajectory)
    if n_points < 100:
        return np.nan
    
    # Compute local divergence
    diffs = []
    for i in range(1, n_points-1):
        d1 = np.linalg.norm(trajectory[i] - trajectory[i-1])
        d2 = np.linalg.norm(trajectory[i+1] - trajectory[i])
        if d1 > 1e-15:
            diffs.append(np.log(d2 / d1 + 1e-15))
    
    if len(diffs) > 10:
        return np.mean(diffs) / dt
    return np.nan


def run_simulation(initial_conditions, masses, t_max=100, dt=0.01, model='newton'):
    """Run a 3-body simulation."""
    
    t_eval = np.arange(0, t_max, dt)
    
    if model == 'newton':
        # Pad state for Newton (no phases)
        state0 = initial_conditions[:12]  # Only positions and velocities
        
        sol = solve_ivp(
            newton_3body,
            [0, t_max],
            state0,
            args=(masses,),
            t_eval=t_eval,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )
        
        trajectory = sol.y.T
        energies = [compute_energy_newton(s, masses) for s in trajectory]
        
    else:  # QO+R
        # Full state including phases
        if len(initial_conditions) < 18:
            # Add initial phases (random or zero)
            theta0 = np.random.uniform(0, 2*np.pi, 3)
            omega0 = np.zeros(3)
            state0 = np.concatenate([initial_conditions[:12], theta0, omega0])
        else:
            state0 = initial_conditions
        
        sol = solve_ivp(
            qor_3body,
            [0, t_max],
            state0,
            args=(masses,),
            t_eval=t_eval,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )
        
        trajectory = sol.y.T
        energies = [compute_energy_qor(s, masses) for s in trajectory]
    
    # Compute diagnostics
    energy_drift = np.abs(energies[-1] - energies[0]) / np.abs(energies[0] + 1e-10)
    lyap = lyapunov_exponent(trajectory, dt)
    
    # Check for escape (any body too far)
    final_pos = trajectory[-1][:6].reshape(3, 2)
    max_dist = np.max([np.linalg.norm(final_pos[i]) for i in range(3)])
    escaped = max_dist > 100
    
    return {
        'trajectory': trajectory,
        'times': sol.t,
        'energies': energies,
        'energy_drift': energy_drift,
        'lyapunov': lyap,
        'escaped': escaped,
        'success': sol.success
    }


def figure_8_initial_conditions():
    """
    Figure-8 orbit initial conditions (Chenciner & Montgomery 2000).
    Three equal masses on a figure-8 trajectory.
    """
    # Positions
    x1, y1 = -0.97000436, 0.24308753
    x2, y2 = 0.0, 0.0
    x3, y3 = 0.97000436, -0.24308753
    
    # Velocities
    vx1, vy1 = 0.4662036850, 0.4323657300
    vx2, vy2 = -0.93240737, -0.86473146
    vx3, vy3 = 0.4662036850, 0.4323657300
    
    return np.array([x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3])


def lagrange_triangle_initial_conditions():
    """Equilateral triangle (Lagrange) initial conditions."""
    # Equilateral triangle with side = 1
    r = 1.0
    pos = np.array([
        [r * np.cos(0), r * np.sin(0)],
        [r * np.cos(2*np.pi/3), r * np.sin(2*np.pi/3)],
        [r * np.cos(4*np.pi/3), r * np.sin(4*np.pi/3)]
    ])
    
    # Circular velocities for rotation
    omega = np.sqrt(G * M / r**3)
    vel = np.array([
        [-omega * r * np.sin(0), omega * r * np.cos(0)],
        [-omega * r * np.sin(2*np.pi/3), omega * r * np.cos(2*np.pi/3)],
        [-omega * r * np.sin(4*np.pi/3), omega * r * np.cos(4*np.pi/3)]
    ])
    
    return np.concatenate([pos.flatten(), vel.flatten()])


def euler_collinear_initial_conditions():
    """Collinear (Euler) configuration."""
    # Bodies on a line with specific spacing
    pos = np.array([
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0]
    ])
    
    # Small perpendicular velocities to avoid trivial collision
    vel = np.array([
        [0.0, 0.3],
        [0.0, -0.6],
        [0.0, 0.3]
    ])
    
    return np.concatenate([pos.flatten(), vel.flatten()])


def hierarchical_triple_initial_conditions():
    """Hierarchical triple: tight binary + distant third."""
    # Tight binary
    a_bin = 0.2
    pos1 = np.array([-a_bin/2, 0])
    pos2 = np.array([a_bin/2, 0])
    
    # Binary orbital velocity
    v_bin = np.sqrt(G * M/3 / a_bin) * 0.5
    vel1 = np.array([0, -v_bin])
    vel2 = np.array([0, v_bin])
    
    # Distant third
    pos3 = np.array([3.0, 0])
    v_outer = np.sqrt(G * 2*M/3 / 3.0) * 0.3
    vel3 = np.array([0, v_outer])
    
    pos = np.array([pos1, pos2, pos3])
    vel = np.array([vel1, vel2, vel3])
    
    return np.concatenate([pos.flatten(), vel.flatten()])


def random_initial_conditions(seed=None):
    """Random initial conditions for chaos comparison."""
    if seed is not None:
        np.random.seed(seed)
    
    # Random positions in unit box
    pos = np.random.uniform(-1, 1, (3, 2))
    
    # Random velocities
    vel = np.random.uniform(-0.5, 0.5, (3, 2))
    
    # Center of mass correction
    pos -= np.mean(pos, axis=0)
    vel -= np.mean(vel, axis=0)
    
    return np.concatenate([pos.flatten(), vel.flatten()])


def main():
    print("="*70)
    print(" QO+R THREE-BODY PROBLEM TEST")
    print(" Testing phase-coherence prediction: λ_QOR ≤ λ_Newton")
    print("="*70)
    
    # Equal masses
    masses = np.array([M/3, M/3, M/3])
    
    # Test configurations
    configurations = {
        'Figure-8': figure_8_initial_conditions(),
        'Lagrange Triangle': lagrange_triangle_initial_conditions(),
        'Euler Collinear': euler_collinear_initial_conditions(),
        'Hierarchical': hierarchical_triple_initial_conditions(),
    }
    
    # Add random configurations
    for i in range(3):
        configurations[f'Random_{i+1}'] = random_initial_conditions(seed=42+i)
    
    results = {}
    
    print("\n" + "-"*70)
    print(f"  {'Configuration':<20} {'Model':<10} {'λ (Lyap.)':<12} {'ΔE/E':<12} {'Escaped':<10}")
    print("-"*70)
    
    for name, ic in configurations.items():
        # Run Newton
        res_newton = run_simulation(ic, masses, t_max=50, dt=0.01, model='newton')
        
        # Run QO+R
        res_qor = run_simulation(ic, masses, t_max=50, dt=0.01, model='qor')
        
        results[name] = {
            'newton': {
                'lyapunov': float(res_newton['lyapunov']) if np.isfinite(res_newton['lyapunov']) else None,
                'energy_drift': float(res_newton['energy_drift']),
                'escaped': res_newton['escaped']
            },
            'qor': {
                'lyapunov': float(res_qor['lyapunov']) if np.isfinite(res_qor['lyapunov']) else None,
                'energy_drift': float(res_qor['energy_drift']),
                'escaped': res_qor['escaped']
            }
        }
        
        # Print results
        lyap_n = f"{res_newton['lyapunov']:.4f}" if np.isfinite(res_newton['lyapunov']) else "N/A"
        lyap_q = f"{res_qor['lyapunov']:.4f}" if np.isfinite(res_qor['lyapunov']) else "N/A"
        
        print(f"  {name:<20} {'Newton':<10} {lyap_n:<12} {res_newton['energy_drift']:<12.2e} {str(res_newton['escaped']):<10}")
        print(f"  {'':<20} {'QO+R':<10} {lyap_q:<12} {res_qor['energy_drift']:<12.2e} {str(res_qor['escaped']):<10}")
        print()
    
    # Analyze results
    print("="*70)
    print(" ANALYSIS")
    print("="*70)
    
    n_chaos_reduced = 0
    n_total = 0
    
    for name, res in results.items():
        lyap_n = res['newton']['lyapunov']
        lyap_q = res['qor']['lyapunov']
        
        if lyap_n is not None and lyap_q is not None:
            n_total += 1
            if lyap_q <= lyap_n:
                n_chaos_reduced += 1
                print(f"  {name}: λ_QOR ({lyap_q:.4f}) ≤ λ_Newton ({lyap_n:.4f}) ✓")
            else:
                print(f"  {name}: λ_QOR ({lyap_q:.4f}) > λ_Newton ({lyap_n:.4f}) ✗")
    
    print()
    print(f"  Chaos reduction rate: {n_chaos_reduced}/{n_total} configurations")
    
    if n_total > 0:
        rate = n_chaos_reduced / n_total
        if rate >= 0.7:
            status = "CONFIRMED"
            print(f"\n  PREDICTION STATUS: {status} (≥70% show chaos reduction)")
        elif rate >= 0.5:
            status = "CONSISTENT"
            print(f"\n  PREDICTION STATUS: {status} (50-70% show chaos reduction)")
        else:
            status = "FALSIFIED"
            print(f"\n  PREDICTION STATUS: {status} (<50% show chaos reduction)")
    else:
        status = "INCONCLUSIVE"
        print(f"\n  PREDICTION STATUS: {status} (insufficient data)")
    
    results['summary'] = {
        'n_configurations': n_total,
        'n_chaos_reduced': n_chaos_reduced,
        'reduction_rate': n_chaos_reduced / n_total if n_total > 0 else 0,
        'status': status
    }
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "qor_three_body_test.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
