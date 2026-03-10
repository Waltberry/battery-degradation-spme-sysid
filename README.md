# Battery Degradation SPMe System Identification

Physics-informed system identification package for battery degradation monitoring using an SPMe-inspired proxy state model and staged nonlinear voltage surrogate estimation.

This repository contains the research codebase behind an MSc workflow for identifying battery internal dynamics and degradation indicators directly from terminal current and voltage data. The framework combines physically structured state evolution, staged parameter identification, nonlinear surrogate learning, cross-cycle monitoring, and synthetic validation.

The project is organized as a Python package under `src/battery_deg_spme`, with scripts for full experiment runs and notebooks for inspection and rapid visualization.

---

## Why this repository exists

The central problem is this:

Given measured current and voltage from a battery experiment,

- input: current \( I(t) \)
- output: terminal voltage \( V(t) \)

recover a compact internal model that is still interpretable enough to monitor degradation.

Instead of fitting a purely black-box time-series model, this repository uses:

- a structured SPMe-inspired state model for internal battery dynamics
- a learned additive nonlinear voltage surrogate
- staged system identification
- multi-cycle monitoring of resistance and dynamic parameters
- surface-based nonlinearity tracking across cycles

This makes the method suitable both for system identification and for degradation interpretation.

---

## Scientific goal

The objective is to estimate battery internal behavior and degradation-sensitive parameters from terminal data.

The internal state is represented by an SPMe-inspired proxy:

\[
x(t) =
\begin{bmatrix}
c_n \\
c_p \\
c_e
\end{bmatrix}
\in \mathbb{R}^{14}
\]

with states grouped as:

- negative-particle diffusion proxy states
- positive-particle diffusion proxy states
- electrolyte concentration proxy states

A key reduced view used throughout the voltage model is:

\[
x_{\text{surf}}(t) = \{x_p(t), x_n(t), c_{e,L}(t), c_{e,R}(t)\}
\]

where:

- \(x_p\) is the positive-electrode surface stoichiometry proxy
- \(x_n\) is the negative-electrode surface stoichiometry proxy
- \(c_{e,L}\) and \(c_{e,R}\) are left/right electrolyte concentration proxies

The terminal voltage model is:

\[
V(t) = \hat{Z}(x_p, x_n, c_{e,L}, c_{e,R}) - N_s I(t) R_0 - v_{rc}(t)
\]

where:

- \( \hat{Z}(\cdot) \) is the learned nonlinear surrogate
- \( N_s \) is the number of series cells
- \( R_0 \) is the lumped ohmic resistance
- \( v_{rc}(t) \) is an optional transient RC contribution

---

## Mathematical formulation

### 1. Proxy dynamic model

The internal proxy dynamics are modeled in continuous time as

\[
\dot{x}(t) = A(\theta_A)x(t) + B(\theta_B) I(t)
\]

with:

- \(x(t) \in \mathbb{R}^{14}\)
- \(A(\theta_A) \in \mathbb{R}^{14 \times 14}\)
- \(B(\theta_B) \in \mathbb{R}^{14 \times 1}\)

The matrices are not arbitrary dense matrices. They are structured to preserve SPMe-inspired transport topology.

The parameter vector for the dynamic matrix is:

\[
\theta_A =
\begin{bmatrix}
\theta_1 & \theta_2 & \theta_3 & \theta_4 & \theta_5 & \theta_6 & \theta_7
\end{bmatrix}^\top
\]

and the input matrix parameter vector is:

\[
\theta_B =
\begin{bmatrix}
\theta_8 & \theta_9 & \theta_{10} & \theta_{11}
\end{bmatrix}^\top
\]

The structure is:

- \( \theta_1, \theta_2 \) control solid diffusion blocks
- \( \theta_3,\dots,\theta_7 \) control electrolyte transport coupling
- \( \theta_8,\dots,\theta_{11} \) control current injection into solid/electrolyte states

This structure means parameter variation across cycles can still be interpreted physically, rather than treated as arbitrary neural weights.

---

### 2. Structured dynamic matrix \(A(\theta_A)\)

The global system matrix is block diagonal in the nominal proxy formulation:

\[
A(\theta_A) =
\begin{bmatrix}
A_n(\theta_1) & 0 & 0 \\
0 & A_p(\theta_2) & 0 \\
0 & 0 & A_e(\theta_3,\dots,\theta_7)
\end{bmatrix}
\]

The negative-electrode solid diffusion block is

\[
A_n(\theta_1)=
\begin{bmatrix}
-24\theta_1 & 24\theta_1 & 0 & 0 \\
16\theta_1 & -40\theta_1 & 24\theta_1 & 0 \\
0 & 16\theta_1 & -40\theta_1 & 24\theta_1 \\
0 & 0 & 16\theta_1 & -16\theta_1
\end{bmatrix}
\]

The positive-electrode block \(A_p(\theta_2)\) has the same structure.

The electrolyte block is parameterized as

\[
A_e(\theta_3,\dots,\theta_7)=
\begin{bmatrix}
-4\theta_3 & 4\theta_3 & 0 & 0 & 0 & 0 \\
4\theta_3 & -(4\theta_3+16\theta_4) & 16\theta_4 & 0 & 0 & 0 \\
0 & 16\theta_4 & -(16\theta_4+4\theta_5) & 4\theta_5 & 0 & 0 \\
0 & 0 & 4\theta_5 & -(4\theta_5+16\theta_6) & 16\theta_6 & 0 \\
0 & 0 & 0 & 16\theta_6 & -(16\theta_6+4\theta_7) & 4\theta_7 \\
0 & 0 & 0 & 0 & 4\theta_7 & -4\theta_7
\end{bmatrix}
\]

This is important for the thesis because it shows that identification is performed over a structured physical manifold, not over unrestricted matrices.

---

### 3. Structured input matrix \(B(\theta_B)\)

The current input enters only physically meaningful locations:

\[
B(\theta_B) =
\begin{bmatrix}
0 \\
0 \\
0 \\
6\theta_8 \\
0 \\
0 \\
0 \\
6\theta_9 \\
\theta_{10} \\
\theta_{10} \\
0 \\
0 \\
\theta_{11} \\
\theta_{11}
\end{bmatrix}
\]

This preserves the interpretation of current acting through solid and electrolyte transport channels.

---

### 4. Nonlinear voltage surrogate

The nonlinear voltage contribution is modeled as an additive polynomial surrogate:

\[
\hat{Z}(x_p, x_n, c_{e,L}, c_{e,R})
=
c_0
+ \sum_{k=1}^{d} a_k \tilde{x}_p^k
+ \sum_{k=1}^{d} b_k \tilde{x}_n^k
+ k_{\ln} \ln\left(\frac{c_{e,R}}{c_{e,L}}\right)
\]

where:

\[
\tilde{x}_p = \frac{x_p - x_{p,\text{ref}}}{x_{p,\text{scale}}}, \qquad
\tilde{x}_n = \frac{x_n - x_{n,\text{ref}}}{x_{n,\text{scale}}}
\]

The learned surrogate parameter vector is:

\[
\theta_Z =
\begin{bmatrix}
c_0 & a_1 & \cdots & a_d & b_1 & \cdots & b_d & k_{\ln}
\end{bmatrix}^\top
\]

or without the log term when electrolyte coupling is disabled.

This surrogate approximates a nonlinear voltage layer that, in a more mechanistic model, would arise from:

- open-circuit potential difference
- Butler–Volmer overpotential contribution
- electrolyte potential/logarithmic concentration correction

That is why the learned surface is tracked directly in the thesis: it is not just a fit device; it is a reduced nonlinear electrochemical signature.

---

### 5. Full output equation

The modeled terminal voltage is:

\[
\hat{V}(t) = \hat{Z}(x(t)) - N_s I(t)R_0 - v_{rc}(t)
\]

When RC dynamics are disabled:

\[
\hat{V}(t) = \hat{Z}(x(t)) - N_s I(t)R_0
\]

When RC dynamics are enabled, the transient is typically:

\[
\dot{v}_{rc}(t) = -\frac{1}{\tau} v_{rc}(t) + \frac{R_1}{\tau} I(t)
\]

and the output becomes:

\[
\hat{V}(t) = \hat{Z}(x(t)) - N_s I(t)R_0 - v_{rc}(t)
\]

---

## Identification algorithm

The identification is staged.

### Stage 2 — surrogate identification

Keep the proxy dynamics fixed to nominal matrices:

\[
A = A_{\text{nom}}, \qquad B = B_{\text{nom}}
\]

Estimate:

\[
\theta_Z,\; R_0 \quad (\text{and optional RC parameters})
\]

This stage answers:

Can the nonlinear voltage layer explain the measured voltage, assuming nominal internal dynamics?

This is the first crucial identifiability step. It isolates the voltage nonlinearity from the dynamic model.

---

### Stage 3a — dynamics identification

Freeze the learned surrogate from Stage 2 and estimate only:

\[
\theta_A,\; \theta_B
\]

That is:

\[
\dot{x}(t) = A(\theta_A)x(t) + B(\theta_B)I(t)
\]

while

\[
\hat{V}(t) = \hat{Z}_{\text{stage2}}(x(t)) - N_s I(t)R_0
\]

This stage tests whether the dynamic structure itself can be recovered once the nonlinear voltage layer is fixed.

---

### Stage 3b — full refinement

Unfreeze all parameters jointly:

\[
\theta_A,\; \theta_B,\; \theta_Z,\; R_0
\]

and optimize the full model end-to-end:

\[
\min_{\theta_A,\theta_B,\theta_Z,R_0}
\sum_{t=1}^{T}
\left(
V_{\text{meas}}(t) - \hat{V}(t)
\right)^2
+ \lambda \mathcal{R}(\theta)
\]

This is the final model used for reporting fit quality and degradation tracking.

Stage 3b is the final stage for all reported final-cycle results.

---

## Degradation indicators tracked across cycles

For each cycle, the package can monitor:

- Stage 2 fit quality
- Stage 3b fit quality
- \(R_0\) trend across cycles
- \(\theta_A\) trends across cycles
- \(\theta_B\) trends across cycles
- learned surrogate shape drift across cycles

The shape drift is computed directly on a common \((x_n, x_p)\) grid:

\[
\Delta Z^{(k)} = Z^{(k)} - Z^{(\text{ref})}
\]

with metrics such as:

\[
\text{RMSE}_{\text{drift}} = \sqrt{\frac{1}{N}\sum_i (\Delta Z_i)^2}
\]

This is more reliable than comparing raw polynomial coefficients alone because the surface is what actually influences voltage behavior.

---

## Repository architecture

```mermaid
flowchart TD
    A[scripts/] --> B[src/battery_deg_spme/config]
    A --> C[src/battery_deg_spme/io]
    A --> D[src/battery_deg_spme/preprocessing]
    A --> E[src/battery_deg_spme/models]
    A --> F[src/battery_deg_spme/fitting]
    A --> G[src/battery_deg_spme/evaluation]
    A --> H[src/battery_deg_spme/analysis]
    A --> I[src/battery_deg_spme/visualization]

    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    G --> H
    H --> I