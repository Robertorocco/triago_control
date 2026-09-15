# A Safety-Aware Shared Autonomy Framework with BarrierIK Using Control Barrier Functions

**Authors:** Berk Guler\*, Kay Pompetzki\*, Yuanzheng Sun, Simon Manschitz, Jan Peters (\*equal contribution)
**Affiliations:** TU Darmstadt; DFKI; hessian.AI; Robotics Institute Germany (RIG); Centre for Cognitive Science; Honda Research Institute Europe GmbH
**Venue:** Accepted, Proc. IEEE International Conference on Robotics and Automation (ICRA), 2026
**Preprint:** arXiv:2603.01705v1 [cs.RO], 2 March 2026
**Project page:** BarrierIK (additional materials)
**BibTeX key used in this thesis:** `guler2026barrierik`

> Markdown transcription of the source PDF, kept as reference material for the state-of-the-art chapter. This is the single closest work in the literature to the present thesis: same mathematical instrument (control barrier functions), same architectural question (where safety is enforced relative to shared-autonomy blending).

---

## Abstract

Shared autonomy blends operator intent with autonomous assistance. In cluttered environments, linear blending can produce unsafe commands even when each source is individually collision-free. Many existing approaches model obstacle avoidance through potentials or cost terms, which only enforce safety as a soft constraint; safety-critical control instead requires hard guarantees. The authors investigate control barrier functions (CBFs) at the inverse kinematics (IK) layer of shared autonomy, targeting post-blend safety while preserving task performance. The approach is evaluated in simulation on cluttered environments and in a VR teleoperation study comparing pure teleoperation with shared autonomy. Across conditions, CBFs at the IK layer reduce violation time and increase minimum clearance while maintaining task performance. In the user study, participants report higher perceived safety and trust, lower interference, and an overall preference for shared autonomy with the safety filter.

## Core claim and pipeline

The pipeline preserves the standard shared-autonomy ordering and adds a safety stage after it:

1. The operator supplies a target pose `T_h(t) ∈ SE(3)`.
2. A guiding policy supplies an autonomous target pose `T_r(t)` from state `s(t) = {θ(t), O(t), L(t), task context}`, where `θ` is the joint configuration and `O`, `L` are obstacle and robot-link capsule sets.
3. The two poses are **linearly blended in SE(3)** with arbitration weight `α(t)` into an *unfiltered* blended target `T(t)` — which may violate safety margins.
4. **BarrierIK** receives `T(t)` and enforces CBF conditions to compute a safe joint command `θ*(t)`, which is applied to the robot.
5. The updated state is fed back to the policy and the user.

Central hypothesis: *projecting blended commands into the safety set, rather than treating safety as a soft objective, preserves task performance while increasing constraint satisfaction and maintaining user satisfaction.*

## Stated contributions

1. A shared-autonomy architecture that blends task-space commands and then addresses **post-blend** safety at the IK layer, handling failure modes of command-space blending in cluttered, nonconvex scenes.
2. **BarrierIK**, a CBF-constrained IK formulation treating safety as a hard inequality constraint while remaining compatible with standard IK objectives (pose tracking, joint limits, smoothness).
3. A comparative evaluation in simulation (dynamic obstacles; frame/shelf) and in a VR teleoperation study with ten participants, against IK baselines.

## Related-work positioning (as argued by the authors)

- **Shared autonomy.** Two families: linear policy blending (Dragan & Srinivasa; Muelling et al.; Gopinath et al.; Jeon et al.; Gottardi et al.; Song et al.; Atan et al.) and policy/POMDP methods (Javdani et al.; Nikolaidis et al.; Allenspach et al.). Arbitration typically derives from goal-probability confidence via maximum entropy or recursive Bayesian inference.
- **Failure mode of linear blending.** Even when human and autonomy commands are individually safe, their combination may fall outside a **nonconvex** safe set, producing unsafe or misaligned actions (Trautman). Attractive/repulsive field extensions add safety but keep it soft.
- **Collision avoidance at the IK layer.** Null-space/operational-space projection of an obstacle-avoidance gradient (Maciejewski & Klein; Khatib) requires redundancy and imposes a strict task hierarchy. Optimization-based IK: CollisionIK adds environment-distance penalties (safety remains *tradeable*); hard-constraint QP-IK variants (IKinQP) enforce minimum-distance inequalities but need reliable signed distances and careful aggregation; DawnIK is myopic with no forward-invariance guarantee.
- **CBFs.** Sufficient conditions for forward invariance of a safe set, enforced via minimal-deviation QPs (Ames et al.). Applied to teleoperation (Xu & Sreenath; Zhang et al.; Qin et al.) but rarely to shared autonomy (He et al., barrier pairs, low-dimensional navigation). Operational-space CBFs now scale to hundreds/thousands of constraints (Morton & Pavone).

## Method

### Blending

Position blends linearly and orientation by SLERP,

```
x(t) = (1 − α_t) x_h(t) + α_t x_r(t)
q(t) = slerp(q_h(t), q_r(t), α_t)
```

with a positive-dot-product antipodal fix before interpolation. Arbitration follows Muelling et al. — a logistic function of the *position-space disagreement*:

```
α_t = σ( p · ( ||x_h(t) − x_r(t)||₂ / s + b ) )
```

with slope `p`, scale `s`, bias `b`. Autonomy is reduced when human and robot intent diverge and increased when they align.

### IK backbone (Baseline N)

Nonlinear optimization-based IK, re-implemented in JAX for differentiable forward kinematics and collision checking, building on RelaxedIK:

```
θ* = argmin_θ  J(θ)
s.t. c_k(θ) ≤ 0,  k = 1..K;   l_i ≤ θ_i ≤ u_i  ∀i
```

with `J(θ) = Σ_m w_m g_m(F_m(θ))` a weighted sum of differentiable task features: end-effector position/orientation tracking, motion-smoothness regularizers on joint velocity/acceleration/jerk and Cartesian velocity, and a self-collision term using an ANN predictor of link-pair distances. A manipulability constraint `c_m(θ)` keeps the Jacobian well-conditioned (smallest singular value floored, condition number capped). Solved with **SLSQP**; because SLSQP linearizes locally, feasibility holds only to numerical tolerance — approximate, not continuous-time, safety.

**Baseline N** = the above, with no obstacle model at all.

### BarrierIK (Baseline B)

For each obstacle `o ∈ O`, the barrier is `h_o(θ) = φ_o(θ) − ε`, where `φ_o` is the minimum signed robot–obstacle distance and `ε` the safety margin. The active (closest) link–obstacle pair realizing `φ_o` is identified and differentiated automatically. A **discrete-time CBF condition** is imposed:

```
∇h_o(θ)ᵀ Δθ + K(h_o(θ)) ≥ 0,     K(h) = γh + βh³,  γ, β > 0
```

Note this uses a **position-based update rule** `Δθ` rather than a velocity-controlled formulation — an explicit design concession to their position-based control loop.

All CBF constraints are aggregated with a **temperature-scaled soft-max (log-sum-exp)** to avoid discontinuous constraint switching and numerical under/overflow:

```
c_CBF(θ) = (1/T) log Σ_{o∈O} exp( T [ −ḣ_o(θ) − K(h_o(θ)) ] ),    ḣ_o(θ) = ∇h_o(θ)ᵀ Δθ
```

Higher `T` approaches a true max. The BarrierIK program is then

```
θ* = argmin_θ  J(θ)
s.t. c_m(θ) ≤ 0,  c_CBF(θ) ≤ 0,  l_i ≤ θ_i ≤ u_i  ∀i
```

### Soft-constraint comparator (Baseline P)

Same objective, obstacle avoidance moved *into* the cost following CollisionIK. Per link `ℓ` and obstacle `o`, with signed distance `φ_ℓo(θ)`,

```
χ_ℓo(θ) = w_safe (φ²_ℓo(θ) + δ)⁻¹ ,   w_safe = (5ε)²
J_P(θ) = J_N(θ) + w_col · g_col( Σ_ℓ Σ_o χ_ℓo(θ) )
```

Smooth and differentiable, but with no forward-invariance or strict-satisfaction guarantee.

## Experiments

**Autonomous rollouts (40 trials, randomized seeds), two scenes:**
- *Dynamic Obstacles (DO)*: three cylinders oscillating on orthogonal axes at up to 0.025 m/s, end-effector mostly reorienting near centre — stresses responsiveness to moving constraints.
- *Frame/Shelf (FS)*: rectangular frame with four windows separated by vertical bars; the end-effector must pass each opening in sequence, and the reference trajectory deliberately lightly penetrates frame edges.

Metrics: number of collisions, minimum clearance `min_{ℓ,o} φ_ℓo(θ)`, violation-time percentage, position and orientation error, task jerk, joint jerk. Evaluation collision checking with HPP-FCL/Coal on convex capsules.

**VR teleoperation user study:** 7-DoF arm, HTC Vive controllers, Unity3D, pick three colour-coded cubes into a basket in clutter. Ten participants (6 male, 3 female, 1 non-binary; mean age 27.9 ± 3.14), six conditions — N, P, B, SA-N, SA-P, SA-B — five zero-shot trials each, randomized condition order, 3-minute familiarization, 3-minute and five-collision failure thresholds. Full pipeline at 90 Hz (VR-refresh-bound); in 100 Hz-capped tests B ran at 98.2 ± 7.91 Hz, P at 99.0 ± 3.20 Hz, N fixed at 100 Hz. CPU only (Intel Core i7-14700F).

Subjective instrument: modified NASA-TLX with all dimensions inverted (higher = better) — I-PD, I-TD, I-MD, I-PER, I-EFF, I-FL — extended with shared-autonomy dimensions control level (CL), assistance level (AL), safety level (SL), plus a final forced ranking of the six configurations.

### Table I — autonomous trajectory evaluation (mean ± s.d.)

Higher is better for minimum clearance (negative ⇒ penetration); lower is better elsewhere.

| Task | Solver | #Collisions | Min. clearance [m] | Violation time [%] | Pos. err. [m] | Ori. err. [°] | Task jerk [m/s³] | Joint jerk [rad/s³] |
|---|---|---|---|---|---|---|---|---|
| Shelf | N | 44.13 ± 9.74 | −0.162 ± 0.016 | 24.52 ± 1.19 | 0.020 ± 0.001 | 4.15 ± 0.07 | 4.197 ± 0.354 | 48.15 ± 2.89 |
| Shelf | P | 19.30 ± 5.85 | −0.171 ± 0.017 | 4.26 ± 5.22 | 0.049 ± 0.018 | 4.50 ± 0.12 | 6.518 ± 0.738 | 57.22 ± 3.35 |
| Shelf | **B** | 16.80 ± 4.31 | **−0.153 ± 0.020** | **0.53 ± 0.19** | 0.043 ± 0.009 | 4.76 ± 0.49 | **6.037 ± 0.496** | 65.89 ± 6.22 |
| Dynamic | N | 13.43 ± 3.30 | −0.091 ± 0.025 | 56.25 ± 16.62 | 0.002 ± 0.004 | 7.26 ± 0.13 | 2.835 ± 0.437 | 45.55 ± 3.51 |
| Dynamic | P | 1.55 ± 1.80 | −0.033 ± 0.022 | 4.04 ± 9.17 | 0.023 ± 0.015 | 7.35 ± 0.25 | 3.219 ± 0.911 | **33.62 ± 6.25** |
| Dynamic | **B** | 2.05 ± 3.74 | **−0.024 ± 0.022** | **0.84 ± 1.92** | **0.017 ± 0.010** | 8.13 ± 1.32 | 5.949 ± 1.927 | 54.47 ± 4.92 |

N = no collision avoidance; P = objective-based (soft) collision avoidance; B = BarrierIK (CBF-based). Bold marks statistically significant B vs. P differences; N is a reference only and is excluded from significance testing.

## Results and interpretation

- **Autonomous.** In both scenes B achieves markedly lower violation time and less-negative minimum clearance than P: contacts are *shorter and shallower*. Residual violation for B is attributed to discrete-time position updates and SLSQP's local linearization, producing transient numerical penetration between control steps. The controller does not linger at the boundary — it *recovers* rather than getting *stuck*.
- **Subjective.** Participants split into two archetypes: one favouring N (no assistance) and one favouring SA-B — opposite ends of the autonomy spectrum, both delivering favourable experiences. On the inverted TLX radar, N and SA-B cover the largest areas. SA-B combines high control level with high assistance and safety level, suggesting constraint-aware assistance reduces workload *without eroding agency*. Intermediate modes SA-N and SA-P show smaller, more dispersed footprints — when assistance deviates from user intent, operators report reduced control and lower perceived support. With n = 10, a Friedman test with Nemenyi post-hoc found **no significant pairwise differences (all p > 0.05)**; patterns are reported as indicative trends.
- **Objective teleoperation.** Success rates cluster around ~80% except SA-P (lower, higher variance). SA-P's apparently faster completion time reflects exclusion of failed runs, not efficiency. For collisions, unstructured assistance *increases* contacts in clutter: SA-N and SA-P exceed their non-SA counterparts, while SA-B reduces collisions among shared-autonomy modes; B (no SA) is best overall on raw collision count.

## Limitations acknowledged by the authors

- Hard safety constraints induce detours, appearing as **higher average joint jerk** for SA-B — a safety–smoothness trade-off undesirable for contact-sensitive tasks.
- The CBF implementation is **discrete** and ignores obstacle-velocity estimates, so the safe set is conservative when obstacles move.
- Barrier design choices (class-K function and shaping) materially influence intervention behaviour and were **not tuned per-user or per-scene**.
- CBFs are normally posed in velocity/torque space, but the teleoperation stack runs in **position space**; task-space velocity is approximated by a temporal difference between the current end-effector pose and the shared-autonomy target — myopic and step-wise.
- Simulation/VR only; no physical hardware.

## Future work stated

Deploy on physical hardware and measure contact dwell directly; incorporate velocity observers for dynamic CBF constraints; study class-K choices and adaptive shaping on user acceptance with a larger user study.

---

## Delta with respect to this thesis

Recorded here so the state-of-the-art chapter can position the contribution precisely rather than restate the paper.

| Axis | Guler et al. (BarrierIK) | This thesis |
|---|---|---|
| Safety layer | Nonlinear IK with CBF inequality, SLSQP, position-space `Δθ` update; safety only approximate between steps | Velocity-level convex QP in `q̇` with CLF tracking rows and CBF rows in their native first-order form; dense active-set solve at the control rate |
| Constraint aggregation | Log-sum-exp soft-max over obstacles, temperature `T` | Per-arm SoftMin over link–obstacle capsule pairs, same smoothing motive, applied per kinematic chain |
| Task term | Tracking as a weighted cost | Tracking as a **relaxed CLF constraint** with per-arm slack, so tracking degrades gracefully and measurably when it conflicts with safety |
| Feasibility under adversarial input | Not addressed; relies on solver tolerance | Explicit **reference governor** upstream of the CLF layer preserving QP feasibility for arbitrary operator commands |
| Constraint tuning | Fixed class-K coefficients, not adapted | **Adaptive scheduling from the QP's own KKT dual variables** (`λ_cbf` shadow prices) |
| Arbitration signal | Logistic function of position **disagreement** magnitude | Function of user–policy twist **alignment** (direction, not distance), over a discounted Bayesian belief on goal manifolds |
| Goals | Heuristic grasp poses on target objects | Dynamically resolved goal **manifolds**, with a geometric grasp state machine executing and returning authority |
| Operator interface | VR controllers, no force feedback | Haption Virtuose 6D impedance device; the CBF gradients and `λ_cbf` are **rendered back to the operator as forces** — the safety margin becomes palpable rather than invisible |
| Motion mapping | Single mapping (VR pose) | Two classical mappings compared: position control with clutch indexing vs. spring-centred rate control |
| Study design | 3 solvers × {SA, no SA} = 6 cells, n = 10, VR | 2×2×2 factorial {mapping} × {haptic guidance} × {reference blending}, on **real hardware** |
| World model | Given obstacle capsules | Camera-derived world built by an autonomous active-vision head, latched on a convergence criterion, feeding the barriers |

The shared conclusion the thesis can build on and cite directly: enforcing safety **after** arbitration is what makes shared autonomy acceptable to operators, and soft penalty terms are not a substitute for a hard constraint. The thesis extends this from an approximate position-space filter in simulation to a first-order QP with feasibility guarantees, dual-driven adaptation, and haptic disclosure of the constraint on real hardware.
