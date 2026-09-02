# MILP formulation for the core-remapping policy

This note separates the model that produced the current segment-ablation result
from the additional objectives proposed for the final deployable scheduler.

## Part I. MILP before applying it to core remapping

### 0.1 What MILP is

MILP stands for **Mixed-Integer Linear Programming**.

- **Mixed**: the model may contain both continuous and integer variables;
- **Integer**: some decisions must be whole numbers or binary choices;
- **Linear**: every objective and constraint is a linear expression;
- **Programming**: in this context means mathematical optimization, not writing
  software source code.

Its canonical minimization form is

```math
\begin{aligned}
\min_x \quad & c^T x \\
\text{s.t.}\quad
& A x \le b,\\
& E x = f,\\
& \ell \le x \le u,\\
& x_i\in\mathbb{Z}, \qquad i\in\mathcal I.
\end{aligned}
```

The symbols have generic meanings:

| Symbol | Meaning |
|---|---|
| `x` | Decisions the solver must choose |
| `c^T x` | Quantity to minimize or maximize |
| `A x <= b`, `E x = f` | Rules every acceptable decision must satisfy |
| `l`, `u` | Lower and upper bounds on decisions |
| `I` | Indices of variables that must take integer values |

MILP is therefore not one specific equation. It is a modeling framework:
the researcher defines the decisions, objective, and valid boundaries; the
solver searches the resulting feasible combinations.

### 0.2 Decision variables, constraints, and objective

A MILP model has three essential parts.

#### Decision variables

These represent unknown choices. Examples include:

```math
x_i\in\{0,1\}
```

for a yes/no decision, or

```math
n_i\in\{0,1,\ldots,H\}
```

for how many times an option is used.

#### Constraints

Constraints define the feasible set:

```math
\mathcal F=\{x\mid Ax\le b,\ Ex=f,\ x_i\in\mathbb Z\}.
```

They express statements such as:

- choose exactly one option;
- do not exceed a resource capacity;
- preserve a required count;
- use only implementation-supported combinations.

The solver is not allowed to return a point outside `F`.

#### Objective

The objective ranks all feasible solutions:

```math
x^*=\arg\min_{x\in\mathcal F} c^T x.
```

Changing the objective changes which feasible solution is preferred. Changing
the constraints changes which solutions are possible at all.

### 0.3 A small abstract example

Suppose three jobs must each be assigned to one of two machines. Define

```math
x_{jm}\in\{0,1\}
```

to be `1` if job `j` is assigned to machine `m`. Every job must be assigned
exactly once:

```math
\sum_m x_{jm}=1,
\qquad \forall j.
```

If job `j` contributes load `a_{jm}` on machine `m`, machine load is

```math
L_m=\sum_j a_{jm}x_{jm}.
```

Introducing `z` with

```math
L_m\le z,
\qquad \forall m
```

allows the model to minimize the busiest machine:

```math
\min z.
```

Our core-remapping problem uses the same mathematical pattern, but replaces
jobs with placement domains and machines with physical workers.

### 0.4 Why integer variables make the problem difficult

If integrality is temporarily removed, the model becomes a Linear Program
(LP). Its feasible region is convex, and modern solvers can optimize it very
efficiently.

An LP may return a fractional answer such as

```math
x_1=0.37,
\qquad x_2=0.63.
```

That is invalid when the decision means selecting a placement or executing it
an integer number of times. Requiring integer values makes the feasible set
discrete and introduces combinatorial search.

### 0.5 How a MILP solver finds and certifies a solution

Solvers such as Gurobi and HiGHS commonly use **branch-and-bound**, strengthened
by presolve, cutting planes, heuristics, and other techniques.

For a minimization problem:

1. **LP relaxation**: temporarily ignore integrality and solve the easier LP.
   Its objective is a lower bound on any valid integer solution.
2. **Incumbent search**: find a feasible integer solution. Its objective is the
   current upper bound.
3. **Branch**: if an integer variable is fractional, create subproblems that
   restrict it to alternative integer ranges.
4. **Bound and prune**: discard a subproblem if it is infeasible or its lower
   bound is already no better than the incumbent.
5. **Repeat** until the bounds meet or the time limit is reached.

The relative optimality gap is

```math
\text{gap}
=\frac{\text{incumbent}-\text{lower bound}}
       {|\text{incumbent}|}.
```

- `gap = 0`: the incumbent is mathematically proven optimal;
- small nonzero gap: the solution is feasible and its maximum possible
  improvement is quantitatively bounded;
- no incumbent: the solver has not yet found a feasible integer solution.

This distinction is why the current two-segment result is described as a
near-optimal solution with `0.000989%` certified gap, rather than simply called
optimal.

### 0.6 What the solver does not do

The solver does not discover architectural rules by itself. It does not know
that a placement deadlocks, violates shard order, fails PCC, or introduces
resharding unless those possibilities are removed or represented by explicit
constraints and measured costs.

The workflow is therefore

```text
architecture and measurements
        -> variables, legal candidates, costs, and constraints
        -> MILP solver
        -> feasible schedule plus an optimality certificate
```

### 0.7 Why MILP fits this scheduler

The core-remapping scheduler naturally contains:

- discrete placement choices;
- integer use counts over a finite horizon;
- linear accumulation of per-worker compute exposure;
- linear assignment and capacity constraints;
- a min-max objective that becomes linear through an auxiliary variable;
- a small offline problem where runtime solution latency is not critical.

MILP is appropriate because it can jointly optimize these coupled choices and
report a bound on how far the returned schedule may be from the best possible
schedule. A round-robin or greedy heuristic can produce a schedule but cannot
normally provide that certificate.

## Part II. Applying MILP to core remapping

## 1. One-sentence formulation

Over a periodic horizon of `H` inferences, choose how many times each legal
placement is used so that the most heavily exposed worker receives as few
TRISC-KERNEL cycles as possible.

## 2. Sets and indices

| Symbol | Meaning | Current value |
|---|---|---:|
| `c in C` | Physical compute worker | `|C| = 20` on Eagle-N `5 x 4` |
| `d in D` | Placement domain | `|D| = 17` |
| `p in P_d` | Legal placement candidate for domain `d` | Generated offline |
| `H` | Number of inferences in one repeated schedule period | `20` |

A **placement domain** is the largest consecutive operator chain that can
retain one placement without an `ExitToStock`. The optimizer selects one
placement for the whole domain, not independently for every Conv.

## 3. Measured and predicted parameters

Let

```math
A_{dpc}
```

be the compute exposure assigned to physical worker `c` when domain `d` uses
placement `p` once. The current unit is TRISC-KERNEL engine cycles.

Let

```math
B_c
```

be the per-inference compute exposure on worker `c` from operations that are
not changed by the scheduler.

The current candidate model constructs `A_dpc` by preserving the measured
logical-core cost pattern and mapping that pattern to the physical workers of
candidate `p`. It is therefore a compute-only prediction until candidate
placements are profiled on the board.

## 4. Legal candidate set and segment limit

Let `s_x(p)` and `s_y(p)` be the numbers of contiguous active coordinate runs
along the worker `x` and `y` axes. The three ablation candidate sets use the
actual implemented axis constraints:

```math
\begin{aligned}
P_d^{(1)} &= \{p\in P_d \mid s_x(p)=1,\ s_y(p)=1\},\\
P_d^{(2)} &= \{p\in P_d \mid s_x(p)+s_y(p)-2\le 1\},\\
P_d^{(4)} &= \{p\in P_d \mid s_x(p)\le2,\ s_y(p)\le2\}.
\end{aligned}
```

For a Cartesian BLOCK_SHARDED placement, `s_x(p)s_y(p)` is the number of
rectangular regions. For a partial HEIGHT_SHARDED prefix it is best treated as
the structural upper bound induced by the two physical axes, while ordered
worker coordinates remain the authoritative representation.

Because each axis currently has at most two runs:

- `K=1`: one contiguous placement region;
- `K=2`: one axis may be divided into two runs;
- `K=4`: both axes may be divided, producing four Cartesian regions.

The selected policy uses `K=2`. This is a hard candidate-set constraint, not a
weighted segment penalty.

Every retained candidate must already satisfy the Conv implementation rules:

- the placement fits the physical compute grid;
- core count and logical shard order are preserved;
- the placement propagates through every member of its domain;
- its declared exit boundary is supported;
- WIDTH_SHARDED and COL_MAJOR / `transpose_mcast` remain outside the enabled
  scope.

These properties are checked during candidate generation. They do not need to
be rediscovered by the MILP.

## 5. Decision variable

```math
n_{dp} \in \{0,1,\ldots,H\}
```

is the number of phases in one `H`-inference period for which domain `d` uses
placement `p`.

Each domain executes exactly once per inference, so

```math
\sum_{p \in P_d^{(K)}} n_{dp} = H,
\qquad \forall d \in D.
```

This count-based formulation avoids creating a separate binary placement
variable for every `(phase, domain, candidate)` tuple.

## 6. Per-worker cumulative load

The cumulative compute exposure of worker `c` over one complete period is

```math
L_c = H B_c
    + \sum_{d \in D}\sum_{p \in P_d^{(K)}} n_{dp} A_{dpc}.
```

The total work is invariant because every candidate only remaps the same
logical work. Therefore the period-average worker load is

```math
\bar{L} = \frac{1}{|C|}\sum_{c \in C} L_c,
```

and does not depend on the selected placements.

## 7. Primary objective: minimize the hottest worker

Introduce a continuous epigraph variable `z` and impose

```math
L_c \le z,
\qquad \forall c \in C.
```

The primary MILP is

```math
\begin{aligned}
\min_{n,z} \quad & z \\
\text{s.t.}\quad
& \sum_{p \in P_d^{(K)}} n_{dp}=H,
&& \forall d \in D, \\
& H B_c + \sum_{d,p} n_{dp}A_{dpc} \le z,
&& \forall c \in C, \\
& n_{dp}\in\{0,\ldots,H\}.
\end{aligned}
```

This objective directly prevents one hot worker from being hidden by an
unchanged average.

## 8. Theoretical upper bound on reduction

No scheduler can reduce the maximum worker below the conserved mean load. For
stock maximum load `L_stock^max`, the ideal reduction ceiling is

```math
R_{ideal}
= 1 - \frac{\bar{L}}{L_{stock}^{max}}.
```

For the current full-model profile,

```math
L_{stock}^{max}=3,777,635,
\qquad
\bar{L}=2,605,492,
```

so

```math
R_{ideal}=31.0285\%.
```

## 9. Current segment-ablation result

The same primary MILP was solved independently for the three candidate sets.

| Maximum placement regions | Peak cycles/inference | Reduction vs. stock |
|---:|---:|---:|
| 1 | 3,275,181.0 | 13.3008% |
| 2 | 2,777,211.9 | 26.4828% |
| 4 | 2,776,839.8 | 26.4926% |
| Ideal equal load | 2,605,492.0 | 31.0285% |

Allowing two regions captures

```math
\frac{26.4828}{31.0285}=85.35\%
```

of the mathematically available peak reduction. Allowing four regions adds
only `0.00985` percentage points. The two-region solve ended with a certified
MILP gap of `0.000989%`.

## 10. Secondary objective: reduce residual imbalance

Min-max optimization may leave multiple schedules with the same hottest-worker
load. Introduce

```math
e_c \ge 0
```

with

```math
e_c \ge L_c-\bar{L},
\qquad
e_c \ge \bar{L}-L_c.
```

After obtaining the best peak `z*`, solve

```math
\min \sum_{c \in C} e_c
```

subject to the original constraints and

```math
z \le (1+\epsilon_z)z^*.
```

`epsilon_z` should be a declared engineering tolerance. It avoids sacrificing
large implementation simplicity to preserve an insignificant last few cycles
of the Stage-1 incumbent.

This is an L1 deviation objective. It remains linear and is less sensitive to
single outliers than squared deviation.

## 11. Implementation-complexity objective

Let

```math
u_{dp}\in\{0,1\}
```

indicate whether candidate placement `p` is used at least once for domain `d`.
The exact indicator relation is linearized by

```math
n_{dp} \le H u_{dp},
\qquad
u_{dp} \le n_{dp}.
```

Under the accepted peak and deviation tolerances, minimize

```math
\min \sum_{d\in D}\sum_{p\in P_d^{(2)}} u_{dp}.
```

This minimizes the number of distinct domain-placement programs that must be
compiled, cached, validated, and selected at runtime.

## 12. Why phase switches are not another MILP objective

The MILP first determines only the counts `n_dp`. For each domain, all
occurrences of one selected placement can then be placed in one contiguous
cyclic block.

If a domain uses `m_d` distinct placements, its minimum cyclic switch count is

```math
T_d^{min}=
\begin{cases}
0, & m_d\le 1,\\
m_d, & m_d>1.
\end{cases}
```

Therefore phase ordering is a deterministic post-processing step. Solving it
again as a large MILP adds symmetry without changing the minimum.

## 13. Final optimization hierarchy

The proposed scheduler uses lexicographic optimization:

```text
Hard constraints
  legal candidates, at most two placement regions, PCC pass,
  and eventually measured latency overhead <= epsilon_latency

Stage 1
  minimize maximum worker compute exposure

Stage 2
  minimize total absolute worker-load deviation
  within a declared Stage-1 tolerance

Stage 3
  minimize the number of used domain-placement programs
  within the accepted compute tolerances

Post-processing
  arrange placement counts into minimum-switch cyclic blocks
```

Segment count is no longer a soft objective: the ablation justifies fixing the
candidate set to at most two placement regions.

## 14. Latency and correctness constraints

PCC is a candidate validation filter, not an optimization term. A candidate
that fails correctness is removed from `P_d`.

After board measurements are available, latency should also be a hard budget,
for example

```math
\frac{T_{schedule}-H T_{stock}}{H T_{stock}}
\le \epsilon_{latency}.
```

`T_schedule` must include candidate execution time, entry resharding,
`ExitToStock`, and program-cache effects. The current compute-only MILP does
not claim to model those costs.

## 15. What the model does and does not claim

The current objective balances measured TRISC-KERNEL compute exposure. It does
not yet model:

- SRAM aging;
- BRISC/NCRISC activity;
- NIU/NoC stress;
- temperature, voltage, or energy;
- a transistor-level aging transfer function;
- measured placement-dependent latency.

The defensible claim is therefore **compute-exposure balancing**, used as the
currently observable component of a broader aging problem.

## 16. Suggested presentation sequence

### Slide 1 — MILP selects the best feasible combination of discrete decisions

- Introduce variables, constraints, feasible set, and objective.
- Show the canonical form `min c^T x` subject to linear boundaries.

### Slide 2 — Branch-and-bound provides both a schedule and an optimality bound

- LP relaxation gives a lower bound.
- A feasible integer schedule gives an incumbent upper bound.
- Their difference gives the reported MILP gap.

### Slide 3 — Stock execution creates persistent per-worker compute imbalance

- Show the stock per-worker TRISC-KERNEL profile.
- State the metric boundary: compute exposure, not total-chip aging.

### Slide 4 — Full-model profiling converts every placement into a load vector

- Input: one uninterrupted BOS ResNet50 full-model profile.
- Show `A_dpc` as a 20-element cost vector.
- Explain placement domains and why a Conv chain is scheduled together.

### Slide 5 — The scheduler chooses placement-use counts over 20 inferences

Show only:

```math
n_{dp}\in\{0,\ldots,H\},
\qquad
\sum_p n_{dp}=H.
```

### Slide 6 — Min-max optimization directly lowers the hottest worker

Show the cumulative-load equation and the Stage-1 MILP.

### Slide 7 — Load conservation gives a 31.03% ideal ceiling

Show

```math
R_{ideal}=1-\bar{L}/L_{stock}^{max}.
```

Use the measured stock maximum and mean.

### Slide 8 — Two placement segments capture nearly all useful flexibility

Use `segment_scope_ablation.pdf`.

- 1 segment: `13.30%` reduction;
- up to 2 segments: `26.48%`;
- up to 4 segments: `26.49%`.

### Slide 9 — Secondary objectives choose a deployable schedule

Show the L1 deviation linearization and placement-use indicator, not the old
five-stage formulation.

### Slide 10 — Runtime execution is a deterministic periodic lookup

```math
phase = inference\_index \bmod H.
```

The MILP runs offline; Conv kernels do not solve optimization problems.

### Slide 11 — Board validation closes the remaining modeling gap

- candidate PCC;
- measured end-to-end latency;
- entry/exit reshard overhead;
- measured vs. predicted per-core exposure over one complete period.
