# ADAPTS-HCT RL API — Hyperparameters

> **Status: implemented, 2026-10-16.** This document specifies the configuration
> interface of the learner `RL_ALGORITHM = "inf_lsvi_eb"` (the default), which
> implements `Overleaf_bridge/Algorithm_Design/main_new.tex` of 2026-10-16.
> The loader and validator are `app/hyperparams.py`; the learner that reads the
> values is `app/learner/`. §7 lists the legacy `config.py` keys they replace.

> Equation, algorithm, table, and section names refer to `main_new.tex` of
> 2026-10-16 (e.g. Algorithm 3 is the weekly update and warm-up loop;
> Algorithm 2 is the empirical-Bayes step; eq. (rho-params) sets the
> allocation constants).

Every number the learner uses is listed here exactly once, with its default,
its allowed range, and the place in the design document that defines it. The
defaults **are** the design: Table 1 of `main_new.tex`.
A deployment that changes no key runs the designed algorithm.

---

## 1. The four kinds of quantity

| Kind | Who sets it | When | Examples |
|---|---|---|---|
| **Fixed by design** | the study team, in the hyperparameter file | before the trial | discounts γ, REL reward weights, noise scales σ², trial length |
| **Tunable constants** | the study team, in the hyperparameter file | before the trial | run-in constant `m`, ν₀, λ₀, allocation shape, warm-up sizes |
| **Numerical** | the RL team, in the hyperparameter file | before the trial | iteration cap, tolerances, ridge ε |
| **Derived** | **the API**, never a user | at start-up, from the three kinds above | penalty weights λ_b, prior scales τ²₀,b, per-agent slopes, reward divisors, σ̄² |

The first three kinds are read from one file (§2). The fourth kind is computed
from them by a fixed formula (§5), reported, and included in the fingerprint.
**Nothing is estimated from trial data**: σ² is fixed by design
(`main_new.tex` §5.6), the state standardization uses fixed design ranges
(§5.4), and the Monte Carlo bank comes from a fixed seed.

---

## 2. How to specify hyperparameters

### 2.1 The file

Hyperparameters live in one YAML file, `ADAPTS-HCT-RL-API/hyperparameters.yaml`.
A different path can be given with the environment variable
`ADAPTS_HYPERPARAMS=/path/to/file.yaml`. The shipped file contains comments
only, so it runs the design as written.

- **Only overrides are needed.** A key that is absent takes its default from §3–§4.
  An empty file, or no file at all, runs the design as written.
- **Unknown keys are an error.** A misspelled key stops the app at start-up; it
  is never silently ignored.
- **Values are validated at start-up** against the "Allowed" column. A value
  out of range stops the app with a message naming the key.
- **Fractions** may be written as decimals or as strings such as `"13/14"`.

Tests and simulations may also pass overrides through the config key
`HYPERPARAMS_OVERRIDES` (nested or dotted keys, e.g. `{"warmup.n_warm": 2}`);
they are applied after the file and enter the fingerprint like any other value.

A complete file with every default spelled out is in §6. Copy it, then delete
the lines you do not change.

### 2.2 Commands

| Command | What it does |
|---|---|
| `flask hyperparams show` | prints the resolved set (defaults + file + derived values), one row per key, with its source (`default` / `file` / `override` / `derived`), the SHA-256 fingerprint, and whether it matches the frozen set |
| `flask hyperparams validate [FILE]` | checks a file without starting the app; names the offending key |
| `flask hyperparams diff [FILE]` | compares a file against the newest frozen set (§2.3); exits 1 if they differ |
| `flask hyperparams amend --reason "..." [--file FILE]` | records a protocol amendment after freezing (§2.3) |

### 2.3 Freezing and amendments

The set in force is **frozen at the first learning update**: the first weekly
`/update` of a week W with `|E_{W-1}| ≥ N_warm` (N_warm dyads have completed
week 1; §3.5) that has at least one usable decision to fit. At that moment
the API writes the resolved set, including the derived values of §5, to the
table `hyperparameter_sets` as version 1 with its SHA-256 fingerprint, in
force from that update's `timestamp`. In the same update it writes one
`noise_init` row per agent recording the design σ², σ̄², and τ²₀ values
(Monitoring-Spec C10).

The fingerprint is the SHA-256 of the canonical JSON of
`{"settings": ..., "derived": ...}`. It is recorded on every
`policy_versions` row, every learner row of `model_parameters`
(`local_fit`, `hyper`, `posterior`), and every `/action` decision.

- **Before freezing** (cohort warm-up), the file may be edited freely; the
  change takes effect at the next app restart.
- **After freezing**, the app refuses to start if the resolved set differs
  from the newest frozen version; it first records a red C9 alarm with the
  differing keys. A change during the trial is a protocol amendment: run
  `flask hyperparams amend --reason "..."`, which stores the new set as the
  next version with the reason and its in-force time. Monitoring check C9
  raises a red alarm if any fit or decision used settings other than the
  approved version in force at that time (`Monitoring-Spec.md`).
- The commands `flask hyperparams`, `flask db`, `flask export-csv` and
  `flask monitor` may run while the file differs from the frozen set (they
  inspect or repair it, or only read the database); they log a warning.
- **No value is ever re-estimated** during the trial: everything in §5 is a
  formula of the settings, so an amendment changes it only through the
  settings themselves.

---

## 3. Algorithm hyperparameters

Notation follows `main_new.tex`. *Per-agent* keys take a mapping
`{aya: …, cp: …, rel: …}`.

### 3.1 Fixed by design

| Key | Symbol | Meaning | Scope | Default | Allowed | Defined in |
|---|---|---|---|---|---|---|
| `discount` | γ | discount in the Bellman target; 1/(1−γ) is the agent's decisions per week; REL γ = 0 makes it a contextual bandit | per-agent | `{aya: 13/14, cp: 6/7, rel: 0}` | [0, 1) | eq. (gammas) |
| `trial_weeks` | T | decision weeks per dyad (the active set I_w is study weeks 1..T) | shared | `14` | integer ≥ 1 | eq. (cohort-index) |
| `rel_reward.respondent_weights` | (ϖ_AYA, ϖ_CP) | weights of the AYA and CP survey means in the relationship-quality part of the REL reward and in the REL state's dyad summary | REL | `[1/3, 2/3]` | positive, sum to 1 | eq. (reward-rel-quality) |
| `rel_reward.w_adh` | w_adh | weight on the week's mean AYA level in the REL reward; `0` gives the relationship-quality-only reward | REL | `0` | ≥ 0 | eq. (reward-rel) |
| `noise.sigma_sq` | σ² | noise scale of the local working model, on the **normalized** reward scale; the per-decision reward variance under π = 1/2 on the vanilla testbed | per-agent | `{aya: 0.158, cp: 0.122, rel: 0.078}` | > 0 and ≤ σ̄² (§5.1) | eq. (sigma-design) |
| `rewards.normalize` | — | learn on each reward divided by its range maximum r_max (so every reward is on [0, 1]) | shared | `true` | `true` / `false` | §5.3 |

σ² is fixed by design and is never estimated or updated from trial data
(`main_new.tex` §5.6). The REL value 0.078 holds at `w_adh = 0`; if `w_adh`
is changed, set `noise.sigma_sq.rel` explicitly as well (the loader logs a
warning when it is left at its default). With `rewards.normalize: false` the
σ² values must be restated on the native scale (0.63, 3.05, 1.95 at
`w_adh = 0`, `main_new.tex` §5.6), and the slope of §3.4 is no longer
multiplied by r_max.

Not configurable (structural):

- the baseline covariates `z_i ≡ 1` (p_z = 1);
- the feature maps h and f (Table 3, §5.5), hence p_h / p_f =
  7/4 (AYA), 6/4 (CP), 7/2 (REL). In order, the design state g is
  AYA `[1, engagement, burden, game, prior AYA level, mood, physical, relationship quality (AYA)]`,
  CP `[1, engagement, burden, game, CP missing rate, CP mood, relationship quality (CP)]`,
  REL `[1, relationship quality (dyad summary), prior-week mean AYA level, mean engagement, mean burden, prior game, mean diary summary]`;
  h is g without the agent's own app burden (AYA, CP), and f is
  `[1, engagement, burden, game]` (AYA, CP) or `[1, relationship quality]` (REL);
- the reward definitions (`main_new.tex` §5.3). The native reward ranges are
  **[0, 2]** for AYA (levels 2 / 1 / 0 for a dose reported taken / reported
  missed / no report), **[0, 5]** for CP (the CP diary mood if the diary was
  completed, else 0), and **[0, 5 + 2·w_adh]** for REL; so
  r_max = 2, 5, 5 + 2·w_adh;
- the standardization (`main_new.tex` §5.4, eq. (range-std)): every state
  coordinate x with design range [l, u] enters as
  (x − (u + l)/2) / ((u − l)/2), clipped to [−1, 1] (the clipped coordinates
  are recorded with the decision). The ranges are [1, 4] for app engagement
  and the REL mean engagement, [0, 2] for the prior AYA level and the
  prior-week mean AYA level, [0, 5] for diary items and relationship quality
  (level 0 is a missing report; there are no missingness indicators),
  [0, 1] for the CP missing rate, the game indicators and the diary summary,
  and [0, 1/(1 − γ)] for app burden (AYA [0, 14], CP [0, 7], REL mean
  [0, 10.5]). The burden ranges describe the host's counter, which uses the
  API-Spec §4.1 discounts 13/14 and 6/7, so they do not follow the
  `discount` key. Nothing is estimated from data.

`discount` is tied to the decision schedule. If the schedule changes (for
example messaging six days a week instead of seven), the discounts change with
it, and so do the derived bounds σ̄² of §5.1.

### 3.2 Local learner (Inf-LSVI)

| Key | Symbol | Meaning | Scope | Default | Allowed | Defined in |
|---|---|---|---|---|---|---|
| `penalty.m` | m | decisions of information asked per coefficient; the single constant behind all six penalty weights | shared | `2` | > 0 | eq. (runin-choice) |

The weights follow from `m` and the block sizes, λ_b = m · p_b with
p_main = p_h + 1 and p_mod = p_f − 1. The penalty is
Λ = diag(λ_main I_{p_h}, λ_main, λ_mod I_{p_f−1}) (eq. (localprior)). The
weights are reported by `flask hyperparams show` and are not separate keys:

| Agent | p_main | p_mod | λ_main | λ_mod | run-in, main (decisions) | run-in, mod (decisions) |
|---|---|---|---|---|---|---|
| AYA | 8 | 3 | 16 | 6 | 16 | 24 |
| CP | 7 | 3 | 14 | 6 | 14 | 24 |
| REL | 8 | 1 | 16 | 2 | 16 | 8 |

The run-in columns use c₀ = (1, 1/4) for (main, mod), as in Table 1; both are
upper bounds, so each run-in is a lower bound. c₀ is used only to report
run-ins; it does not enter the fit.

The weekly local fit takes `numerics.j_backups` Bellman backups (§4),
warm-started at the dyad's fit from the most recent earlier week; rows are the
dyad's decisions stamped before the update week's Monday 00:00 with a stored
design state and a derived reward.

### 3.3 Empirical-Bayes pooling

| Key | Symbol | Meaning | Scope | Default | Allowed | Defined in |
|---|---|---|---|---|---|---|
| `eb.nu0` | ν₀ | degrees of freedom of the inverse-χ² prior on each τ²_b; counts pseudo-dyads | shared | `4` | > 0 | eqs. (hyperprior-dist), (hyperconstants) |
| `eb.lambda0` | λ₀ | ridge weight on the hyper-mean Θ | shared | `0.1` | > 0 | eq. (hyperprior) |

The prior scales τ²₀,act = σ²/λ_main and τ²₀,mod = σ²/λ_mod are derived (§5.2).
The hyper-update minimizes the negative penalized log likelihood
(eq. (map)) with Θ profiled out in closed form, over (τ²_act, τ²_mod)
directly with scipy's L-BFGS-B: bounds [floor_b, ∞) with the floor of
§5.2, `maxiter = k_max`, `gtol = eps_tol`, `ftol = 0`, analytic gradient,
warm-started at last week's τ² clipped to this week's floor (τ²₀ at the first
update). It has converged when every value is finite and the projected
sup-norm gradient with respect to (Θ, τ²) is at most `eps_tol`. When
L-BFGS-B's line search stops on rounding before that, projected Newton steps
on the analytic gradient finish within the same `k_max` budget. Otherwise
the update falls back (F-U5) to last week's (Θ̂, Σ̂₀), or to the prior
(0, diag(τ²₀)) at the first update; posteriors are still recomputed from the
week's local fits.

### 3.4 Allocation

| Key | Symbol | Meaning | Scope | Default | Allowed | Defined in |
|---|---|---|---|---|---|---|
| `allocation.l_min` | L_min | lower clip of the randomization probability | shared | `0.2` | (0, l_max) | eq. (rho) |
| `allocation.l_max` | L_max | upper clip | shared | `0.8` | (l_min, 1) | eq. (rho) |
| `allocation.c_rho` | c_ρ | generalized-logistic offset; sets ρ(0) = L_min + (L_max − L_min)/(1 + c_ρ)^κ_ρ | shared | `3.5` (ρ(0) = 1/3) | > 0 | eq. (rho-params) |
| `allocation.b_rho` | b_ρ | slope **per unit of native-scale contrast** (the MiWaves value) | shared | `20` | > 0 | eq. (rho-params) |
| `allocation.kappa_rho` | κ_ρ | exponent | shared | `1` | > 0 | eq. (rho-params) |
| `allocation.mc_samples` | M | size of the Monte Carlo bank of N(0, 1) draws | shared | `500` | integer ≥ 100 | eq. (alloc-mc) |
| `allocation.mc_seed` | — | dedicated seed of the bank `default_rng(mc_seed).standard_normal(M)`; drawn once and stored with the sample buffer | shared | `12345` | integer in [0, 2⁶³) | eq. (alloc-mc) |

Because each agent learns on its normalized reward, the slope it applies to
the contrast μ = f′α^EB is **b_ρ · r_max**: 40 (AYA), 100 (CP) and
100 + 40·w_adh (REL) at the defaults (eq. (rho-params)); with
`rewards.normalize: false` it is b_ρ itself. With `c_rho = 1`, ρ(0) = 1/2
(`main_new.tex` §5.9 discusses the choice). The randomization probability is
π = (1/M) Σ_j ρ(μ + √v Z_j) with v = f′Σ^EB f, a deterministic function of
(μ, v) because the bank is fixed.

### 3.5 Warm-up

| Key | Symbol | Meaning | Scope | Default | Allowed | Defined in |
|---|---|---|---|---|---|---|
| `warmup.n_warm` | N_warm | cohort warm-up: every decision of calendar week W is Bernoulli(1/2), and no learning update runs, while fewer than N_warm registered dyads have **completed week 1** (`|E_{W−1}| < N_warm`, i.e. week-1 start on or before Monday(W) − 7 days) | shared | `5` | integer ≥ 2 | Algorithm 3 line 3, §5.2 |
| `warmup.t_warm` | T_warm | per-dyad warm-up, in weeks: each dyad's partial week (if it enrolled on a non-Monday) and its study weeks 1..T_warm are Bernoulli(1/2) | shared | `1` | integer ≥ 1 | Algorithm 3 line 12, §5.2 |

---

## 4. Numerical constants

| Key | Symbol | Meaning | Default | Allowed | Defined in |
|---|---|---|---|---|---|
| `numerics.j_backups` | J | Bellman backups per weekly local fit, warm-started at last week's fit (backups stop early once θ changes by at most 1e-6, which matters only for J > 1) | `1` | integer ≥ 1 | Algorithm 4 |
| `numerics.k_max` | K_max | iteration cap per hyper-update (L-BFGS-B iterations plus any Newton steps) | `200` | integer ≥ 1 | Algorithm 2 |
| `numerics.eps_tol` | ε_tol | projected-gradient tolerance of the hyper-update | `1e-4` | > 0 | Algorithm 2 |
| `numerics.eps_ridge` | ε | ridge εI added to every matrix inverse (local fit, hyper-update, posterior); also the tolerance of the C3 pooling check | `1e-6` | > 0 | Table 1; Monitoring-Spec C3 |

---

## 5. Derived values (computed by the API; not user-set)

These are formulas of the settings in §3–§4, computed at start-up. They are
shown by `flask hyperparams show` with source `derived`, written into the
frozen set (§2.3) and covered by the fingerprint. None of them depends on
trial data.

### 5.1 Noise bound σ̄² (per agent)

σ̄²_g = (r_max − r_min)² / (4 (1 − γ²)) on the scale the agent learns on
(eq. (sigma-plugin)): range 1 when rewards are normalized, r_max otherwise.

| Agent | σ̄² (normalized) | design σ² | σ² / σ̄² |
|---|---|---|---|
| AYA | 196/108 = 1.815 | 0.158 | 9% |
| CP | 49/52 = 0.942 | 0.122 | 13% |
| REL | 1/4 = 0.25 | 0.078 | 31% |

The bound is informational: σ² is fixed by design and is **not** clipped to
it. A configured `noise.sigma_sq` above σ̄² is rejected at start-up as an
out-of-range value naming the key, so `noise_init.clip_used` is always
`false`.

### 5.2 Hyper-prior scales τ²₀,b and floors (per agent, two blocks)

τ²₀,act = σ² / λ_main and τ²₀,mod = σ² / λ_mod (eq. (hyperconstants)). They
also set the floor of the hyper-update, τ̂²_b ≥ ν₀ τ²₀,b / (N_w + ν₀ + 2)
(eq. (floor)), with N_w the number of enrolled dyads with at least one
decision in the fit (finished dyads included).

| Agent | τ²₀,act | τ²₀,mod | floor at N_w = 5 (act, mod) | floor at N_w = 25 (act, mod) |
|---|---|---|---|---|
| AYA | 0.0099 | 0.0263 | (0.0036, 0.0096) | (0.0013, 0.0034) |
| CP | 0.0087 | 0.0203 | (0.0032, 0.0074) | (0.0011, 0.0026) |
| REL | 0.0049 | 0.0390 | (0.0018, 0.0142) | (0.0006, 0.0050) |

### 5.3 Reward divisors and per-agent slopes

reward divisor = r_max when `rewards.normalize` (AYA 2, CP 5, REL 5 + 2·w_adh),
else 1; the allocation slope is b_ρ times the divisor (§3.4). The stored
`study_data.reward` is the normalized reward; the native value is kept in
`study_data.outcome["reward_native"]`.

### 5.4 Other reported values

p_h, p_f, p_main, p_mod, λ_main, λ_mod, the run-ins of §3.2, and
ρ(0) = L_min + (L_max − L_min)(1 + c_ρ)^(−κ_ρ).

### 5.5 Not derived from data

- **Noise scale.** σ² is a setting (§3.1). The week-1 estimate of earlier
  drafts is gone; `noise_init` records the design value once, at the first
  learning update (`source = "design"`, no week-1 statistics).
- **Standardization.** Fixed design ranges (§3.1); no baselines are
  estimated, so the per-dyad week-1 and REL population baselines of earlier
  drafts are gone.
- **Monte Carlo bank.** M draws from N(0, 1) under `allocation.mc_seed`,
  generated by `flask init-buffer` and stored in the sample-buffer file (or
  regenerated identically from the seed). The same bank is used for every
  decision of the trial and is never consumed by the action draws.

---

## 6. Complete example file (all defaults)

```yaml
# ADAPTS-HCT RL API hyperparameters.  Every value below is the default from
# main_new.tex Table 1; delete any line you do not change.

discount:            {aya: "13/14", cp: "6/7", rel: 0}
trial_weeks: 14

rel_reward:
  respondent_weights: ["1/3", "2/3"]   # (AYA, CP)
  w_adh: 0.0

noise:
  sigma_sq: {aya: 0.158, cp: 0.122, rel: 0.078}   # fixed by design, normalized reward scale

rewards:
  normalize: true             # learn on reward / r_max (AYA 2, CP 5, REL 5 + 2 w_adh)

penalty:
  m: 2.0                      # lambda_b = m * p_b

eb:
  nu0: 4.0
  lambda0: 0.1

allocation:
  l_min: 0.2
  l_max: 0.8
  c_rho: 3.5                  # rho(0) = 1/3; c_rho = 1 gives rho(0) = 1/2
  b_rho: 20.0                 # native scale; per-agent slope 20 * r_max = 40 / 100 / 100
  kappa_rho: 1.0
  mc_samples: 500
  mc_seed: 12345

warmup:
  n_warm: 5                   # dyads that have completed week 1
  t_warm: 1                   # weeks

numerics:
  j_backups: 1
  k_max: 200
  eps_tol: 1.0e-4
  eps_ridge: 1.0e-6
```

Fractions may be written as decimals or as strings such as `"13/14"`; the
loader parses both. YAML reads `1e-4` (without a decimal point) as a string;
the loader accepts numeric strings too.

---

## 7. Legacy `config.py` keys

These keys belong to the superseded learners and are ignored by
`inf_lsvi_eb`. They will be removed when the legacy learners move to
`app/algorithms/legacy/`.

| Legacy key | Replaced by | Note |
|---|---|---|
| `SMOOTH_ALLOC_LMIN`, `SMOOTH_ALLOC_LMAX`, `SMOOTH_ALLOC_K`, `SMOOTH_ALLOC_MC_SAMPLES`, `SMOOTH_ALLOC_MC_SEED` | `allocation.*` | same meaning |
| `SMOOTH_ALLOC_B` | `allocation.b_rho` | the new key is on the native scale; the slope applied is b_ρ · r_max |
| `SMOOTH_ALLOC_C` | `allocation.c_rho` | **legacy default 5.0 differs from the design's 3.5** |
| `EB_PRIOR_TAU0_SQ` | derived τ²₀,b = σ²/λ_b (§5.2) | no longer a constant |
| `EB_PRIOR_NU0` | `eb.nu0` | legacy default 2.0 (code) / 5.0 (module) vs design 4 |
| `WARMUP_COHORT_MIN_DYADS` | `warmup.n_warm` | now a read-only mirror set at start-up; the rule counts dyads that have **completed** week 1, not started it |
| `WARMUP_WEEKS` | `warmup.t_warm` | read-only mirror set at start-up |
| `MODEL_PRIORS`, `PRIORS_PICKLE_FILE` | — | not used by `inf_lsvi_eb` (the legacy policy row is still written at start-up) |

Operational settings stay in `config.py` and are not hyperparameters: the
database URI, `RL_ALGORITHM`, the sample-buffer path and seed
(`SAMPLE_BUFFER_*`), backups, repro snapshots, the clock mode, and fault
injection. The weekly `/update` is triggered by the host's app server; the
policy publication deadline (`MONITOR_WEEK_READY_BY`, Monday 05:45) and the
other monitoring settings are in `Monitoring-Spec.md` §10.
