# Pre-registration — Structural Role Coverage vs. Talent Accumulation

Registered **before** running the confirmatory. Same discipline as the Structural Influence / Fragility /
Concentration / Portability pre-registrations: ONE primary hypothesis, ONE primary outcome, ONE frozen
talent control, a feasibility gate, franchise/team-season non-independence handling, and a binding
no-fallback clause.

---

## 1. Question

Are successful offenses characterized by structural role **coverage** — the presence of all three core
offensive functions (**creation / connection / conversion**) — rather than by the **accumulation** of
individually high-scoring players? Concretely:

> Conditional on the individual scoring talent on the floor **and** on the linear count of each role,
> does a lineup whose composition **covers all three core functions** (≥1 organizer AND ≥1 shaper AND
> ≥1 terminal) produce more offense (points per possession) than one missing a function?

This is the apex of the role program: taxonomy (roles exist) → orthogonality (roles ≠ value) →
portability (roles travel) → composition→behavior (roles shift play selection) → **coverage→efficiency
(does completeness beat accumulation?)**.

**Why graph-native / why not just lineup net rating.** Lineup studies (5-man net rating, RAPM, EPM,
lineup optimization) answer "*which group performed well*." They cannot answer "*what structural
configuration produced it*," because they have no role-topology object. Here the unit of the hypothesis
is a **configuration property** (coverage), defined from the co-initiation-derived role taxonomy, tested
*net of* the linear accumulation of the same roles and of individual talent. The claim is about the
**completeness of the configuration**, not about any player.

---

## 2. Decision logic (primary hypothesis, metric, single model)

Single possession-level linear model (no two-stage generated regressor):

```
PPP_i = β·COVERED_i
      + a1·n_ORG_i + a2·n_CONN_i + a3·n_TERM_i        (linear accumulation of roles)
      + g·TALENT_i                                    (frozen individual-talent control, §4)
      + FE(team-season) [absorbed] + dummies(opponent, period, score-margin bucket)
      + ε_i
```

- `COVERED_i = 1` iff the on-court five has `n_ORG≥1 AND n_CONN≥1 AND n_TERM≥1` (OCCUPANT is not a core
  function). Because `COVERED` is a **non-linear (threshold/interaction)** function of the counts, and the
  linear counts are included, **β isolates the completeness effect over and above having more of each
  role and over and above talent** — it is not "more organizers = more points."

**T1 (coverage > accumulation) confirmed iff ALL THREE hold (primary tier):**
1. `β > 0`;
2. team-season cluster-robust `p < 0.01`;
3. team-season cluster-**bootstrap** 95% CI of `β` excludes 0.

**Both outcomes are informative:**
- **T1 confirmed** → offensive efficiency reflects *structural completeness*, not just accumulated
  scoring: a lineup that covers creation+connection+conversion outperforms its talent-and-count
  expectation. Front-office read: role *coverage* is a roster-construction target distinct from
  accumulating scorers.
- **T1 not confirmed (honest negative)** → *accumulation, not coverage*: once you know the role counts
  and the talent on the floor, "having all three functions present" adds nothing. Reported as-is (§9).

This is the **terminal confirmatory test** of the role program. No metric/definition swaps after seeing
the result.

---

## 3. Representation & data (reused, no drift)

Possession-level, regular season, **2022-23 / 2023-24 / 2024-25 only** (ON_COURT absent for 2025-26).
Exactly-5 on-court offensive lineups (`ON_COURT {side:'offense'}`, 99.7% of possessions). Role classes
from `structural_role_taxonomy.py` (median-split of usage% × structural influence L(p), computed on the
rotation pool; thresholds frozen there). Outcome `PPP_i = Possession.points_scored`. Play type,
`initiator_player_name`, `defensive_team_tricode`, `period`, `score_margin` are existing possession
properties. Cache: `role_composition_possessions.parquet`.

**Inclusion (frozen):** possessions with **all 5 on-court players classified** (`n_classified = 5`), so
coverage is unambiguous. (`≥4/5` reported as robustness only.)

---

## 4. Frozen talent control (the identification crux)

For each player-season, individual offensive quality:

```
q_p = (Σ points_scored over possessions p initiated that season, LEAVE-CURRENT-GAME-OUT)
      / (count of such possessions, leave-current-game-out)
```

i.e. each player's points-per-initiation, computed **excluding the current possession's game** to remove
mechanical same-game feedback. `TALENT_i = mean(q_p over the 5 on-court players)`. Frozen: **mean** (not
sum); points-per-initiation (not ppg/usage); leave-game-out. Players with <20 season initiations get the
team-season mean q imputed (flagged).

**Stated interpretation ceiling (binding, appears in every writeup):** `q_p` is a *box/initiation*
efficiency measure and therefore **under-measures shapers by construction** (a shaper's value is
largely off the box score). Consequently a positive `β` cannot be cleanly separated from "shaper
talent the control missed"; and because lineups are *chosen*, not assigned, residual bench-unit/game-state
confounding remains after FE. **The maximal licensed claim is "association consistent with a coverage
effect, conditional on measured talent" — NOT a causal or talent-independent optimum.** No decision-support
phrasing ("adding a shaper gains X").

---

## 5. Null / inference

- **Primary SE:** cluster-robust by **offensive team-season**.
- **Primary CI:** team-season cluster **bootstrap** (resample team-seasons with replacement, refit β),
  95% percentile interval; must exclude 0.
- **Robustness (reported, not decisive):** franchise-clustered bootstrap; model **with play-type FE**
  (within-play efficiency vs the selection channel); `≥4/5`-classified inclusion; talent control removed
  (naïve β, to show how much talent alone explains).

---

## 6. Non-independence

Possessions nest in team-seasons nest in franchises. Inference clusters at team-season (primary) and
franchise (robustness). If distinct franchises `< 20` or team-seasons with both COVERED and NOT-COVERED
strata `< 20`, the run is flagged **UNDERPOWERED** but the pre-committed rule is still applied.

---

## 7. Two-stage protocol

### 7.1 Feasibility gate (`coverage_gate.py`) — feasibility ONLY, β NOT computed
Passes only if all three hold:
- **G1 coverage varies:** `≥20` team-seasons have `≥200` possessions in **each** of COVERED and
  NOT-COVERED (so β is estimable within-team).
- **G2 coverage separable from talent & counts:** `|Spearman(COVERED, TALENT)| < 0.80` **and** `COVERED`
  is not near-perfectly explained by the linear counts (auxiliary regression `R² < 0.95`, i.e. the
  threshold carries independent variation). If either fails, the test is vacuous → gate FAILS.
- **G3 talent control behaves:** `TALENT` has a positive raw association with `PPP` (sanity).

The gate deliberately does **not** compute β or the covered-vs-not PPP contrast (no optional-stopping).
Any single failure ⇒ do **not** run the confirmatory.

### 7.2 Confirmatory (`coverage_full_study.py`) — one-shot
Run only if the gate passes. Fit the §2 model, compute β, cluster-robust p, and the team-season
cluster-bootstrap CI. Apply §2 rule. **Result recorded verbatim in §8 before any interpretation.**
Frozen settings: `SEED=17`, `N_BOOT=2000`, `CONF_P=0.01`, score-margin buckets
`{<-15,-15..-8,-8..-3,-3..3,3..8,8..15,>15}`, period categories `{1,2,3,4,OT}`.

---

## 8. Gate + confirmatory outcomes (to be appended before interpretation)

### 8.1 Feasibility gate (`coverage_gate.py`, run 2026-07-13) — **PASSED**
Sample: 651,784 possessions pulled; **420,993** with all 5 on-court classified + pts present
(frozen inclusion). Role thresholds: usage median 0.181, influence median 0.009. 90 team-seasons, 30 opponents.

- **G1 coverage varies — PASS.** Overall COVERED rate = 30.0%. Team-seasons with ≥200 possessions in
  BOTH covered and not-covered = **72** (need ≥20).
- **G2 separable — PASS.** `|Spearman(COVERED, TALENT)| = 0.042` (need <0.80); aux OLS
  `R²(COVERED ~ linear counts) = 0.203` (need <0.95). Coverage is near-orthogonal to the talent control
  and is not mechanically implied by the role counts.
- **G3 talent behaves — PASS.** `Spearman(TALENT, PPP) = +0.027` (need >0).

Verdict: **GATE PASSED (G1=G2=G3=True)** → proceed to the one-shot confirmatory. (β was not computed at
the gate.)

### 8.2 Confirmatory (`coverage_full_study.py`, run 2026-07-13) — **T1 NOT CONFIRMED**
Sample: 420,993 possessions (all 5 classified), 90 team-seasons, overall PPP = 1.2174.
Coefficients are in PPP units; ×100 = points per 100 possessions.

**M1 PRIMARY (no play-type FE):**
- **COVERED: b = +0.0067 (+0.67 pts/100), cluster-robust t = 1.35, p = 0.176.**
- team-season cluster-bootstrap 95% CI: **[−0.28, +1.65] pts/100 — includes 0**.
- franchise cluster-bootstrap 95% CI: [−0.27, +1.58] — includes 0.
- Accumulation terms: n_ORGANIZER +2.24 pts/100 (t=7.29); n_SHAPER +0.53 (t=1.35);
  n_TERMINAL +0.08 (t=0.18); TALENT +3.98 pts/100 (t=0.60).

**Decision rule (§2):** b>0 = True; p<0.01 = **False** (0.176); CI excludes 0 = **False**.
→ **T1 NOT CONFIRMED.**

**Robustness (all directionally positive, none clears the bar):**
- M2 (with play-type FE): COVERED +0.64 pts/100, t=1.32, p=0.188, CI [−0.32,+1.59] incl. 0.
- Naïve (talent removed): COVERED +0.68 pts/100, t=1.35, p=0.177, CI [−0.28,+1.67] incl. 0.
- ≥4/5 inclusion (n=581,640): COVERED +0.83 pts/100, t=1.82, **p=0.068**, CI [−0.08,+1.75] incl. 0.

**Verdict (per §9): honest negative — ACCUMULATION, not coverage.** Conditional on the linear role
counts and measured talent, having all three core functions present adds no detectable offense
(point estimate small-positive, ~+0.7 pts/100, but not distinguishable from 0 at the registered bar).
This is the **terminal confirmatory test of the role program**; no metric/definition/outcome/control
swaps (§9). The one robust scoring driver is accumulation of organizers (n_ORGANIZER, +2.2 pts/100).

---

## 9. Stop conditions / no-fallback (binding)

- Gate fails **G1** (no coverage variation): report "coverage not separately estimable"; stop.
- Gate fails **G2** (coverage collinear with talent or mechanically implied by counts): report the test
  is vacuous at this granularity; stop. No re-defining coverage to manufacture separation.
- Confirmatory `β ≤ 0` or fails p / CI: **honest negative — "accumulation, not coverage."** No second
  coverage definition, no alternate outcome (shot quality / TOV / ORB / entropy remain **exploratory,
  non-confirmatory** — testing them as primaries would be multiple-outcome fishing), no alternate talent
  control. Terminal test of the role program.

---

## 10. Out of scope

- **Causation / optimal-topology claims.** §4 ceiling binds: association conditional on measured talent,
  not a talent-independent optimum, not "build this lineup."
- **Player value / "best lineup."** Not a lineup-rating or impact metric; the object is a configuration
  property (coverage) net of accumulation and talent.
- **Outcomes other than PPP** (shot quality, turnover rate, offensive rebounding, play-type entropy,
  transition rate): exploratory description only, explicitly outside the confirmatory tier to prevent a
  multiple-outcome fishing expedition.
- **2025-26** (no ON_COURT). **No data outside `graph_schema.md`.**

---

## 11. Provenance

- **Reuses (no drift):** role classes from `structural_role_taxonomy.py`; possession cache + class map
  from `role_composition_playmix.py`; fixed-effects + cluster-bootstrap machinery from
  `role_composition_ppp.py` / `concentration_full_study.py`.
- **New for this study:** `coverage_gate.py` (feasibility), `coverage_full_study.py` (T1), the frozen
  `q_p` talent control. Each committed before its run.
- **One-and-only-correction clause (binding):** at most one pre-run, clearly-motivated specification
  correction, timestamped before the confirmatory run. No post-result changes.
