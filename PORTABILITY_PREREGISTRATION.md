# Pre-registration — Structural Role Portability ("Does the wiring travel with the player?")

Registered **before** running the confirmatory. Same discipline as `WIRING_PREREGISTRATION.md`,
`FRAGILITY_PREREGISTRATION.md`, `ABSENCE_VALIDATION_PREREGISTRATION.md`,
`WIRING_CONCENTRATION_PREREGISTRATION.md`: ONE primary hypothesis, ONE metric, a feasibility gate,
a decoy null, franchise/player non-independence handling, and a pre-committed no-fallback clause.

---

## 1. Question

Fragility (confirmed) showed that a player's **structural load** `L(p)` — how much his removal re-wires
his team's play-type co-initiation topology — is a real, usage-decoupled quantity. It left one thing
undetermined that a front office would actually pay differently for:

> **Is a player's structural role a portable property of the *player*, or an emergent property of the
> *team* he happens to be on?**

Operationally: when a player appears on two different franchises, does the way he re-wires the offense
(his **adjacency-displacement signature** `ΔA_p`) travel with him — *beyond what his own play-type
profile trivially predicts* — or does it reset to whatever the destination system makes of him?

This is the one direction on the original list never tested, and it matches the pattern of what has
held up: **the dependent variable is structure, not performance.** We never ask whether the player
makes the new team better (that is the crowded impact-metric space where V2 / SRP / concentration all
failed). We ask only whether the *wiring pattern* is a player trait.

**Why graph-native (essential, not convenient).** The trivial version — "does a PNR guard run PNR
everywhere" — is answerable from a play-type table and is explicitly the thing we *control away*. The
claim is about the **relational displacement pattern** (which play-type associations collapse when he
is removed), which exists only in the co-initiation graph and is not a per-player tabular feature.

---

## 2. Decision logic (primary hypothesis + metric)

**T1 (portability):** across movers, the within-player cross-franchise signature similarity exceeds
the profile-matched decoy similarity:
```
Δ(p) = s_self(p) − s_decoy(p)
     = cos(ΔA_p^{f1}, ΔA_p^{f2})  −  mean_{q∈D(p)} cos(ΔA_p^{f1}, ΔA_q^{f2})
```
**T1 confirmed iff ALL THREE hold** (primary tier):
1. `median_p Δ(p) > 0`;
2. player-clustered permutation `p < 0.01` (self is not special among profile-matched destination
   players under H0);
3. destination-franchise cluster-bootstrap 95% CI of `median Δ` **excludes 0**.

**Interpretation is symmetric and both outcomes are useful (§10 states the boundary):**
- **T1 confirmed** → structural role is a *scoutable player trait*. A trade target's wiring on his
  current team is informative about the wiring you are acquiring; hidden hubs (Gobert-type) are
  systematically mispriced by usage-based valuation and can be identified on other rosters *before* a
  move.
- **T1 not confirmed (honest negative)** → structural role is *emergent from the system*. This is a
  warning with teeth: any "he will organize our offense the way he organized theirs" claim is
  unsupported, and it cleanly explains why the V2 absence simulation failed (it assumed roles are
  fixed player properties). Reported as-is, no fallback (§9).

This is the **last** planned test in this line of inquiry. If T1 is not confirmed, the program's
contribution is the confirmed descriptive result (Structural Load + Fragility); no further variants.

---

## 3. Representation (reused, no drift)

Exactly the Structural Load / Fragility object. L2 player→play-type initiation biadjacency `B` (rows = players,
cols = the 8 registered play types `TRANSITION, PNR, DRIVE, POST_UP, SPOT_UP, CUT, PUTBACK, PULL_UP`),
per regular-season team-season. Association matrix `A` = min-overlap co-initiation
`O[i,j]=Σ_player min(B[·,i],B[·,j])` (i≠j, diagonal 0), standardized against the fixed-both-margins
token-shuffle null. Identical `assoc_O`, `null_O_stats`, `standardize` from `wiring_gate.py`.

---

## 4. Player-level objects (frozen definitions)

For player `p` on an included team-season `(tri, yy)` with `A_full` (all players) and `A_{−p}` (p removed,
each standardized against its own margin-preserving null, exactly as `fragility_gate.loo_loads`):

- **Adjacency-displacement signature** `ΔA_p = A_full − A_{−p}`, vectorized on the **28 off-diagonal
  upper-triangle cells** (`np.triu_indices(8,k=1)`). This is the Fragility heatmap object. Signatures
  are compared by **cosine similarity** `cos(x,y)=x·y/(‖x‖‖y‖)` (pattern, not magnitude — consistent
  with the Fragility primary being cosine-based).
- **Play-type profile** `π_p = B_full[p] / Σ B_full[p]`, the player's own initiation distribution over
  the 8 play types (an 8-vector). This is the **control**: the trivially-portable, table-recoverable
  part of a player's identity. Decoys are matched on `π`.

Player inclusion: `init(p) ≥ 50` in the team-season (Fragility `MIN_INIT_SEASON`), team-season passes
Structural Load §6 inclusion (`≥300` off. poss and `≥6` players with `≥20` init per half; we use the
season-level analogue already enforced by the Fragility loads pipeline).

---

## 5. Movers, decoys, and the null

- **Mover:** a player with a valid signature under **≥ 2 distinct franchises** (`tri`). For each mover
  we select exactly **one signature per franchise** — the two franchises with the largest total
  initiations for that player, and within each the team-season with the most initiations. This yields
  **one `Δ(p)` per player** (fully independent at the player level; no within-player repetition).
  Label the higher-init franchise `f1` (anchor / origin), the other `f2` (destination).
- **Decoy pool `D(p)`:** players `q ≠ p` on the **same destination team-season** as `f2` (init ≥ 50)
  whose play-type profile is closest to p's destination profile — the **K = 5** nearest by
  `cos(π_q, π_p^{f2})`. A mover needs `≥ 3` decoys or it is dropped. Decoys are on the *same* destination
  context, so destination-team style is **held constant** in the comparison; the only thing that varies
  between `s_self` and `s_decoy` is *identity of the person* (matched on play-type profile).
- **Null (permutation):** under H0, p's origin signature is no more similar to his own destination
  signature than to a profile-matched destination player's. For each mover pool
  `{cos(ΔA_p^{f1}, ΔA_p^{f2})} ∪ {cos(ΔA_p^{f1}, ΔA_q^{f2}) : q∈D(p)}`, randomly designate one element
  as "pseudo-self" and set `Δ_null(p) = pseudo_self − mean(rest)`. `median_p Δ_null` over all movers is
  one null draw; `≥ 5000` draws give the permutation p-value. This exactly nulls the "is self special"
  claim while preserving each mover's own decoy geometry.

---

## 6. Non-independence

- Unit of the hypothesis = **player** (one `Δ` each), so there is no within-player pseudo-replication.
- Residual dependence is through the **destination franchise** (two movers landing on the same team
  share its context and can appear in each other's decoy pools). Primary CI = **destination-franchise
  cluster bootstrap** (resample destination franchises with replacement, take all movers landing there).
  Secondary robustness = origin-franchise cluster bootstrap and unclustered player bootstrap; all three
  reported, primary is decisive.
- Effective N is the number of movers and the number of distinct destination franchises. If distinct
  destination franchises `< 20`, the run is flagged **UNDERPOWERED** (as in Fragility §9) and the
  verdict is annotated accordingly, but the pre-committed rule is still applied.

---

## 7. Two-stage protocol

### 7.1 Feasibility gate (`portability_gate.py`) — feasibility ONLY, no peeking at Δ
Passes only if all three hold:
- **G1 sample exists:** `≥ 25` movers with valid two-sided signatures. (Data check earlier found 296
  multi-franchise players; the floor guards against attrition from the init/decoy filters.)
- **G2 decoy pools populated:** `≥ 80%` of movers have `≥ 3` profile-matched decoys.
- **G3 not vacuous:** signature similarity is **not** trivially determined by profile similarity.
  Across a large random sample of cross-player signature pairs, `|Spearman(cos_profile, cos_signature)|
  < 0.90`. If `≥ 0.90`, the control and the object are collinear, the test is vacuous, and the gate
  **FAILS** (report the negative; no fallback — mirrors Concentration §12 G3).

The gate deliberately does **not** compute `median Δ` or the self-vs-decoy contrast (an appetizer of
the primary statistic would create optional-stopping). Any single failure ⇒ do **not** run the full
study.

### 7.2 Confirmatory (`portability_full_study.py`) — one-shot
Run only if the gate passes. Compute `median Δ`, the player-clustered permutation p (≥5000), and the
destination-franchise cluster-bootstrap 95% CI (≥2000). Apply §2 rule. **Result recorded verbatim in
§8 of this file before any interpretation.** One-shot: no re-runs, no metric/decoy/K changes after
seeing the result.

Frozen settings: `SEED=17`, `N_NULL=300` (standardization nuisance scale; signatures use cosine and are
insensitive to this — stated now, not chosen after), `K=5` decoys, `N_PERM=5000`, `N_BOOT=2000`,
`CONF_P=0.01`.

---

## 8. Gate + confirmatory outcomes (recorded 2026-07-12, before interpretation)

### 8.1 Feasibility gate (`portability_gate.py`)
Signature cache: 1,962 player-signatures across 120 team-seasons (662 players).

- **G1 sample exists:** PASS — 296 movers with valid two-sided signatures (≥25).
- **G2 decoy pools:** PASS — 296/296 movers (1.00) have ≥3 profile-matched decoys (≥0.80).
- **G3 not vacuous:** PASS — |Spearman(cos_profile, cos_signature)| = **0.048** (<0.90). The
  displacement signature is essentially uncorrelated with the play-type profile, so the test is
  well-identified — portability, if found, is not a restatement of "he runs the same plays."

**Gate verdict: PASSES.** Proceeded to the one-shot confirmatory.

### 8.2 Confirmatory (`portability_full_study.py`, one-shot, verbatim)
seed 17, N_NULL 300, K=5 decoys, 5,000 permutations, 2,000 bootstraps.

- Movers analyzed: **296**; distinct destination franchises: **30** (not underpowered).
- Median within-player cross-franchise similarity (`self_sim`) = **+0.148**; median profile-matched
  decoy similarity (`decoy_sim`) = **+0.092**.
- **T1 median Δ = +0.0819.**
  1. median Δ > 0 — **PASS**
  2. player-clustered permutation **p = 0.0000** (<0.01) — **PASS**
  3. destination-franchise cluster-bootstrap 95% CI **[+0.037, +0.125]** excludes 0 — **PASS**
  - Robustness: origin-franchise CI [+0.035, +0.123]; unclustered player CI [+0.041, +0.125] — both
    exclude 0.

**VERDICT: T1 CONFIRMED.** A player's adjacency-displacement signature travels across franchises
**beyond his play-type profile**: his own destination wiring is reliably more similar to his origin
wiring than a profile-matched destination teammate's is. Structural role is (in part) a **portable
player trait**, not purely team-emergent. All three primary criteria and all robustness clusterings
agree.

Interpretation boundary (per §10): this is an *associational* statement about the wiring *pattern*
co-traveling with the player; it is not a claim about player value, performance, or causation, and
origin-side selection is not fully ruled out. Effect size is modest (median self−decoy ≈ 0.06 in
cosine units) but robust across 296 players and every clustering scheme.

---

## 9. Stop conditions / no-fallback (binding)

- Gate fails on **G1 or G2** (infeasible sample): report "insufficient data to test role portability",
  stop. Not a scientific negative, a feasibility limit.
- Gate fails on **G3** (vacuity): report that the signature is not separable from the play-type profile
  at this granularity; stop. No switching to a different signature/embedding to force separability.
- Confirmatory `median Δ ≤ 0` or fails the permutation/CI thresholds: **honest negative** — structural
  role is context-emergent, not a portable player trait. Reported as the finding (§2). **No** second
  metric, **no** K sweep, **no** re-definition of mover/decoy. This is the terminal test of the line.

---

## 10. Out of scope

- **Player value / impact / performance.** `ΔA_p` is a structural displacement pattern, not points,
  wins, or quality (Fragility §10 boundary). Portability says the *wiring pattern* travels, not that the
  player is good or makes teams better.
- **Causation & selection.** Players are not randomly assigned to teams; a portable signature could
  partly reflect that similar players choose/are-chosen-by similar systems. The decoy design holds the
  *destination* context constant but cannot fully rule out origin-side selection; the claim is
  associational ("the wiring pattern co-travels"), not "the player causes the wiring".
- **No data outside `graph_schema.md`.** Signatures and profiles come solely from `INITIATED` edges
  and `play_type` already used by Structural Load / Fragility.

---

## 11. Provenance

- **Reuses (no drift):** invariant/null/standardization and `IU` from `wiring_gate.py`; all-teams edge
  pull + split from `wiring_full_study.py`; `build_B_labeled` / leave-one-out construction from
  `fragility_gate.py` (`loo_loads`), extended to also emit the `ΔA` off-diagonal vector and the
  play-type profile.
- **New for this study:** `portability_common.py` (per-(team-season,player) signature+profile cache),
  `portability_gate.py` (feasibility), `portability_full_study.py` (T1). Each committed before its run.
- **One-and-only-correction clause (binding):** at most one pre-run, clearly-motivated specification
  correction, timestamped before the confirmatory run. No post-result changes.
