# Successor-Representation Critic — Exploration Findings (position-locomotion)

Empirical companion to the theory reference `sf_critic_design_doc.pdf`. This records what was
**tried** and **observed** on `Isaac-Position-v0` (anymal_c, terrain curriculum, 16384 envs), and — kept
separate on purpose — what is **structurally established** vs what is only a **hypothesis**. It is a
checkpoint of a closed exploration; the code keeps the variants selectable (see the `successor` preset and
the `RslRlSuccessorCfg` enums), but the conclusion is that none of the SR-as-value forms beat a scalar
critic on this task, and the principled way forward is the validation protocol at the bottom.

## Bottom line

A scalar PPO critic reaches `Curriculum/terrain_levels/success` ≈ 0.22 and climbing; **every**
SR-as-the-value form tried stays at the random floor (~0.001) or needs a flexible value head that does the
work while the SR is decorative. The representation can be made to *look* healthy (occupancy bounded,
B decorrelated) while the **value it produces does not drive the policy** on this tight-threshold,
subspace-goal task.

## Variants tried and outcome

| anchor | occupancy | read-out | value head | reward | outcome on curriculum |
|---|---|---|---|---|---|
| dynamics | bilinear FB | learned `w` (value-reg) | pure `⟨ψ,w⟩` | task | flat — `w` can't bootstrap from sparse reward → V≈0 |
| dynamics | bilinear FB | hard-√d ψ (pure SR) | pure | task | flat — `⟨ψ,w⟩` too weak/inexpressive |
| dynamics | bilinear FB | learned `w` | hybrid `⟨ψ,w⟩+b(h)` | task | **moved** (~0.10) but `b(h)` carries the value; SR ~decorative |
| dynamics | vector-TD SF | learned `w` | pure | task | stable, self-bounding, but ortho-only φ too thin → weak value |
| dynamics | z-cond bilinear FB | `w=z` (goal embed) | pure `⟨F(s,z),z⟩` | FB-native `⟨B(s'),z⟩` | flat — see the two structural findings below |

## Structurally established (math — independent of any bug)

1. **Subspace goals need a marginalized read-out.** When success depends on `k ≪ d_obs` dims (base pose,
   not perception/joint-config), the correct goal weight is `w = E[r·B]` = `E[B | success-set]`, which
   averages B over states sharing the target subspace: the dims *consistent* across that set (the relevant
   subspace) survive, the *varying* (irrelevant) ones average out. A **single-state** `z = B(goal)` does
   **not** marginalize — it carries that one state's irrelevant dims, so the actor is told to match the
   whole state, not the subspace. This is distinct from observation aliasing.
2. **A smooth occupancy reward cannot express a tight threshold.** The implied reward `r̂(s)=⟨B(s),z⟩` is
   the B-projection of the success *indicator*, so it relaxes a hard "within ε" criterion into a smooth
   alignment that **saturates before the success radius**. Pure goal-reaching FB therefore under-reaches a
   tight criterion; supplying it needs either sufficient B-resolution on the relevant subspace (not
   guaranteed — the reward-free objective allocates capacity by occupancy, not task relevance) or a thin
   explicit terminal success term. So "no reward needed" holds for *shaping*, not for a *tight threshold*.

## Observed collapse modes — HYPOTHESES, NOT VERIFIED

History warning: an earlier confident claim ("F diverges because it's single-task, no z to bound it") was
**wrong** — it was bugs (target-net normalization, ortho_coef, target_tau). Treat the below the same way.

- **CONFIRMED bug:** the FB target network Polyak-copied only `.parameters()`, leaving the
  `EmpiricalNormalization` buffers frozen at init — so the bootstrap was scored under a different
  normalization than the live measure, severely so for large-magnitude obs (absolute pose ~tens of m).
  Fixed by hard-copying buffers in `update_target`.
- **HYPOTHESIS (unverified):** `m_diag` inverting to negative under a concentrated single goal-z
  (off-diagonal squared-TD terms outvoting the one diagonal pull). Signature-consistent; could be another
  bug.
- **HYPOTHESIS (unverified):** `m_diag` drifting past `1/(1−γ)` with slow `f_norm` inflation (Polyak target
  too slow to pin the fixed point under on-policy minibatch reuse; ψ growing in φ-blind directions).
  Signature-consistent; could be another bug.

Do not cite the two hypotheses as settled. The validation protocol below is how to actually find the cause.

## Diagnostics

`m_diag` (= `E[⟨F(s,z),B(s')⟩]`) should approach `1/(1−γ)` and **hold**; `f_norm` bounded and flat; `ortho`
at its floor (B decorrelated). **Caveat:** a rising `value_pred` that tracks `f_norm` inflation is measure
drift, **not** policy improvement — `value_pred` is not a clean health signal under drift.

## Validation protocol (the principled way forward)

Validate each layer before stacking, instead of debugging the full on-policy + new-env + new-task stack at
once:

1. **Reproduce Meta-Motivo** as published (their env, their off-policy learning).
2. **Port only the learning loop** to the rsl_rl structure (keep it off-policy); confirm it still learns.
3. **Reproduce the Meta-Motivo environment in IsaacLab**; confirm reproduction holds.

Then vary one axis at a time:
- **(a)** off-policy, but treat the demos as **discrete reset-states** (not trajectories) — if this trains,
  position-locomotion can be done the same way.
- **(b)** convert the off-policy portion to **on-policy** at IsaacLab scale (16384 envs vs MM's ~50) — if
  this trains, we are close to reproducing it in position.

Either success is a strong result and a much more principled basis than in-place tuning of the full stack.
