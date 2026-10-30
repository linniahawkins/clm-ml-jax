# Phase 7 decision log

Decisions Claude makes while doing the Phase 7 work packages, kept for posterity: anything the
scope (`phase7_scope.md`) and the maintainer's decisions (plan §0, decision 35; scope D1-D12)
do not already settle. Examples: names and file layout, how an adapter works, the order of a
move, a test skipped or added, a deviation from the scope and why, an option considered and
rejected. Science choices are not made here: they go to the maintainer (and plan §0).

Add an entry in the same commit as the change. Number entries in order (C1, C2, ...), never
renumber; if a later decision replaces one, say so in both.

Format:

```
### C<n>. <short title> (<package>, <date>, <commit>)
Context: what needed deciding.
Decision: what was chosen.
Why: the reason (measured where possible).
Alternatives: what was considered and why not.
```

## Decisions

### C0. Minimum testing for Phase 7 (setup, 2026-10-30)
Context: the maintainer asked to cut testing to the bare minimum to speed up Phase 7.
Decision: per commit only lint, the bit-for-bit + noise-twin test and the moved routines'
golden tests; a focused CPU gradient test when a routine with a hand-written gradient moves;
the CPU gradient tier and touched feature tests at package ends; everything else, including
one GPU gradient job, at the end of Phase 7 (`phase7_progress.md`). Replaces D12's "GPU
gradient job at package ends".
Why: Phase 7 does not change arithmetic, and the bit-for-bit test compares every field after
every step; gradients can change without the forward values changing only where gradients
are hand-written (implicit-function solvers, `stop_gradient`), so those moves get a gradient
check.
Alternatives: the Phase 6 tiers (fast tier after each edit, feature tests per commit, GPU at
package ends): safer, slower; GPU at package ends dropped because the CPU tier with
optimization off checks the same gradients without the queue.
