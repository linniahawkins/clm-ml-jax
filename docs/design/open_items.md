# Open items: tasks, decisions and issues

One list of everything still open after Phase 7, gathered from the design documents
(`global_input.md`, `phase6_progress.md`, `phase7_scope.md`, `phase7_progress.md`,
`phase7_decisions.md`, `modernization_plan.md`, `numerics_and_regression.md`, `timing.md`),
CLAUDE.md, the code, and the Phase 8 survey. Each item keeps its original IDs under
"Source" so its history can be traced. Items appearing in several places are merged.

**Targets:** **8a** = housekeeping before the documentation; **8b** = documentation and
release; **after 8** = after Phase 8 (already planned that way); **known** = a known model
limitation, not to be changed without discussion. **Decides:** **M** = maintainer (science
or project choice); **I** = implementation (can be done without a science decision, reviewed
as usual).

**How to use:** IDs are stable (never renumber). A new item gets the next free ID, in its
group and in the summary table. When an item is settled, move it to "Recently closed" with
the commit and, if a decision was made, its number in [`decisions.md`](decisions.md).
The model as it is now: [`model_overview.md`](model_overview.md).

## Summary

| ID | Title | Kind | Target | Decides |
|---|---|---|---|---|
| **1. Science issues needing the maintainer's decision** |||||
| O1 | ~~Soil texture class looked up one class too far~~ (closed: decision 92) | science issue (bug) | closed | M |
| O2 | Missing `organic_matter` gives a negative organic fraction | science issue (bug) | 8a | M |
| O3 | Soil ice read one level too deep in `SoilResistance` | science issue (bug, latent) | 8a | M |
| O4 | Fortran's checks of the implicit flux-profile solution not run | science issue | 8a | M |
| O5 | Solver stopping rules differ from Fortran (Obukhov, gs, ci) | science issue (fidelity) | 8a decide / after 8 fix | M |
| O6 | Latent heat outputs of transpiration and canopy evaporation use the last sub-step's latent heat | science issue | 8a | M |
| O7 | LAI and SAI gradients have a kink at the minimum plant area | science issue | after 8 | M |
| O8 | Minimum plant area protects radiation only for clumping above ~0.55 | science issue | after 8 | M |
| O9 | Frozen soil: all prescribed soil water is liquid | science issue | after 8 | M |
| O10 | No snowpack on the ground | science issue | after 8 | M |
| O11 | Root biomass 500 g m⁻² everywhere; canopy bottom height unused (grid runs) | science issue | after 8 | M |
| O12 | Initial state from monthly means, not a spun-up restart (grid runs) | science issue | after 8 | M |
| O13 | Forcing height fixed at setup; CLM6 uses Meier2022 roughness (varies with LAI) | science issue | after 8 | M |
| O14 | Very short canopies (down to 5 mm) not filtered | science issue | after 8 | M |
| O15 | Wet needleleaf canopies: large latent heat with CLM6's `dleaf` | science issue | after 8 | M |
| O16 | CO₂ profile reset covers the wrong levels (no effect today) | science issue (latent) | after 8 | M |
| **2. Other maintainer decisions pending** |||||
| O17 | ~~Repository organization: where the real configuration files live~~ (closed: decisions 89, 90) | maintainer decision | closed | M |
| O18 | Norman and well-mixed code in git history only (D8 said "kept") | maintainer decision | 8a | M |
| O19 | Options that only ever refuse (`wue_solver: brent`, `mlcan_to_clm`) | maintainer decision | 8a | M |
| O20 | `nitrogen_decay_kn` is a static option, not a differentiable parameter | maintainer decision | 8a | M |
| O21 | Public API: which names `import clm_ml` offers | maintainer decision | 8a | M |
| O22 | Review of Phase 7 behaviour changes and re-recorded references | maintainer decision | 8a | M |
| O23 | Authorship, citation and canonical repository | maintainer decision | 8a | M |
| O24 | Release logistics: `main` 156 commits behind; version 1.0.0; PyPI or not | maintainer decision | 8a (merge) / 8b (release) | M |
| **3. Bugs and robustness** |||||
| O25 | XLA CPU gradient crash (issue 16) | bug (upstream JAX) | known / recheck on JAX upgrade | I |
| O26 | XLA CPU compile crash (LLVM linker recursion) | bug (upstream JAX) | known / recheck on JAX upgrade | I |
| O27 | GPU: two fresh compiles can differ by ~1e-11 | bug (reproducibility) | after 8 (GPU work) | I |
| O28 | A patch failing a check still enters its chunk's gridcell sums | robustness | 8a | M |
| O29 | Padding slots combine one gridcell's weather with another patch's data | robustness | 8a (verify) | I |
| O30 | Smaller Phase 6 review items (R11) | robustness | 8a (verify) | I |
| O31 | netCDF read on a background thread while writing on the main thread | robustness | 8a | I |
| O32 | `Model.chunks()` start at step 1 while runs chunk from step 0 | robustness (API) | 8a / 8b document | M |
| O33 | The `euler` tableau has `b = [0]` (trap for future code) | robustness | 8a | I |
| **4. Housekeeping and repository organization (8a)** |||||
| O34 | `.gitignore` names absent directories (`docs/` rule removed) | housekeeping | 8a | I |
| O35 | Empty `.gitmodules`; CONTRIBUTING describes a submodule | housekeeping | 8a | I |
| O36 | `.github/copilot-instructions.md` describes deleted packages | housekeeping | 8a | M (delete or rewrite) |
| O37 | README out of date (badges, features, citation, PR workflow) | housekeeping | 8a | M (with O23) |
| O38 | CONTRIBUTING out of date | housekeeping | 8a | I |
| O39 | `data/namelists/README.md` describes the retired `clm-ml-offline` | housekeeping | 8a | I |
| O40 | Namelists: `nl.US-UMB.test` points to scratch; `nl.CHATS7.debug`, `nl.CHATS7.test3` unclear | housekeeping | 8a | M |
| O41 | ~~Example files not checked~~ (closed: `configs/` checked by `test_configs.py`) | housekeeping | closed | I |
| O42 | Stale design documents (plan header and §8, `api_sketch.py`, numerics §4 title, `timing.md`) | housekeeping | 8a | I |
| O43 | CI lint skips `timing/` | housekeeping | 8a | I |
| O44 | ~~One open-items list and one decision log~~ (done: this document, `decisions.md`, `model_overview.md`) | housekeeping | closed | I |
| O65 | `numerics.checkpoint_steps` in model.yaml does nothing | housekeeping | 8a | I |
| O66 | PFT parameters used only at setup have no effect through `Model.run_chunk` | maintainer decision | 8a | M |
| O67 | PFT optics keep the old 1-based layout `[unused, vis, nir]` | housekeeping | 8a | I |
| O68 | Stale docstrings and comments (`model.py`, `legacy_ascii`, `clm_ml/__init__.py`, CLAUDE.md `rtol`) | housekeeping | 8a | I |
| O69 | Plan says output is a nested scan (records × steps); the code is one scan with a record slot per step | housekeeping | 8a | I |
| O70 | ~~Implement experiment folders~~ (closed: decision 91 implemented) | housekeeping | closed | I |
| **5. Documentation (8b)** |||||
| O45 | Getting-started guide | documentation | 8b | I |
| O46 | Reference of every YAML key, generated from the code | documentation | 8b (code part in 8a) | I |
| O47 | Examples: gradient calibration, ensemble, multi-tower, grid | documentation | 8b | M (calibration target) |
| O48 | Page for Fortran CLM-ml users (namelist and option codes to YAML) | documentation | 8b | I |
| O49 | API reference from docstrings | documentation | 8b | I |
| O50 | CONTRIBUTING: "add a process variant", testing tiers, regression policy | documentation | 8b | I |
| O51 | Output caveats for users (`soil_water_stress` = 1 with WUE; NaN above the top layer; `stem_area_added`) | documentation | 8b | I |
| O52 | Release 1.0.0: version, CHANGELOG, CITATION.cff, tag | documentation | 8b | M |
| **6. Deferred to after Phase 8** |||||
| O53 | Round-off amplification in real 1° patches (issue 10) | science issue | after 8 | M |
| O54 | Bare-ground patches (issue 13) | deferred feature | after 8 | M |
| O55 | GPU performance | performance | after 8 | M |
| O56 | Unsupported variants: well-mixed, dataset profiles, Norman, Brent | deferred feature | after 8 | M |
| O57 | Option B: shared soil columns | deferred feature | after 8 | M |
| O58 | Orbital parameters of the start year only | known departure | after 8 (not scheduled) | M |
| O59 | Forcing preprocessed once to model-step files (repeated runs) | deferred feature | after 8 (idea) | M |
| O60 | Reduced-precision (float32) option | deferred feature | after 8 (idea) | M |
| **7. Known model limitations** |||||
| O61 | Strongly stable conditions: near-ground air too warm | known limitation | known | M |
| O62 | Daytime latent heat biased high (~70 W m⁻²) | known limitation | known | M |
| O63 | Soil heat flux biased high | known limitation | known | M |
| O64 | Soil moisture prescribed, not predicted | known limitation | known | M |

---

## 1. Science issues needing the maintainer's decision

### O1. Soil texture class looked up one class too far
- **Closed 2026-11-01:** fixed (decision 92; `tests/clm_ml/test_setup_soil.py`).
- **Kind:** science issue (translation bug) · **Target:** 8a · **Decides:** M
- **What:** a tower given only `texture_class` (no sand/clay percentages) gets the next
  class's sand, clay and Clapp-Hornberger values ("loam" gets "sandy clay loam"), and
  "clay" (class 11) crashes with an index error. Fortran indexes 1-based arrays, so it has
  no shift. Wrong soil hydraulic and thermal properties for those towers.
- **Now:** kept bit for bit in `clm_ml.setup.soil.soil_properties` (docstring flags it,
  `src/clm_ml/setup/soil.py:88`). CHATS7 and grid runs give percentages: not affected.
- **Next:** fix (index `tex - 1`), add a test with "loam" and "clay"; changes only those
  towers' soils.
- **Source:** `global_input.md` issue 21; C27; Phase 7 "Next".

### O2. Missing `organic_matter` gives a negative organic fraction
- **Kind:** science issue (bug) · **Target:** 8a · **Decides:** M
- **What:** towers.yaml allows `organic_matter: null`; it becomes -999, and the setup
  divides by 130 kg m⁻³ (`om_frac` = -7.7), so porosity, suction, b, conductivities and
  heat capacity are extrapolated far outside their range.
- **Now:** kept bit for bit (`src/clm_ml/setup/patches.py:83`). CHATS7 (50 kg m⁻³) and
  grid runs (surface data) not affected.
- **Next:** choose: 0 (mineral soil) or an error asking for a value.
- **Source:** `global_input.md` issue 22; C27; Phase 7 "Next".

### O3. Soil ice read one level too deep in `SoilResistance`
- **Kind:** science issue (latent bug) · **Target:** 8a · **Decides:** M
- **What:** Fortran stops root uptake from a level holding ice; the JAX code tests the ice
  of the level below. No effect today (ice is always 0, see O9); would matter once frozen
  soil is represented.
- **Now:** kept (`src/clm_ml/step.py:290`, comment cites C14).
- **Next:** one index; bit for bit while ice is 0.
- **Source:** `global_input.md` issue 19; C14; C21.

### O4. Fortran's checks of the implicit flux-profile solution are not run
- **Kind:** science issue (checking) · **Target:** 8a · **Decides:** M
- **What:** Fortran `ErrorCheck01/02` stop the run if the implicit solution disagrees with
  the leaf and soil fluxes or does not conserve energy and water vapor (15 checks). No JAX
  run has ever checked them.
- **Now:** `clm_ml.physics.flux_profile.implicit_errors` computes all 15 and a unit test
  runs them; they are not in `BALANCE_CHECKS`.
- **Next:** decide whether every run records them (the state's `balance_error` grows; the
  reference is re-recorded).
- **Source:** `global_input.md` issue 20; C17.

### O5. Solver stopping rules differ from Fortran
- **Kind:** science issue (fidelity) · **Target:** decide in 8a, fix after 8 · **Decides:** M
- **What:** three iterative solvers differ from Fortran's algorithms: Obukhov length (25
  fixed secant steps vs Fortran's secant/Brent hybrid, tol 0.1 m); stomatal conductance for
  WUE (20 halvings plus a Newton step vs Fortran's bisection to 0.001 mol m⁻² s⁻¹, so gs
  differs by up to ~1e-2 relative); intercellular CO₂ for Medlyn/Ball-Berry (40 secant steps
  plus Newton vs hybrid, tol 0.1). All within the validated agreement (CHATS7 ≤ 2.5 %).
- **Now:** unchanged since before the modernization; recorded as known deviations.
- **Next:** decide: keep (more converged, differentiable) and document, or a fidelity step
  matching Fortran's stopping rules (validated against Fortran; a science change).
- **Source:** `numerics_and_regression.md` §4.1-4.3; CLAUDE.md "Iterative solvers".

### O6. Transpiration and canopy-evaporation latent heat outputs use the last sub-step's latent heat
- **Kind:** science issue (diagnostic consistency) · **Target:** 8a · **Decides:** M
- **What:** decision 32 fixed the canopy-top latent heat (averaged sub-step by sub-step).
  The outputs `transpiration_latent_heat_flux` and `canopy_evaporation_latent_heat_flux`
  (and the legacy `.out` writer) still multiply the averaged water flux by the last
  sub-step's latent heat: slightly inconsistent on steps where the air crosses 0 °C. Not in
  any check; CHATS7 never crosses freezing.
- **Now:** unchanged.
- **Next:** decide whether to apply decision 32's treatment to these outputs too.
- **Source:** `global_input.md` issue 12 ("Still using the last sub-step's latent heat").

### O7. LAI and SAI gradients have a kink at the minimum plant area
- **Kind:** science issue (differentiability) · **Target:** after 8 · **Decides:** M
- **What:** in a layer held at 0.01 m² m⁻² (decision 30), the layer's total plant area does
  not respond to LAI or SAI, and the slope changes where a layer crosses the floor; the
  gradient with respect to SAI is zero in floored layers. Matters for calibration and
  sensitivity with respect to LAI/SAI.
- **Now:** the floor as decided (decision 30).
- **Next:** a smooth alternative would change forward values near the threshold: science
  choice. Related to O54 (bare ground).
- **Source:** `global_input.md` issue 17b; decision 30.

### O8. Minimum plant area protects radiation only for clumping above ~0.55
- **Kind:** science issue · **Target:** after 8 · **Decides:** M
- **What:** the 0.01 m² m⁻² floor keeps the diffuse transmittance below 1 only when
  `clumping_index` is above ~0.55. All built-in PFTs use 1.0, but a calibration could push
  it lower and bring back the runaway leaf temperatures of issue 14.
- **Now:** not guarded.
- **Next:** decide: bound the parameter, or a radiation formulation that stays well
  conditioned (see O54 option 1).
- **Source:** Phase 6 review R10; `global_input.md` issue 14.

### O9. Frozen soil: all prescribed soil water is liquid
- **Kind:** science issue · **Target:** after 8 · **Decides:** M
- **What:** `H2OSOI` is total water; CLM-ml puts it all in liquid, so frozen soils supply
  liquid water to roots and have the thermal properties of wet unfrozen soil. CLM keeps the
  model's ice fraction (`SOILLIQ`, `SOILICE` are in the history files).
- **Now:** ice always 0 (as Fortran CLM-ml).
- **Next:** science change for global/cold sites; also activates O3.
- **Source:** `global_input.md` issue 3.

### O10. No snowpack on the ground
- **Kind:** science issue · **Target:** after 8 · **Decides:** M
- **What:** snowfall is intercepted by the canopy but snow reaching the ground is not
  accumulated: no change of ground albedo, ground heat flux or LAI burial.
- **Now:** no snow model (towers: snowfall 0, as Fortran).
- **Source:** `global_input.md` issue 4.

### O11. Root biomass 500 g m⁻² everywhere; canopy bottom height unused (grid runs)
- **Kind:** science issue · **Target:** after 8 · **Decides:** M
- **What:** satellite phenology has no fine-root biomass, so every grid patch uses
  500 g m⁻² (CLM-ml's forest-tower value). `MONTHLY_HEIGHT_BOT` is read but the leaf
  profile still runs from the ground to the top.
- **Now:** `domain.grid.root_biomass` (500); `Patches.canopy_bottom` kept, unused.
- **Source:** `global_input.md` issue 5, G13, G17.

### O12. Initial state from monthly means, not a spun-up restart (grid runs)
- **Kind:** science issue · **Target:** after 8 · **Decides:** M
- **What:** grid runs start from interpolated monthly `TSOI`/`H2OSOI`; expect a soil
  temperature drift over the first weeks.
- **Now:** as G11.
- **Next:** spin-up procedure (restarts exist since 7.5) or a CLM restart reader.
- **Source:** `global_input.md` issue 6, G11.

### O13. Forcing height fixed at setup; CLM6 roughness is Meier2022
- **Kind:** science issue · **Target:** after 8 · **Decides:** M
- **What:** CLM6 computes `z0m` and displacement from LAI each step (so its forcing height
  varies in time); this model fixes the forcing height at setup with the ZengWang2007 rule
  (decision 2 of `global_input.md`).
- **Source:** `global_input.md` issue 1, G14.

### O14. Very short canopies not filtered
- **Kind:** science issue · **Target:** after 8 · **Decides:** M
- **What:** grass and crop tops go down to 5 mm in the surface data, still given 10 + 10
  layers.
- **Source:** `global_input.md` issue 9.

### O15. Wet needleleaf canopies
- **Kind:** science issue (observation) · **Target:** after 8 · **Decides:** M
- **What:** CLM6's needleleaf `dleaf` is 0.65 mm (CLM5: 4 cm); the leaf boundary-layer
  conductance grows as `dleaf` shrinks. South-eastern US needleleaf patches have January
  latent heat 16 W m⁻² (others 6.9), with wet days of 150-226 W m⁻² fed by negative
  sensible heat. To compare with CLM. Also the patch most sensitive in O53.
- **Source:** `global_input.md` issue 15, issue 10.

### O16. CO₂ profile reset covers the wrong levels (no effect today)
- **Kind:** science issue (latent) · **Target:** after 8 · **Decides:** M
- **What:** Fortran resets `cair` to the reference CO₂ in canopy layers 1…ncan; the
  translation reset levels 0…ncan−1. No effect: `cair` is never computed (no CO₂ profile).
- **Now:** kept through Phase 7 (0-based: layers 0…ncan−2, the top layer not reset;
  `src/clm_ml/physics/canopy.py:843`); fix with any change that makes `cair` vary.
- **Source:** `numerics_and_regression.md` §4.5.

## 2. Other maintainer decisions pending

### O17. Repository organization: where the real configuration files live
- **Closed 2026-11-01:** `configs/` with complete, generated model.yaml files (decision 90);
  params files complete, no `extends`, `run.yaml` must name one (decision 89). Kept below
  as it was found.
- **Kind:** maintainer decision · **Target:** 8a · **Decides:** M
- **What:** a user cannot find "the" CHATS7 configuration. Today:
  - **model.yaml:** no file in the repo is used by the model; the defaults are in the code
    (`clm_ml/config.py` dataclasses; `model: null` in run.yaml means all defaults). Example
    files are in `docs/design/examples/` (`model.yaml`, `model_grid.yaml`), and `clm-ml
    convert` writes one (only `canopy_layers.min_plant_area_per_layer`).
  - **params.yaml:** built-in sets in the package, `src/clm_ml/data/params/`
    (`clm5_defaults.yaml`, `chats.yaml` = CHATS overrides via `extends`, `clm6.yaml`); an
    excerpt and a CHATS example in `docs/design/examples/`.
  - **towers.yaml:** built-in `src/clm_ml/data/towers.yaml` (CHATS7 only); example in
    `docs/design/examples/towers_example.yaml`.
  - **run.yaml:** none committed for CHATS7 except `docs/design/examples/run_chats7_1day.yaml`;
    the tests and CI create them with `clm-ml convert` from `data/namelists/`.
  - **Input data:** `data/input_files/` (Fortran's layout, `clm5_0/` kept by C35); plan 7b
    wanted `examples/data/` (not done).
- **Next:** decide the layout, e.g. a top-level `configs/` (or `examples/`) with validated,
  commented CHATS7 run/model files that CI runs `clm-ml check` on (jax-gcm's "named config
  files"), the package data staying the built-in defaults; design examples kept as history
  or removed.
- **Source:** maintainer question (2026-11-01); plan §4, plan 7b; Phase 7 "Not done"; C33.

### O18. Norman and well-mixed code in git history only
- **Kind:** maintainer decision · **Target:** 8a · **Decides:** M
- **What:** D8 said WellMixed, dataset profiles and Norman solar radiation stay "kept,
  registered as unsupported". The options are registered and refused with a clear error,
  but their translations went with the containers in 7.4f (git history, commit `1f1b651`).
- **Next:** accept (code from history when the variants are redone, O56) or restore as
  reference files that do not run.
- **Source:** C34; D8; C13, C17.

### O19. Options that only ever refuse
- **Kind:** maintainer decision · **Target:** 8a · **Decides:** M
- **What:** D9 kept `gs_solver` (`wue_solver: brent`) and `mlcan_to_clm`. Both now refuse
  their non-default value (`run.py:551`; `driver.py:360`); `mlcan_to_clm` is not a model.yaml
  key, only an `MLOptions` field (`options.py:43`).
- **Next:** keep (documented as planned variants) or remove until implemented.
- **Source:** D9; C16; C21; `global_input.md` issue 18.

### O20. `nitrogen_decay_kn` is static, not differentiable
- **Kind:** maintainer decision · **Target:** 8a · **Decides:** M
- **What:** the canopy nitrogen decay coefficient (`kn_val`) is a model.yaml option, so it
  cannot be calibrated with gradients, against the design rule "every numeric tunable must
  be differentiable".
- **Next:** decide whether to move it to params.yaml (per PFT) as a differentiable
  parameter (interface change; bit for bit for the default).
- **Source:** C18; `config.py:34`; CLAUDE.md design guidance.

### O21. Public API
- **Kind:** maintainer decision · **Target:** 8a · **Decides:** M
- **What:** `clm_ml/__init__.py` exports nothing; users must import internal modules
  (`clm_ml.run.load_run`, `clm_ml.model.Model`, `clm_ml.config.ModelConfig`,
  `clm_ml.params.Params`). Documentation fixes what is public.
- **Next:** choose the public names (compare `docs/design/api_sketch.py`), re-export them,
  test the imports.
- **Source:** Phase 8 survey; plan §5.

### O22. Review of Phase 7 behaviour changes and re-recorded references
- **Kind:** maintainer decision · **Target:** 8a · **Decides:** M
- **What:** the maintainer's review of Phase 7 is pending: round-off-only reference
  re-recordings (C10, C14, C16, C17, C19-C21, C28); behaviour changes beyond round-off
  (chunks counted from step 0, restart format 6 netCDF, `ModelError` instead of
  `SystemExit`, health check by layer masks: C28, C32; CLM history validation and soil
  colour 0 refused: C31; raw validation fields renamed: C33; `clm-ml-offline` retired: C30);
  28 + 27 diagnostics no longer computed (`phase7_removed_diagnostics.md`; C19, C20).
- **Next:** accept, or name what to revisit.
- **Source:** `phase7_progress.md` "Next" and "Open for the maintainer".

### O23. Authorship, citation and canonical repository
- **Kind:** maintainer decision · **Target:** 8a · **Decides:** M
- **What:** README badges, codecov, the citation ("Lahlou, Aya, 2024") and CONTRIBUTING's
  clone URL point to `AyaLahlou/clm-ml-jax`; the remote is `linniahawkins/clm-ml-jax`.
- **Next:** decide authors, canonical repo, licence holders (LICENSE is BSD-3).
- **Source:** Phase 8 survey.

### O24. Release logistics
- **Kind:** maintainer decision · **Target:** 8a (merge), 8b (release) · **Decides:** M
- **What:** `origin/main` (2026-05-26) is 156 commits behind `phase7/restructure`; no phase
  branch has been merged (plan risk "Every phase is merged on its own" not followed;
  standing rule: no PRs). Version is 0.1.0.
- **Next:** decide how `main` is updated (local merge and push, authorized explicitly), the
  1.0.0 tag, PyPI or GitHub only.
- **Source:** Phase 8 survey; plan §9; plan Phase 8.

## 3. Bugs and robustness

### O25. XLA CPU gradient crash
- **Kind:** bug (upstream, JAX 0.9.2) · **Target:** known; recheck at each JAX upgrade · **Decides:** I
- **What:** running compiled CPU gradients segfaults in LLVM-optimized code. Workaround:
  CPU gradient tests with LLVM optimization off (`tests/gradient_tier.py`); GPU gradients
  pass.
- **Next:** report upstream / retest with newer JAX; document for users computing CPU
  gradients.
- **Source:** `global_input.md` issue 16; D12; CLAUDE.md gotchas.

### O26. XLA CPU compile crash
- **Kind:** bug (upstream, JAX 0.9.2) · **Target:** known; recheck at JAX upgrade · **Decides:** I
- **What:** large batched programs segfaulted while compiling (LLVM JIT linker recursion).
  Workaround: `clm_ml/__init__.py` sets `--xla_cpu_parallel_codegen_split_count=1`.
- **Source:** CLAUDE.md gotchas; `src/clm_ml/__init__.py`.

### O27. GPU compile-to-compile variation
- **Kind:** bug (reproducibility) · **Target:** after 8 (GPU work) · **Decides:** I
- **What:** on the GPU about one fresh compile in five gives a program whose results differ
  by ~1e-11 (within the noise twin). Diagnosis unfinished (setup programs get different
  cache keys in each process). Bit for bit is a CPU-only test.
- **Source:** C4 ("Open"); CLAUDE.md gotchas.

### O28. A failing patch still enters its chunk's gridcell sums
- **Kind:** robustness · **Target:** 8a · **Decides:** M
- **What:** with `on_patch_error: exclude`, a patch failing a check with finite values still
  enters the gridcell sums (and parameter gradients) for the chunk in which it fails (10
  days by default); only its patch output is filled.
- **Next:** decide whether that is acceptable (document) or fix (needs per-step exclusion).
- **Source:** Phase 6 review R4; decision 25.

### O29. Padding slots combine one gridcell's weather with another patch's data
- **Kind:** robustness · **Target:** 8a (verify) · **Decides:** I
- **What:** padding slots (for equal device shares) on real gridcells combine one
  gridcell's meteorology with another patch's site data, and stop mode checked them as if
  real. Phase 7's health check by layer masks (C28) may have changed this: verify.
- **Source:** Phase 6 review R5.

### O30. Smaller Phase 6 review items
- **Kind:** robustness · **Target:** 8a (verify each) · **Decides:** I
- **What:** per-variable exclusion in `gridcell_sum`; unmasked first-step gridcell sums
  (the first step is an ordinary step since C28: likely closed); `Model.weights()` in slot
  order with devices > 1; a run starting in the first forcing year cannot start; 184
  gridcells use the nearest point where ESMF may interpolate; R7 (tower setup not bit for
  bit when the porosity cap binds; setup rewritten in 7.4: likely closed).
- **Source:** Phase 6 review R7, R11.

### O31. netCDF read on a background thread while writing on the main thread
- **Kind:** robustness · **Target:** 8a · **Decides:** I
- **What:** the forcing reader thread reads netCDF while the output writers write netCDF on
  the main thread; HDF5 builds without thread safety could crash. As since Phase 4.
- **Source:** C31; Phase 7 "Not done".

### O32. `Model.chunks()` start at step 1, runs chunk from step 0
- **Kind:** robustness (API consistency) · **Target:** 8a or document in 8b · **Decides:** M
- **What:** `Model` runs the first step at setup, so its chunks start at step 1; `Run`
  chunks from step 0. A user comparing the two gets an offset.
- **Source:** C31; CLAUDE.md gotchas.

### O33. The `euler` tableau has `b = [0]`
- **Kind:** robustness · **Target:** 8a · **Decides:** I
- **What:** `runge_kutta_update` with the Euler tableau would reset the state; the model is
  right only because `canopy._stage` returns before calling it. A trap for future code.
- **Next:** a correct tableau or an assertion.
- **Source:** C23 (a).

## 4. Housekeeping and repository organization (8a)

### O34. `.gitignore` ignores `docs/`
- **What:** new documentation files are silently left out of commits (tracked files are
  added with `git add -u docs`/`-f`). It also lists directories that do not exist here
  (`CLM-ml_v1/`, `clubb_ML/`, `jax-agents/`, `Paper/`, `tests/integration/`, ...).
- **Now:** the `docs/` rule removed (2026-11-01, with this document).
- **Next:** prune the rest.
- **Source:** Phase 8 survey; `phase7_progress.md` practical notes.

### O35. Empty `.gitmodules`; CONTRIBUTING describes a submodule
- **What:** CONTRIBUTING says to clone `--recurse-submodules` (Fortran in `CLM-ml_v1/`);
  `.gitmodules` is empty. Fortran sources are outside the repo.
- **Next:** delete `.gitmodules`; say where the Fortran references are.
- **Source:** Phase 8 survey.

### O36. `.github/copilot-instructions.md` is stale
- **What:** 491 lines on deleted packages (`clm_src_main`, `clm_varpar`, ...).
- **Next:** delete (CLAUDE.md and CONTRIBUTING cover it) or rewrite.
- **Source:** Phase 8 survey.

### O37. README out of date
- **What:** badges and citation (O23); features list "Namelist-driven configuration"; a
  generic fork/PR contributing section; no mention of grid runs, gradients, restarts.
- **Next:** short rewrite now; full guide in 8b.
- **Source:** Phase 8 survey.

### O38. CONTRIBUTING out of date
- **What:** clone URL, submodule, PR workflow; "Adding a Physics Module" should become "add
  a process variant" (8b, O50).
- **Source:** Phase 8 survey; plan Phase 8.

### O39. `data/namelists/README.md` describes `clm-ml-offline`
- **What:** 121 lines on the retired executable (C30).
- **Next:** rewrite: the namelists are inputs of `clm-ml convert` and the Fortran
  comparison.
- **Source:** C30.

### O40. Namelists of unclear status
- **What:** `nl.US-UMB.test` points to files in the maintainer's scratch (unusable by
  others; US-UMB is not a built-in tower, decision §8.9); `nl.CHATS7.debug` and
  `nl.CHATS7.test3` have no documented purpose.
- **Next:** keep, fix or delete each.
- **Source:** Phase 8 survey.

### O41. Example files not checked
- **Closed 2026-11-01:** `docs/design/examples/` removed (the global example was
  hypothetical); `configs/` replaces it and `tests/clm_ml/test_configs.py` (run in CI) checks
  every model.yaml is complete and current and every run.yaml loads.
- **What:** `docs/design/examples/run_global_example.yaml` fails `clm-ml check` (`domain:
  unknown key 'surface_dataset'`). The plan's risk table says `clm-ml check` runs in CI on
  every example; it does not.
- **Next:** fix or retire the example; CI check of every shipped config (with O17).
- **Source:** Phase 8 survey; plan §9.

### O42. Stale design documents
- **What:** plan header "Status: v4. Phases 0–2 done"; plan §8 rows 4-5 still "Undecided,
  revisit at Phase 7" (row 4 settled by C30; row 5 by Phase 7's `physics/soil.py`);
  `api_sketch.py` (proposed API, pre-implementation); numerics §4 titled "not changed in
  Phase 3"; `timing.md` holds only the Phase 0 baseline (later results only in
  `timing/*.jsonl`).
- **Next:** update or mark as historical.
- **Source:** Phase 8 survey; plan §8.

### O43. CI lint skips `timing/`
- **What:** CI runs `black`/`ruff` on `src/ tests/`; CLAUDE.md's lint command includes
  `timing`.
- **Source:** `.github/workflows/ci.yml` l. 29-32.

### O44. One open-items list and one decision log
- **What:** decisions are spread over plan §0/§8, D1-D12, C0-C35, G1-G20 and the progress
  logs; open items over five files.
- **Next:** this document (open items) and a companion decision log; the old files keep
  their history.
- **Source:** maintainer request (2026-11-01).

### O65. `numerics.checkpoint_steps` does nothing
- **What:** the model.yaml key exists (`src/clm_ml/config.py:63`) but the driver always
  checkpoints the step (module constant `CHECKPOINT_STEP = True`, `src/clm_ml/driver.py:40`).
  A user setting it to false gets no effect and no warning.
- **Next:** remove the key, or make it work (it trades gradient memory for recomputation).
- **Source:** model map (2026-11-01).

### O66. PFT parameters used only at setup
- **What:** `root_profile_beta`, `c3_photosynthesis` and the PFT default canopy height and
  leaf/stem profiles are read once at setup (NumPy) into `PatchConstants`. Changing them in
  the parameters passed to `Model.run_chunk` has no effect, so a calibration of, say, the
  root profile would silently get zero gradient.
- **Next:** decide: document them as setup parameters, refuse them in calibration, or move
  the root fractions into the differentiable path.
- **Source:** model map (2026-11-01).

### O67. PFT optics in the old 1-based layout
- **What:** the spectral fields of `CanopyParams` (leaf/stem reflectance and transmittance)
  are `[unused, vis, nir]`; everything else is 0-based (decision on 0-based indexing).
- **Next:** convert (a refactor, bit for bit).
- **Source:** model map (2026-11-01).

### O68. Stale docstrings and comments
- **What:** the `model.py` header still shows "calendar-month chunks (decision 17)"
  (superseded by fixed-length chunks) and mentions several soil structures (C35);
  `clm_ml/__init__.py` says the public API "is re-exported here as it is built" (nothing
  is); CLAUDE.md said "Regression tolerance is `rtol = 1e-6`" (corrected 2026-11-01: the
  policy is bit for bit plus the noise twin).
- **Next:** correct them (with O42).
- **Source:** model map; decision register (2026-11-01).

### O69. Output accumulation differs from the plan's description
- **What:** the plan describes a nested scan (outer loop over output records, inner over
  steps); the code is one scan over steps, each step adding into a record slot chosen on
  the host. Memory still scales with output records, as intended.
- **Next:** update the plan text (no code change).
- **Source:** model map (2026-11-01).

### O70. Implement experiment folders
- **Closed 2026-11-01:** implemented (decision 91).
- **What:** decision 91: four complete files per experiment folder (`run`, `domain`, `model`,
  `params`), strict loading, `clm-ml new`/`upgrade`, presets in the package, `configs/`
  regenerated as folders, configuration copied into each run's output.
- **Next:** the six steps of `experiment_folders.md` §12 (questions answered, §13).
- **Source:** maintainer, 2026-11-01.

## 5. Documentation (8b)

### O45. Getting-started guide
YAML first: install, convert a namelist, run one CHATS7 day, read the netCDF with xarray,
first plot. **Source:** plan Phase 8.

### O46. Reference of every YAML key, generated from the code
The settings' explanations are code comments; move them (with units and Fortran names) into
the dataclass fields' metadata, generate the reference, test it stays current (8a code
part, bit for bit). **Source:** plan Phase 8; plan §9 ("key reference generated").

### O47. Examples
Gradient calibration (twin experiment or CHATS observations: maintainer's choice; data
files needed for the latter), ensemble (`vmap` over parameters), multi-tower/multi-patch, a
grid run; short versions run in CI. **Source:** plan Phase 8.

### O48. Page for Fortran CLM-ml users
Namelist keys and option codes to YAML, generated from the converter's tables
(`clm_ml/io/namelist.py`, `config.to_fortran_codes`). **Source:** plan Phase 8.

### O49. API reference from docstrings
Plus a pass over docstrings still worded in legacy terms. Depends on O21. **Source:** plan
Phase 8.

### O50. CONTRIBUTING
"Add a process variant" replaces "mirror the Fortran module"; testing tiers; regression
policy (bit for bit, noise twin). **Source:** plan Phase 8.

### O51. Output caveats for users
`soil_water_stress` (`btran`) is 1 everywhere with the WUE stomatal scheme (water stress
acts through leaf water potential, as Fortran); layer outputs are NaN above a patch's top
layer; `stem_area_added` (decision 30). **Source:** `global_input.md` regional test; CLAUDE.md
gotchas; G20.

### O52. Release 1.0.0
Version, CHANGELOG, `CITATION.cff` (O23), tag (O24). **Source:** plan Phase 8.

## 6. Deferred to after Phase 8

### O53. Round-off amplification in real 1° patches
A 1e-15 change of LAI moves latent heat by up to ~8 W m⁻² after sunrise in some patches
(largest: needleleaf, CLM6 `dleaf`): a discrete switch somewhere. Limits grid regression
tests to the night; will matter for gradients. Locate it routine by routine. **Source:**
`global_input.md` issue 10; plan "After Phase 8: science issues".

### O54. Bare-ground patches
No bare-ground path; the minimum plant area keeps such patches as sparse canopies (small
bias). Options written up (preferred: a canopy that behaves correctly as plant area → 0).
New code. **Source:** `global_input.md` issue 13; decision 29; plan "After Phase 8".

### O55. GPU performance
Kernel count per step (Phase 6 step 1c), C3/C4 groups traced twice in mixed batches
(decision 22, D10), the decision-33 GPU check, a full-year global run; a chunk is 11 %
slower than in Phase 6b (program 2.32 GB → 4 MB, compile 840 s → 61 s in Phase 7).
**Source:** plan "After Phase 8: GPU performance"; D10; `phase7_progress.md` timing.

### O56. Unsupported variants
Well-mixed flux profile and dataset profiles (fail on traced layer counts), Norman solar
radiation, Brent's method for the WUE optimization: registered, refused; make them run
compiled and validate against Fortran. **Source:** plan "After Phase 8: unsupported
variants"; D8; C13, C16, C17, C34; issue 18.

### O57. Option B: shared soil columns
`soil.columns: shared` (refused today, `config.py:110`); a separate science change,
validated at mixed-vegetation towers. **Source:** plan §6.7, "After Phase 8: option B";
decision 5.

### O58. Orbital parameters of the start year only
CLM's historical runs update them each year (~0.02° of declination over 150 years).
**Source:** decision 34; D4.

### O59. Forcing preprocessed once to model-step files
For repeated runs over the same years (calibration, ensembles, spin-up); saves CPU time
only. **Source:** decision 36.

### O60. Reduced-precision option
float32 for consumer GPUs (float64 30-60× slower there); never done; needs its own
validation. **Source:** plan §8 row 7.

## 7. Known model limitations (not to be fixed without discussion)

### O61. Strongly stable conditions
Near-ground air several degrees too warm: first-order closure cannot represent non-local
within-canopy mixing. **Source:** CLAUDE.md; Bonan et al. (2026).

### O62. Daytime latent heat biased high
By about 70 W m⁻² at CHATS. **Source:** CLAUDE.md; Bonan et al. (2026).

### O63. Soil heat flux biased high
The point measurement is likely unrepresentative. **Source:** CLAUDE.md.

### O64. Soil moisture prescribed, not predicted
**Source:** CLAUDE.md; plan decision 14.

---

## Recently closed (for reference, not open)

- Soil texture classes take their own values (O1, issue 21; decision 92; 2026-11-01).

- Experiment folders implemented: four complete files, `clm-ml new`/`upgrade`, presets,
  configuration copy with each run's output (O70; decision 91; 2026-11-01).

- Configuration files: `configs/` with complete, generated model.yaml files; complete params
  files, `run.yaml` must name one (O17, O41; decisions 89, 90; 2026-11-01).

- One open-items list (this document), one decision register (`decisions.md`) and a map of
  the model (`model_overview.md`) (O44; maintainer request, 2026-11-01).

- CLM4.5 soil layer structure dropped; CLM soil only (decision 37, C35; also closes Phase 6
  review R8).
- Stray `lat.npy` deleted (C35).
- psihat tables an input of the compiled chunk (C22).
- Calendar: `time.calendar: gregorian | noleap` (decision 34; issue 2, review R2).
- Meteorology memory: forcing stays prepared on the host (decision 36; issue 7).
- Canopy energy balance when the air crosses freezing (decision 32; issue 12, except O6).
- Diffuse transmittance above 1 in thin layers (decision 30; issue 14).
- `wue_solver: brent` refused instead of silently ignored (C16; issue 18; see O19, O56).
- Phase 6 review: R1 (refused), R3, R6, R9 fixed (2026-10-30).
- Template compiled into programs as constants (C23 b; fixed in 7.4).
- Probably closed, to confirm in 8a: issue 8 (`forc_hgt_u_patch` written by the step: the
  containers were deleted in 7.4f; no `forc_hgt_u` in the step code); issue 11 (CPU cache
  loader crash of the batched first-step program: that program no longer exists since C28,
  and `without_persistent_cache_on_cpu` is gone from the code).
