# Global input for Phase 6b (1°)

Inventory of the input files provided by the maintainer (plan Phase 6b step 1, 2026-10-29).
Target resolution: **1° (0.9×1.25, the CESM f09 grid)**, to match the surface data and the soil
moisture.

## Files

| Input | Path | Grid | Time |
|---|---|---|---|
| Surface data | `/glade/campaign/cesm/cesmdata/inputdata/lnd/clm2/surfdata_esmf/ctsm5.4.0/surfdata_0.9x1.25_hist_2000_16pfts_c251022.nc` | 288 × 192 (0.9×1.25) | year-2000 land cover; 12 monthly LAI/SAI/heights |
| Forcing (CRUJRA v4.10) | `/glade/campaign/cgd/tss/projects/TRENDY2026/inputs/three_stream/clmforc.CRUJRAv4.10_0.5x0.5.{TPQWL,Solr_Diffuse,Prec}.YYYY.nc` | 720 × 360 (0.5°) | 1901-2025, 6-hourly, `noleap` |
| Soil moisture (CLM run) | `/glade/derecho/scratch/slevis/archive/wetlands.n04.ctsm5.4.044_f09BgcCrop_flat_174_Hist/lnd/hist/*.clm2.h0a.YYYY-MM.nc` | 288 × 192 (f09) | 1850-01 to 2023-12, monthly means |
| Parameters (CTSM 6.0) | `/glade/work/wwieder/modified_inputs/ctsm60_params.genericCropFix.ch4cal_d.c260410.nc` | 79 PFTs | — |

Also in the CRUJRA directory: `domain.crujra.v2.5_0.5x0.5.c240722.nc` and ESMF meshes (land mask
of the forcing grid). The history directory has a `mislabeled_1915to1944/` subdirectory (720
files): the files for 1915-1944 there are mislabeled; avoid those years until it is clear which
set is right.

## Surface data

- **Patches.** PCT_NATVEG × PCT_NAT_PFT (natural PFTs 1-14; PFT 0, bare ground, skipped:
  decision 29) and PCT_CROP × PCT_CFT (15 C3 crop, 16 C3 irrigated): **78 246 vegetated
  patches on 15 896 gridcells** (mean 4.9, at most 14 per gridcell); 53 930 patches with at
  least 1 % of their gridcell, 39 321 with at least 5 % (`domain.min_patch_weight`). Either
  fits on one 4-GPU node (65 536 patches ran in Phase 6 step 6).
- **By PFT** (patches): NET temperate 4 736, NET boreal 4 661, NDT boreal 3 313, BET tropical
  4 665, BET temperate 1 891, BDT tropical 4 232, BDT temperate 4 539, BDT boreal 4 666, BES
  temperate 240, BDS temperate 4 035, BDS boreal 4 170, C3 arctic grass 5 080, C3 grass 9 669,
  C4 grass 7 513, C3 crop 10 462, C3 irrigated 4 374. C4 grass is the only C4 PFT.
- **Canopy heights.** MONTHLY_HEIGHT_TOP/BOT per PFT and gridcell **do not vary by month** in
  this file (so the fixed layer structure can use them directly: plan Phase 6b step 4). Trees:
  top 1.2-30 m, bottom 0.6-17 m; grasses: top 0.04-0.5 m, bottom ~0.
- **Soil.** PCT_SAND, PCT_CLAY, ORGANIC on 10 soil levels (CLM-ml's setup takes one value per
  column: which level(s) to use is a decision below), SOIL_COLOR, zbedrock (m).
- **LAI/SAI.** MONTHLY_LAI, MONTHLY_SAI per PFT (12 months, climatology).
- Land gridcells: 22 648 with LANDFRAC_PFT > 0; 16 943 with natural vegetation.

## Forcing (CRUJRA)

- TPQWL: TBOT (K), PSRF (Pa), QBOT (kg/kg), WIND (m/s, speed), FLDS (W/m²); time stamps at
  the middle of each 6-hour interval (0.125, 0.375, … days).
- Solr_Diffuse: FSDS_DIRECT, FSDS_DIFFUSE (W/m²); time stamps at the start of each interval
  (0, 0.25, … days).
- Prec: PRECTmms (mm/s); mid-interval time stamps.
- No forcing height and no wind direction (the tower forcing has a height and u, v).

## Soil moisture and soil temperature (CLM history, h0a)

- H2OSOI (mm³/mm³, 20 soil levels), SOILLIQ, SOILICE (kg/m²), TSOI (K, 25 ground levels) on
  the f09 lat/lon grid; averages over the natural vegetated and crop land units; monthly
  means (`noleap`, days since 1850-01-01).
- 21 244 land gridcells in the history (the surface data have 22 648: the masks differ).
- TLAI, TSAI, HTOP are also there (gridcell means, not per PFT).

## Parameters (CTSM 6.0 file)

- 79 PFTs (16 used here), 158 per-PFT variables: e.g. slatop, dleaf, xl, rholvis/rholnir,
  taulvis/taulnir, rhosvis, c3psn, rootprof_beta (2 variants), medlynslope, z0mr.
- **Not in the file:** the CLM-ml-specific parameters (MLpftcon: `iota` for the WUE stomatal
  optimization, plant hydraulics `gplant`, `capac`, root radius/density/resistivity,
  `vcmaxpft`, `emleaf`, `clump_fac`, …); these come from the CLM-ml defaults
  (`clm5_defaults.yaml`). Decision 27 (CLM5 defaults) applies to the CLM part.

## Decisions (maintainer, 2026-10-29)

1. **Forcing grid → 1°: what CLM does.** CDEPS `stream_definition_datm.xml` (CLMCRUJRA2024
   streams): **bilinear** mapping for all three streams; time interpolation **coszen** for
   shortwave, **nearest** for precipitation, **linear** for TPQW; offset 0. (The existing
   forcing rules match: `coszen`, `constant` within the record's interval, `linear`.)
2. **Forcing height: CLM's rule, per patch:** `zref = 30 m + (z0mr + displar)·htop` with the
   CLM6 `z0mr` and `displar` of the patch's PFT (CLM's data atmosphere sets 30 m, `Sa_z = 30`,
   and CLM measures it from the zero-plane displacement plus roughness length,
   `forc_hgt_u_patch = forc_hgt_u + z0m + displa`). Always above the canopy top.
3. **Shortwave: CRUJRA's direct/diffuse split as given.**
4. **Soil texture: the top level** of PCT_SAND, PCT_CLAY, ORGANIC.
5. **Soil moisture: H2OSOI from the h0a files; the domain is the CLM run's land mask.** Only
   gridcells with `landmask == 1` in the history (and vegetation in the surface data) are in
   the domain: **14 346 gridcells, 72 575 vegetated patches** (49 286 with at least 1 % of the
   gridcell; 6 478 C4). The 1 550 vegetated surface-data gridcells outside the run's land
   domain are skipped. (Checked on the top level of H2OSOI: every gridcell of the domain has
   it.)
6. **Parameters: CLM6** (the CTSM 6.0 file) for every PFT, except CHATS (which keeps its
   overrides); CLM-ml-specific parameters from `clm5_defaults.yaml`.
7. **Layer counts: constant, 10 within and 10 above the canopy.**
8. **Year: 2000** for the forcing (CRUJRA 2000), the soil moisture (h0a 2000-01 to 2000-12)
   and the initial soil temperature (TSOI); the surface data are the year-2000 land cover.

## Implementation decisions (Phase 6b steps 2-5, copying CLM)

Where a choice came up while implementing, CLM's behaviour (CTSM 5.4 / CDEPS, source in
`/glade/work/linnia/ctsm5.4.004`) was copied. Code: `clm_ml/grid.py` (domain),
`clm_ml/grid_forcing.py` (forcing and land data), `clm_ml/legacy/grid_setup.py` (setup and
chunk inputs), `clm_ml/params_convert.py` (`clm-ml params`: a CLM parameter file to params.yaml).

| # | Topic | What is done | CLM source |
|---|---|---|---|
| G1 | Time of the meteorology | The land step `[t, t + Δt]` uses the data atmosphere evaluated at `t` (the step start): CLM's run sequence runs LND before ATM in each coupling interval, and datm computes the next interval's data | `nuopc.runseq`, `atm_comp_nuopc.F90` (`nextTime`) |
| G2 | Time interpolation | Per stream in its own calendar, file stamps as given: TPQW **linear** between the bracketing stamps; Precip **nearest** (tie: earlier); Solar **coszen** = lower-bound record × `cosz(t)/⟨cosz⟩` (instantaneous `cosz`, floor 0.001; `⟨cosz⟩` mean over the record interval sampled every model step from its stamp; 0 when the sun is down); model's orbital parameters (run year) | `dshr_strdata_mod.F90`, `dshr_tinterp_mod.F90` |
| G3 | Calendar | Model calendar stays Gregorian (CLM-ml v2 driver); CRUJRA and the history files are `noleap`: a model 29 February uses 28 February data (CDEPS rule for a Gregorian model with a no-leap stream) | `dshr_strdata_advance` (`ymdmod`) |
| G4 | 0.5° → 1° mapping | Bilinear between the four surrounding 0.5° points; nearest valid point when a corner is outside the CRUJRA land mask (ESMF bilinear with source mask + nearest-source extrapolation). Weights planar in lat/lon (ESMF: on the sphere; second-order difference) | `dshr_strdata_mod.F90` (`ESMF_REGRIDMETHOD_BILINEAR`, `EXTRAPMETHOD_NEAREST_STOD`) |
| G5 | datm derivations | `TBOT ≥ 180 K`; `QBOT` capped at saturation (`datm_eSat`, ice below 0 °C); `PSRF = 0` → 101325 Pa; direct and diffuse shortwave each split half visible / half NIR; forcing height 30 m | `datm_datamode_clmncep_mod.F90` |
| G6 | Rain and snow | datm ramp `frac_rain = (T − 273.15)/2` in [0, 1], then CLM's repartition with its defaults (all snow at 0 °C, all rain at 2 °C: the same ramp). Snowfall reaches CLM-ml's canopy interception (`forc_snow`; towers: 0, as Fortran) | `shr_precip_mod.F90`, `atm2lndMod.F90` (`partition_precip`) |
| G7 | Wind | datm sets `u = v = WIND/√2`; the model only uses `√(u² + v²)`, so `u = WIND, v = 0` (same speed, no round-off) | `datm_datamode_clmncep_mod.F90` |
| G8 | CO2 | TRENDY `fco2_datm_global_simyr_1700-2025_TRENDY_c260722.nc` (annual global, mid-year stamps), linear in time at the step start (datm `co2tseries` uses linear); or a constant (`co2: {ppm: ...}`). O2 = 0.209 × pressure | `stream_definition_datm.xml` (`co2tseries`) |
| G9 | LAI, SAI | Surface-data `MONTHLY_LAI/SAI` per PFT and gridcell with CLM's satellite-phenology weights (`interpMonthlyVeg`: daily, at the end of the step plus one step, no-leap month lengths), interpolated inside the compiled chunk; `elai = tlai`, `esai = tsai` (no snow burial: no snow model), set to 0 below 0.05 | `SatellitePhenologyMod.F90` |
| G10 | Soil water | `H2OSOI` (20 levels) from the monthly h0a files, linear between the file stamps (mid-month in CTSM 5.4 history) at the step end (`get_curr_date`); interpolated inside the compiled chunk. The CLM run's 25 ground / 20 soil levels equal CLM-ml's `clm5_0` structure (checked) | `SoilMoistureStreamMod.F90` (`soilm_tintalgo = linear`) |
| G11 | Initial soil temperature and water | `TSOI`/`H2OSOI` with the G10 interpolation at the end of the first step (what a tower setup reads from its CLM history slice), then CLM-ml's `SoilInit` (capped at porosity). CLM would start from a spun-up restart | — |
| G12 | Canopy height | `MONTHLY_HEIGHT_TOP` with the G9 weights at the first step; constant in this file, and CLM-ml's layers are fixed at setup | `SatellitePhenologyMod.F90` |
| G13 | Canopy bottom (step 5) | `MONTHLY_HEIGHT_BOT` is read and kept (`Patches.canopy_bottom`) but **not used**: CLM-ml v2's leaf-area profile is a beta distribution from the ground to the top, and CLM uses the bottom height only for snow burial (no snow model here). Using it (e.g. the profile between bottom and top) is a science change for the maintainer | `SatellitePhenologyMod.F90` (`fb`) |
| G14 | Forcing height | Per patch `30 m + (z0mr + displar)·htop` with each PFT's `z0m_to_height_ratio`, `displacement_to_height_ratio` in params.yaml (CLM's `z0mr`, `displar`; built-in `clm6`) (CLM's `ZengWang2007` roughness, decision 2) | `FrictionVelocityMod.F90`, `BiogeophysPreFluxCalcsMod.F90` |
| G15 | Parameters | One format: params.yaml. A CLM parameter file is converted once with `clm-ml params FILE.nc -o FILE.yaml` (`clm_ml/params_convert.py`), which writes a complete table; built-in `clm6` (`data/params/clm6.yaml`) is the CTSM 6.0 file converted: `c3psn`, `rootprof_beta` (variant 1 = water, CLM's `rooting_profile_varindex_water`), `dleaf`, `xl`, `slatop`, `rhol/taul/rhos/taus` (vis, nir), `medlynslope`, `medlynintercept` (µmol → mol), `mbbopt`, `z0mr`, `displar` from the file; everything else (vcmax25, ι, hydraulics, `g0_ball_berry`: `bbbopt` is not in the file) from `clm5_defaults` | `pftconMod.F90` |
| G16 | Organic matter | Capped at CLM-ml's `organic_max` (130 kg m⁻³): CLM caps the organic fraction at 1, CLM-ml's formula has no cap, and 4 021 patches' soils exceed it (up to 318) | `SoilStateInitTimeConstMod.F90` |
| G17 | Root biomass | Not in CLM (satellite phenology has no fine-root biomass): 500 g m⁻² for every patch (`domain.grid.root_biomass`), the value of every CLM-ml forest tower | — |
| G18 | `min_patch_weight` | Applied to the fraction of the gridcell's *vegetated* area, then renormalized (towers: the weights as given, which sum to 1); a gridcell keeps at least its largest patch | — |
| G20 | Minimum plant area per layer (decision 30, replaces the earlier 0.05 of stems for patches without plant area) | Every leafy layer keeps at least `min_layer_plant_area` = 0.01 m² m⁻² at every step; the missing part is added as **stem** area (LAI is never changed, so no photosynthesis or transpiration is added), and the canopy's SAI includes it (output `stem_area_added`). The value is CLM-ml v1's `dpai_min` ('minimum plant area index to be considered a vegetation layer', m²/m², `MLclm_varctl.F90` l. 37), which v1 applied once at setup with the first step's LAI and SAI (`MLinitVerticalMod.F90` l. 199-265); here it is applied each step without removing layers. A patch with no plant area thus has 0.01 in each of its leafy layers. `0` gives CLM-ml v2 exactly. CHATS7 unchanged (its thinnest layer is 0.011) | CLM-ml v1 `initVerticalStructure` |
| G19 | Gridcell output | Area-weighted mean over the vegetated patches (bare ground and other land units left out, decision 29) | — |

## Open issues (not fixed; for the maintainer)

1. **CLM6's default roughness is `Meier2022`**, not `ZengWang2007`: CLM6 computes `z0m` and
   `displa` from LAI each step, so its forcing height varies in time. CLM-ml fixes the forcing
   height at setup; G14 follows the agreed rule (decision 2).
2. **Calendar**: TRENDY CLM runs use `noleap`; this model is Gregorian (G3). A no-leap model
   calendar would touch the time manager, chunks, output time stamps and restarts.
3. **Frozen soil**: `H2OSOI` is total water; CLM-ml's `SoilInit`/`clmData` put it all in
   liquid (ice = 0), so frozen soils have liquid water for root uptake and soil thermal
   properties. CLM's soil-moisture stream keeps the model's own ice fraction. (`SOILLIQ` and
   `SOILICE` are in the history files.)
4. **No snowpack**: snowfall is intercepted by the canopy (G6) but snow reaching the ground
   is not accumulated; the ground is never snow covered (no change to albedo, ground heat
   flux, LAI burial).
5. **Root biomass** 500 g m⁻² everywhere (G17) and **canopy bottom** unused (G13): science
   choices to revisit.
6. **Initial state** from monthly means (G11), not a spun-up restart: expect a spin-up drift
   of the soil temperature over the first weeks.
7. **Meteorology memory**: the meteorology is built on the host per chunk at the model step
   (~39 numbers per gridcell-step, ~7 GB per month for 14 346 gridcells). Interpolating the
   6-hourly records inside the compiled chunk, as is done for vegetation and soil water, would
   remove it (Phase 7).
8. **`forc_hgt_u_patch` after setup**: the step writes the gridcell's 30 m into this container
   (`TowerMetCurr`); unused (the canopy uses `zref_forcing`, set at setup to the patch's
   height), but misleading if output.
9. **Tiny canopies**: grass and crop tops go down to 5 mm (surface-data heights with 10 + 10
   layers); not filtered.
10. **Round-off amplification in real patches**: a 1e-15 relative change of LAI moves a
    July day of some 1° patches by up to ~8 W m⁻² in latent heat within hours of sunrise
    (largest for a needleleaf patch with CLM6's `dleaf` = 0.65 mm; `test_grid_setup.py`).
    CHATS7's noise twins differ by ≤ 4e-11, so these patches sit on a discrete switch
    (numerics doc §5 describes a similar earlier solver defect). It limits bit-level
    regression tests of grid runs to the night and will matter for gradients: locate the
    switch (compare routines on perturbed vs unperturbed inputs, numerics doc §1).
11. **CPU compilation cache**: loading the cached batched first step of some batches back
    from the on-disk cache crashes in LLVM's object loader (JAX 0.9.2, deterministic; the
    same program compiled afresh runs). That program is now compiled without the cache on
    CPU (`main.without_persistent_cache_on_cpu`). The earlier "intermittent" CPU
    segmentation faults of Phase 6 (`test_c3_c4_groups`, `test_devices`) may have been the
    same bug.
12. **CLM-ml's canopy energy balance check below freezing** (`CanopyFluxesDiagnostics 3`,
    Fortran CLM-ml v2): the canopy-top latent heat is `etflx × LatVap(tref)` (air
    temperature), the leaf and soil latent heats use the latent heat at their own
    temperatures, and `LatVap` jumps from vaporization to sublimation (+13 %) at 0 °C. When
    air, leaves and ground straddle freezing the check fails by ~0.1-2 W m⁻² (56 of 61
    patches in two January days over Illinois; with a single latent heat 0 failed). The
    state's energy is unaffected; the *reported* total latent heat uses the air temperature's
    value. CLM sums the components (`eflx_lh_tot = hvap·qflx_evap_veg + htvp·qflx_evap_soi`).
    Grid runs list the check in `numerics.ignore_checks` (reported every chunk, no
    exclusion). Fixing the diagnostic (sum the components, as CLM) is a science change
    for the maintainer (CHATS7 never freezes, so its results would not change).
13. **Bare-ground patches**: where CLM treats a patch as not vegetated (`elai + esai < 0.05`:
    bare-ground fluxes), this model keeps a sparse canopy (0.01 m² m⁻² of stems per leafy
    layer, G20). A CLM-ml bare-ground path (or leaving the patch out of its gridcell for those
    steps) would copy CLM exactly.
14. **Diffuse transmittance above 1 in thin layers (diagnosed and fixed 2026-10-29: decision 30, G20).**
    Seen as NaN in 5 of 2 902 regional patches in January (deciduous shrubs in Wyoming,
    LAI 0, SAI 0.108). Diagnosis (probes in `probe/issue14/`, no code changed):
    - CLM-ml v2 computes a layer's diffuse transmittance with a 9-angle midpoint rule
      (`MLSolarRadiationMod.F90` lines 203-210; the JAX translation is the same):
      `td = 2·Δθ·Σ exp(-G(θ)/cosθ·dpai·Ω)·sinθ·cosθ`. At `dpai -> 0` it gives 1.00510, not 1
      (the midpoints' weights are not normalized), so **`td > 1` for layers with less than
      ~0.005 m² m⁻² of plant area** (0.0050-0.0058 depending on the leaf angle).
    - Layers are fixed at setup as fractions of the canopy's plant area (`dpai_min` = 0.01
      applies to the unit-LAI profile in `initVerticalStructure`, Fortran too), and each
      step's layer plant area is fraction × (LAI + SAI). The thinnest of the 10 layers holds
      1.2-1.5 % of the plant area, so any patch with LAI + SAI below ~0.33-0.41 has a
      `td > 1` layer.
    - With `1 - td < 0`, the Norman longwave equations (`refld = (1 - td)·ρ`, emission
      `εσT⁴(1 - td)`, absorption containing `-2·src`) give the layer net longwave that
      *increases* with its temperature: about `2(td - 1)/dpai · 4εσT³` (~24 W m⁻² K⁻¹ for the
      shrub's bottom layer, `dpai` 0.0013, `td` 1.0039), against a sensible-heat loss of
      ~`2·cp·gbh` (~33 W m⁻² K⁻¹). In calm conditions the feedback wins: on 13 January at
      sunrise the bottom layer went 275 K -> 310 K -> 440 K -> 1e147 K -> NaN, while the 12
      earlier sunrises (windier) stayed stable. The shortwave also uses `(1 - td)` (energy
      created, not absorbed, in those layers).
    - **Extent** (global domain, monthly LAI/SAI): 21 727 of 72 575 patches have a `td > 1`
      layer in some month: 32 % of the vegetated area in January-February, 8 % in
      July-August. Most do not diverge, but their radiation is not conserving energy, so
      their fluxes are biased. CHATS7 never reaches it (thinnest layer 0.011 m² m⁻²).
    - **Causal test**: the same January with `td` normalized (divided by 1.00510, probe
      only): all 5 patches of the gridcell finite and physical, the shrub included. Capping
      `td` at exactly 1 is not a fix: `refld = 0` makes the Norman equations singular, which
      also shows they are ill-conditioned as a layer's plant area goes to 0.
    - **For the maintainer** (science choices, each changes results, CHATS7 included for the
      first two): (a) normalize the quadrature weights so `td(0) = 1` (changes `td` by -0.5 %
      everywhere); (b) a more accurate integration of the diffuse transmittance; (c) keep
      layers with too little plant area out of the radiation (an absolute minimum plant area
      per layer each step, as `dpai_min` was in CLM-ml v1); (d) a radiation formulation that
      stays well conditioned as `dpai -> 0`. CLM (big leaf) has no layers, so it has nothing
      to copy here.
    - **Fix chosen by the maintainer** (decision 30): a minimum plant area of 0.01 m² m⁻² in
      every leafy layer at every step, added as stems (G20). It keeps every layer's `td` below
      1 (the threshold is ~0.005), keeps the layer structure fixed (no layers switching on and
      off: no orphaned leaf states, no start-date dependence, no new discontinuities), and
      leaves CHATS7 bit for bit (reference regenerated only to add the new field; every
      existing field identical). Global effect (2000 monthly LAI/SAI): a floored layer in
      37 % of patch-months (58 % of the vegetated area in January, 25 % in July), adding a
      median 0.005 m² m⁻² (+1.2 % of the prescribed LAI + SAI; 95th percentile +18 %, max
      0.1). Test: `tests/regression/test_grid_setup.py::test_thin_layers_keep_minimum_plant_area`
      (the Wyoming shrub through 16 January: max `td` 0.996, leaf temperature 256-279 K).
      Options considered and why not: normalizing the quadrature (changes every result,
      CHATS7 included, and leaves the longwave equations ill-conditioned as plant area -> 0);
      v1's setup-time removal of thin layers (start-date dependent; fails for patches with
      almost no plant area); removing layers each step (orphaned leaf heat, water and
      intercepted water, abrupt profile changes, discrete switches); a floor on the canopy's
      LAI + SAI (does not reach the thin bottom layers of tree and shrub profiles: 0.41 needed).
15. **Wet needleleaf canopies** (observation): CLM6's `dleaf` for needleleaf trees is 0.65 mm
    (CLM5: 4 cm), and CLM-ml's leaf boundary-layer conductance grows as `dleaf` shrinks.
    Needleleaf evergreen temperate patches in the south-eastern US have January latent heat
    16 W m⁻² on average (other PFTs 6.9) with single wet days of 150-226 W m⁻² fed by
    sensible heat of -150 to -225 W m⁻². To be compared with CLM.
16. **CPU gradient crash** (not Phase 6b): `tests/test_model_api.py::
    test_model_chunk_gradient_matches_finite_difference` (CHATS7, 50 steps) segfaults while
    *running* the compiled CPU gradient (`xla::cpu::SmallKernelThunk::Execute`), with an empty
    compilation cache too, on Derecho (JAX 0.9.2); it failed the same way in the end-of-Phase-6
    job on the commit before Phase 6b. Resembles the JAX 0.6.2 CPU runtime bug of Phase 4
    (gradients over chunks of several steps). Gradients ran on the A100 (Phase 6 step 1b).

## Regional test (Phase 6b step 6, 2026-10-29)

30-50 N, 110-80 W: 522 gridcells, 2 902 patches (396 C4), January and July 2000, one A100
(job `regional_dev`, commit `6c1b66f`; daily patch and gridcell output in
`/glade/derecho/scratch/linnia/clm-ml-jax/runs/regional/`).

| | January | July |
|---|---|---|
| setup | 71 s | 41 s |
| first step (compiled) | 683 s | 34 s (program from the cache) |
| month chunk, 1 487 steps, incl. its compilation | 1 879 s | 1 893 s |
| patch-steps/s (incl. compilation) | 2 296 | 2 279 |
| peak GPU memory | 1.6 GB | 1.6 GB |
| patches excluded | 5 (0.17 %, issue 14) | 0 |
| canopy check 3 above tolerance (ignored, issue 12) | most patches | — |
| gridcell latent heat, mean (min, max of daily means) | 7.8 (-4.3, 92) W m⁻² | 93 (14, 196) W m⁻² |
| gridcell GPP, mean | 0.47 µmol m⁻² s⁻¹ | 8.7 µmol m⁻² s⁻¹ |
| canopy-top air temperature, mean | 269 K | 295 K |

All gridcell values finite. `soil_water_stress` (`btran`) is 1 everywhere: CLM-ml's WUE
stomatal scheme sets it to 1 and applies water stress through leaf water potential (as the
Fortran); use the leaf water potential outputs instead.
17. **LAI is not everywhere differentiable** (raise again: maintainer, 2026-10-29). Two places
    in the vegetation data path: (a) CLM's satellite-phenology cutoff (G9) sets LAI and SAI to
    exactly 0 below 0.05: a *jump* in the data (zero gradient below 0.05, a step at 0.05);
    (b) the per-layer minimum plant area (G20) is a *kink*: in a floored layer the leaf area
    still follows LAI (d leaf area / d LAI = the layer's fraction) but the layer's total plant
    area is pinned at 0.01 (d total / d LAI = 0), and the slope changes where a layer crosses
    0.01. The gradient with respect to SAI is zero in floored layers. Matters for
    calibration and sensitivity with respect to LAI/SAI (decision 7); a smooth alternative
    for either changes forward values near the threshold and is a science choice.
