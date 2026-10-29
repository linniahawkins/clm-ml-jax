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
