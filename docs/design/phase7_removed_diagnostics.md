# Phase 7 removed diagnostics (D6)

Legacy container fields that are **not** in the step's results since Phase 7.1
(`clm_ml.state.StepState`, `PatchLayers`): the step returns only the prognostic state and
the diagnostics that the output variables, the legacy `.out` files and the conservation
checks read (`clm_ml.state.Diagnostics`). Before 7.1 every field of the `mlcanopy` and
`temperature` containers could be requested as a validation stream variable
(`mlcanopy.<field>`); these no longer can. To add one back: a field in `Diagnostics` (and in
the legacy view's `FIELDS` if it should be named as before), computed by the routine listed.

Inside the step the physics still computes all of them until Phase 7.2 moves the processes:
**Read by** lists the modules that read a field (approximate: found by searching the source
for reads of the attribute, including a module re-reading its own accumulation). Fields read
by another process are work values or inputs of the step; they become part of the step's
within-step state or inputs in 7.2, not outputs. Fields read by no module (`-`) are pure
diagnostics; 7.2 drops their computation when their process moves (and says so here).

Meanings are from the container's comments (`MLCanopyFluxesType`, CLM-ml v2
`MLCanopyFluxesType.F90`).

| Field | Meaning | Set by | Read by |
|---|---|---|---|
| `ztop_canopy` | Canopy foliage top height (m) | MLinitVerticalMod, clm_ml.vector_setup | MLCanopyTurbulenceMod, MLinitVerticalMod |
| `zbot_canopy` | Canopy foliage bottom height (m) | MLinitVerticalMod, clm_ml.vector_setup | MLinitVerticalMod |
| `root_biomass_canopy` | Fine root biomass (g/m2) | clm_ml.vector_setup | MLPlantHydraulicsMod |
| `pbeta_lai_canopy` | Beta distribution parameters for LAI (-); shape (np,3) | MLinitVerticalMod, clm_ml.vector_setup | MLinitVerticalMod, clm_ml.vector_setup |
| `pbeta_sai_canopy` | Beta distribution parameters for SAI (-); shape (np,3) | MLinitVerticalMod, clm_ml.vector_setup | MLinitVerticalMod, clm_ml.vector_setup |
| `zref_forcing` | Reference height (m) | MLinitVerticalMod, clm_ml.vector_setup | MLCanopyTurbulenceMod, MLGetAtmForcingMod, MLinitVerticalMod |
| `qref_forcing` | Specific humidity at zref, interpolated (kg/kg) | MLGetAtmForcingMod | MLCanopyFluxesMod, MLCanopyTurbulenceMod, MLFluxProfileSolutionMod, MLGetAtmForcingMod |
| `uref_forcing` | Wind speed at zref, interpolated (m/s) | MLGetAtmForcingMod | MLCanopyFluxesMod, MLCanopyTurbulenceMod, MLFluxProfileSolutionMod, MLGetAtmForcingMod |
| `co2ref_forcing` | Atmospheric CO2, interpolated (umol/mol) | MLGetAtmForcingMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLGetAtmForcingMod |
| `o2ref_forcing` | Atmospheric O2 (mmol/mol) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLLeafPhotosynthesisMod |
| `lwsky_forcing` | Atmospheric LW, interpolated (W/m2) | MLCanopyFluxesMod, MLGetAtmForcingMod | MLCanopyFluxesMod, MLGetAtmForcingMod, MLLongwaveRadiationMod |
| `qflx_rain_forcing` | Rainfall (kg H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLCanopyWaterMod |
| `qflx_snow_forcing` | Snowfall (kg H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLCanopyWaterMod |
| `tacclim_forcing` | Average temperature for acclimation (K) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLCanopyNitrogenProfileMod, MLLeafPhotosynthesisMod |
| `eref_forcing` | Vapor pressure at zref (Pa) | MLGetAtmForcingMod | MLFluxProfileSolutionMod, MLGetAtmForcingMod |
| `thref_forcing` | Potential temperature at zref (K) | MLGetAtmForcingMod | MLCanopyTurbulenceMod, MLFluxProfileSolutionMod, MLGetAtmForcingMod |
| `thvref_forcing` | Virtual potential temperature at zref (K) | MLGetAtmForcingMod | MLCanopyTurbulenceMod, MLGetAtmForcingMod |
| `rhoair_forcing` | Air density at zref (kg/m3) | MLGetAtmForcingMod | MLGetAtmForcingMod |
| `mmair_forcing` | Molecular mass of air (kg/mol) | MLGetAtmForcingMod | MLGetAtmForcingMod |
| `cpair_forcing` | Specific heat of air, const-P (J/mol/K) | MLGetAtmForcingMod | MLFluxProfileSolutionMod, MLGetAtmForcingMod, MLLeafFluxesMod, MLSoilFluxesMod |
| `lwveg_canopy` | Absorbed LW: vegetation (W/m2) | MLCanopyFluxesMod, MLLongwaveRadiationMod | MLCanopyFluxesMod |
| `lwvegsun_canopy` | Absorbed LW: sunlit canopy (W/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `lwvegsha_canopy` | Absorbed LW: shaded canopy (W/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `etveg_canopy` | Water vapor: vegetation (mol H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `etvegsun_canopy` | Water vapor: sunlit canopy (mol H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `etvegsha_canopy` | Water vapor: shaded canopy (mol H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `etflx_canopy` | Water vapor including soil (mol H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `gac_to_hc_canopy` | Aerodynamic conductance above canopy (mol/m2/s) | MLCanopyTurbulenceMod | - |
| `qflx_intr_canopy` | Intercepted precipitation (kg H2O/m2/s) | MLCanopyFluxesMod, MLCanopyWaterMod | MLCanopyFluxesMod, MLCanopyWaterMod |
| `qflx_tflrain_canopy` | Rain throughfall onto ground (kg H2O/m2/s) | MLCanopyFluxesMod, MLCanopyWaterMod | MLCanopyFluxesMod, MLCanopyWaterMod |
| `qflx_tflsnow_canopy` | Snow throughfall onto ground (kg H2O/m2/s) | MLCanopyFluxesMod, MLCanopyWaterMod | MLCanopyFluxesMod, MLCanopyWaterMod |
| `uaf_canopy` | Wind speed at canopy top (m/s) | MLCanopyTurbulenceMod, MLFluxProfileSolutionMod | MLCanopyTurbulenceMod, MLFluxProfileSolutionMod |
| `taf_canopy` | Air temperature at canopy top for ObuFunc (K) | MLCanopyTurbulenceMod, MLFluxProfileSolutionMod | MLCanopyTurbulenceMod, MLFluxProfileSolutionMod |
| `qaf_canopy` | Specific humidity at canopy top for ObuFunc (kg/kg) | MLCanopyTurbulenceMod, MLFluxProfileSolutionMod | MLCanopyTurbulenceMod, MLFluxProfileSolutionMod |
| `obu_canopy` | Obukhov length (m) | MLCanopyFluxesMod, MLCanopyTurbulenceMod | MLCanopyFluxesMod, MLCanopyTurbulenceMod |
| `zdisp_canopy` | Displacement height (m) | MLCanopyFluxesMod, MLCanopyTurbulenceMod | MLCanopyFluxesMod, MLCanopyTurbulenceMod |
| `z0m_canopy` | Roughness length for momentum (m) | MLCanopyFluxesMod, MLCanopyTurbulenceMod | MLCanopyFluxesMod |
| `g0_canopy` | Ball-Berry/Medlyn minimum conductance (mol H2O/m2/s) | MLLeafPhotosynthesisMod | - |
| `g1_canopy` | Ball-Berry/Medlyn slope parameter | MLLeafPhotosynthesisMod | - |
| `albsoib_soil` | Direct beam albedo of ground (-); shape (np, numrad+1) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLSolarRadiationMod |
| `albsoid_soil` | Diffuse albedo of ground (-); shape (np, numrad+1) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLSolarRadiationMod |
| `swsoi_soil` | Absorbed SW: ground (W/m2); shape (np, numrad+1) | MLCanopyFluxesMod, MLSolarRadiationMod | MLCanopyFluxesMod, MLSolarRadiationMod |
| `lwsoi_soil` | Absorbed LW: ground (W/m2) | MLCanopyFluxesMod, MLLongwaveRadiationMod | MLCanopyFluxesMod |
| `etsoi_soil` | Water vapor: ground (mol H2O/m2/s) | MLCanopyFluxesMod, MLSoilFluxesMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod |
| `tg_bef_soil` | Soil surface temperature, previous step (K) | MLCanopyFluxesMod, MLRungeKuttaMod | MLFluxProfileSolutionMod, MLSoilFluxesMod |
| `dtg_soil` | Change in tg over RK step (K); shape (np, nrk+1) | MLRungeKuttaMod | - |
| `eg_soil` | Soil surface vapor pressure (Pa) | MLSoilFluxesMod | MLFluxProfileSolutionMod |
| `rhg_soil` | Relative humidity at soil surface (-) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLSoilFluxesMod |
| `gac0_soil` | Aerodynamic conductance for soil (mol/m2/s) | MLCanopyFluxesMod, MLCanopyTurbulenceMod, MLFluxProfileSolutionMod | MLCanopyFluxesMod, MLCanopyTurbulenceMod, MLFluxProfileSolutionMod, MLSoilFluxesMod |
| `soil_t_soil` | Temperature of first snow/soil layer (K) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLSoilFluxesMod |
| `soil_dz_soil` | Depth to first snow/soil layer (m) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLSoilFluxesMod |
| `soil_tk_soil` | Thermal conductivity of first layer (W/m/K) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLSoilFluxesMod |
| `soilres_soil` | Soil evaporative resistance (s/m) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLSoilFluxesMod |
| `rsoil_soil` | Soil hydraulic resistance (MPa.s.m2/mmol H2O) | MLPlantHydraulicsMod | MLPlantHydraulicsMod |
| `dlai_frac_profile` | Layer LAI (fraction of canopy total) | MLinitVerticalMod, clm_ml.vector_setup | MLCanopyFluxesMod, MLinitVerticalMod |
| `dsai_frac_profile` | Layer SAI (fraction of canopy total) | MLinitVerticalMod, clm_ml.vector_setup | MLCanopyFluxesMod, MLinitVerticalMod |
| `dlai_profile` | Layer LAI (m2/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLCanopyWaterMod, MLSolarRadiationMod |
| `dsai_profile` | Layer SAI (m2/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod, MLSolarRadiationMod |
| `vcmax25_profile` | Layer vcmax at 25C (umol/m2/s) | MLCanopyNitrogenProfileMod | MLCanopyFluxesMod, MLCanopyNitrogenProfileMod |
| `jmax25_profile` | Layer jmax at 25C (umol/m2/s) | MLCanopyNitrogenProfileMod | MLCanopyNitrogenProfileMod |
| `kp25_profile` | Layer C4 kp at 25C (mol/m2/s) | MLCanopyNitrogenProfileMod | MLCanopyNitrogenProfileMod |
| `rd25_profile` | Layer rd at 25C (umol/m2/s) | MLCanopyNitrogenProfileMod | MLCanopyNitrogenProfileMod |
| `cpleaf_profile` | Layer leaf heat capacity (J/m2 leaf/K) | MLLeafHeatCapacityMod | MLFluxProfileSolutionMod, MLLeafFluxesMod, MLLeafHeatCapacityMod |
| `kb_profile` | Direct beam extinction coefficient (-) | MLSolarRadiationMod | MLCanopyNitrogenProfileMod, MLSolarRadiationMod |
| `tb_profile` | Layer transmittance of direct beam (-) | MLSolarRadiationMod | MLSolarRadiationMod |
| `td_profile` | Layer transmittance of diffuse radiation (-) | MLSolarRadiationMod | MLLongwaveRadiationMod, MLSolarRadiationMod |
| `lwsrc_profile` | Layer source: absorbed LW (W/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `rnsrc_profile` | Layer source: net radiation (W/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `stsrc_profile` | Layer source: storage heat flux (W/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `shsrc_profile` | Layer source: sensible heat (W/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `lhsrc_profile` | Layer source: latent heat (W/m2) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `etsrc_profile` | Layer source: water vapor (mol H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `trsrc_profile` | Layer source: transpiration (mol H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `evsrc_profile` | Layer source: evaporation (mol H2O/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `fco2src_profile` | Layer source: CO2 (umol CO2/m2/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `tair_bef_profile` | Layer air temperature, previous step (K) | MLCanopyFluxesMod, MLRungeKuttaMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod |
| `eair_bef_profile` | Layer vapor pressure, previous step (Pa) | MLCanopyFluxesMod, MLRungeKuttaMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod |
| `cair_bef_profile` | Layer CO2, previous step (umol/mol) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `dtair_profile` | Change in tair over RK step (K); shape (np, nlevmlcan+1, nrk+1) | MLRungeKuttaMod | - |
| `deair_profile` | Change in eair over RK step (Pa); shape (np, nlevmlcan+1, nrk+1) | MLRungeKuttaMod | - |
| `wind_data_profile` | Layer wind speed FROM DATASET (m/s) | setup | MLFluxProfileSolutionMod |
| `tair_data_profile` | Layer air temperature FROM DATASET (K) | setup | MLFluxProfileSolutionMod |
| `eair_data_profile` | Layer vapor pressure FROM DATASET (Pa) | setup | MLFluxProfileSolutionMod |
| `stair_profile` | Layer air storage heat flux (W/m2) | MLCanopyFluxesMod, MLFluxProfileSolutionMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod |
| `kc_eddy_profile` | Layer eddy diffusivity, HF (m2/s) | MLCanopyFluxesMod, MLCanopyTurbulenceMod | MLCanopyFluxesMod, MLCanopyTurbulenceMod |
| `lwleaf_mean_profile` | Layer mean: absorbed LW per leaf (W/m2 leaf) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `rnleaf_mean_profile` | Layer mean: net radiation per leaf (W/m2 leaf) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `stleaf_mean_profile` | Layer mean: storage heat per leaf (W/m2 leaf) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `shleaf_mean_profile` | Layer mean: sensible heat per leaf (W/m2 leaf) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `lhleaf_mean_profile` | Layer mean: latent heat per leaf (W/m2 leaf) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `etleaf_mean_profile` | Layer mean: water vapor per leaf (mol H2O/m2 leaf/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `trleaf_mean_profile` | Layer mean: transpiration per leaf (mol H2O/m2 leaf/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `evleaf_mean_profile` | Layer mean: evaporation per leaf (mol H2O/m2 leaf/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `fco2_mean_profile` | Layer mean: net photosynthesis (umol CO2/m2 leaf/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `apar_mean_profile` | Layer mean: absorbed PAR (umol/m2 leaf/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `gs_mean_profile` | Layer mean: stomatal conductance (mol H2O/m2 leaf/s) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `tleaf_mean_profile` | Layer mean: leaf temperature (K) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `h2ocan_bef_profile` | Layer intercepted water, previous step (kg H2O/m2) | MLCanopyFluxesMod, MLRungeKuttaMod | MLCanopyFluxesMod, MLCanopyWaterMod |
| `dh2ocan_profile` | Change in h2ocan over RK step; shape (np, nlevmlcan+1, nrk+1) | MLRungeKuttaMod | - |
| `fwet_profile` | Layer wet fraction of PAI (-) | MLCanopyWaterMod | MLCanopyFluxesMod, MLCanopyWaterMod, MLFluxProfileSolutionMod, MLLeafFluxesMod |
| `fdry_profile` | Layer green-dry fraction of PAI (-) | MLCanopyWaterMod | MLCanopyFluxesMod, MLCanopyWaterMod, MLFluxProfileSolutionMod, MLLeafFluxesMod |
| `tleaf_bef_leaf` | Leaf temperature, previous step (K) | MLCanopyFluxesMod, MLRungeKuttaMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLLeafFluxesMod |
| `lwleaf_leaf` | Leaf absorbed LW (W/m2 leaf) | MLCanopyFluxesMod, MLLongwaveRadiationMod | MLCanopyFluxesMod, MLLongwaveRadiationMod |
| `stleaf_leaf` | Leaf storage heat flux (W/m2 leaf) | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLLeafFluxesMod | MLCanopyFluxesMod, MLFluxProfileSolutionMod |
| `trleaf_leaf` | Leaf transpiration (mol H2O/m2 leaf/s) | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLLeafFluxesMod | MLCanopyFluxesMod, MLCanopyWaterMod, MLFluxProfileSolutionMod, MLPlantHydraulicsMod |
| `evleaf_leaf` | Leaf evaporation (mol H2O/m2 leaf/s) | MLCanopyFluxesMod, MLFluxProfileSolutionMod, MLLeafFluxesMod | MLCanopyFluxesMod, MLCanopyWaterMod, MLFluxProfileSolutionMod |
| `gbh_leaf` | Boundary layer conductance: heat (mol/m2 leaf/s) | MLLeafBoundaryLayerMod | MLFluxProfileSolutionMod, MLLeafBoundaryLayerMod, MLLeafFluxesMod |
| `gbv_leaf` | Boundary layer conductance: H2O (mol H2O/m2 leaf/s) | MLLeafBoundaryLayerMod | MLFluxProfileSolutionMod, MLLeafBoundaryLayerMod, MLLeafFluxesMod, MLLeafPhotosynthesisMod |
| `gbc_leaf` | Boundary layer conductance: CO2 (mol CO2/m2 leaf/s) | MLLeafBoundaryLayerMod | MLLeafBoundaryLayerMod, MLLeafPhotosynthesisMod |
| `jmax25_leaf` | Leaf jmax at 25C (umol/m2/s) | MLCanopyNitrogenProfileMod | MLCanopyNitrogenProfileMod, MLLeafPhotosynthesisMod |
| `kp25_leaf` | Leaf C4 kp at 25C (mol/m2/s) | MLCanopyNitrogenProfileMod | MLCanopyNitrogenProfileMod, MLLeafPhotosynthesisMod |
| `rd25_leaf` | Leaf rd at 25C (umol/m2/s) | MLCanopyNitrogenProfileMod | MLCanopyNitrogenProfileMod, MLLeafPhotosynthesisMod |
| `kc_leaf` | Michaelis-Menten constant for CO2 (umol/mol) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `ko_leaf` | Michaelis-Menten constant for O2 (mmol/mol) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `cp_leaf` | CO2 compensation point (umol/mol) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `vcmax_leaf` | Vcmax at leaf temperature (umol/m2/s) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `jmax_leaf` | Jmax at leaf temperature (umol/m2/s) | MLLeafPhotosynthesisMod | - |
| `kp_leaf` | C4 kp at leaf temperature (mol/m2/s) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `ceair_leaf` | Vapor pressure constrained for stomata (Pa) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `leaf_esat_leaf` | Leaf saturation vapor pressure (Pa) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `je_leaf` | Electron transport rate (umol/m2/s) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `ac_leaf` | Rubisco-limited gross An (umol CO2/m2 leaf/s) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `aj_leaf` | RuBP-limited gross An (umol CO2/m2 leaf/s) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `ap_leaf` | Product/CO2-limited gross An (umol CO2/m2 leaf/s) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `agross_leaf` | Gross photosynthesis (umol CO2/m2 leaf/s) | MLCanopyFluxesMod, MLLeafPhotosynthesisMod | MLCanopyFluxesMod, MLLeafPhotosynthesisMod |
| `rd_leaf` | Leaf respiration rate (umol CO2/m2 leaf/s) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `ci_leaf` | Intercellular CO2 (umol/mol) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `cs_leaf` | Leaf surface CO2 (umol/mol) | MLLeafPhotosynthesisMod | MLLeafPhotosynthesisMod |
| `lwp_bef_leaf` | Leaf water potential, previous step (MPa) | MLCanopyFluxesMod, MLRungeKuttaMod | MLCanopyFluxesMod, MLPlantHydraulicsMod |
| `hs_leaf` | Fractional humidity at leaf surface (-) | MLLeafPhotosynthesisMod | - |
| `vpd_leaf` | Leaf vapor pressure deficit (Pa) | MLLeafPhotosynthesisMod | - |
| `gspot_leaf` | Stomatal conductance without water stress (mol H2O/m2 leaf/s) | MLLeafPhotosynthesisMod | - |
| `lhtop_canopy` | Canopy-top latent heat flux: sub-step average of LatVap(tref)·etair(ncan) (W/m2; issue 12) | MLCanopyFluxesMod | MLCanopyFluxesMod |
| `t_ref2m_patch` | 2-m air temperature (K; `temperature` container, not set by CLM-ml) | TemperatureType | MLCanopyFluxesMod |

134 fields (11 read by no module).
