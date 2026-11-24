"""Soil properties at setup (``clm_ml.setup.soil``)."""

import jax
import numpy as np
import pytest

from clm_ml.setup import soil, tables

jax.config.update("jax_enable_x64", True)
NAN = float("nan")


@pytest.mark.parametrize("k, name", list(enumerate(tables.SOIL_TEXTURES)))
def test_texture_class_takes_its_own_row(k, name):
    """A texture class gets its own sand, clay and Clapp-Hornberger values
    (SoilTexMod.F90; Fortran's class number m indexes its 1-based tables), every class
    including the last (clay)."""
    z, dz, zi = soil.soil_layers()
    # no organic matter: the mineral values unchanged
    props = soil.soil_properties(np.array([NAN]), np.array([NAN]), np.array([0.0]), [name], z,
                                 soil.NLEVSOI)  # fmt: skip
    np.testing.assert_allclose(props["watsat"][0, 0], tables.WATSAT_TEX[k], rtol=1e-15)
    np.testing.assert_allclose(props["sucsat"][0, 0], -tables.SMPSAT_TEX[k], rtol=1e-15)
    np.testing.assert_allclose(props["bsw"][0, 0], tables.BSW_TEX[k], rtol=1e-15)
    np.testing.assert_allclose(props["hksat"][0, 0], tables.HKSAT_TEX[k] / 60.0, rtol=1e-15)


def test_percentages_do_not_use_the_class():
    """With sand and clay given, the hydraulic values come from them (CLM5's formulas), as in
    Fortran; the class is not used."""
    z, dz, zi = soil.soil_layers()
    args = (np.array([10.0]), np.array([35.0]), np.array([0.0]))
    a = soil.soil_properties(*args, ["sand"], z, soil.NLEVSOI)
    b = soil.soil_properties(*args, ["clay"], z, soil.NLEVSOI)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_allclose(a["watsat"][0, 0], 0.489 - 0.00126 * 10.0, rtol=1e-15)


def test_unknown_class():
    z, dz, zi = soil.soil_layers()
    with pytest.raises(ValueError, match="not found"):
        soil.soil_properties(np.array([NAN]), np.array([NAN]), np.array([0.0]), ["peat"], z,
                             soil.NLEVSOI)  # fmt: skip


# The US-MMS PLUMBER2 surface dataset of CTSM 5.4 (surfdata_1x1_PLUMBER2_US-MMS_hist_2000_
# 16pfts_c251023.nc): sand, clay (%) and organic matter (kg m-3) of its 10 layers.
MMS_SAND = [31.0, 31.0, 31.0, 31.0, 31.0, 32.0, 37.0, 38.0, 40.0, 40.0]
MMS_CLAY = [19.0, 19.0, 19.0, 19.0, 25.0, 27.0, 25.0, 24.0, 24.0, 24.0]
MMS_ORGANIC = [37.945926666259766] * 4 + [14.575501441955566, 8.504103660583496,
               5.048689365386963, 4.06270694732666, 3.2244069576263428, 3.2244069576263428]  # fmt: skip

# CLM's soil properties of the 20 soil levels at US-MMS (history fields WATSAT, SUCSAT, BSW,
# HKSAT of a ctsm5.4.004 run on that dataset, single precision).
CLM_MMS = {
    "watsat": [0.589481770992279, 0.5877304077148438, 0.5848115086555481, 0.5807250142097473,
               0.4979337751865387, 0.47493281960487366] + [0.45743364095687866] * 2
              + [0.45327311754226685] * 2 + [0.44830793142318726] * 10,
    "sucsat": [213.81076049804688] * 4 + [265.52825927734375, 270.694580078125,
               239.22572326660156, 239.22230529785156, 233.87538146972656, 233.8716278076172,
               221.59512329101562, 221.59133911132812, 221.58717346191406, 221.5823211669922,
               221.5764617919922, 221.5696258544922, 221.56178283691406, 221.55294799804688,
               221.54312133789062, 221.53231811523438],
    "bsw": [5.042189598083496, 5.2050652503967285, 5.476524353027344, 5.8565673828125,
            6.957989692687988, 7.395127296447754] + [7.083646297454834] * 2
           + [6.8908209800720215] * 2 + [6.8568115234375] * 10,
    "hksat": [0.0038631886709481478, 0.00386215397156775, 0.003860094817355275,
              0.0038562011905014515, 0.003085747128352523, 0.003033724147826433]
             + [0.003393391380086541] * 2 + [0.0035150698386132717] * 2
             + [0.0037716722581535578] * 10,
}  # fmt: skip


def test_texture_profile_maps_levels_to_surface_data_layers():
    """Each soil level takes the surface-data layer whose interfaces bracket its node depth
    (SoilStateInitTimeConst, ctsm5.4.004); levels below nlevsoi take the bottom layer's
    texture and no organic matter."""
    z, dz, zi = soil.soil_layers()
    layer_values = np.arange(10, dtype=float) + 1.0
    sand, clay, organic = soil.texture_profile(layer_values, layer_values, layer_values, z,
                                               soil.NLEVSOI)  # fmt: skip
    expected = [1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 9] + [10] * 8 + [10] * 5
    np.testing.assert_array_equal(sand[0], expected)
    np.testing.assert_array_equal(clay[0], expected)
    np.testing.assert_array_equal(organic[0], expected[:20] + [0.0] * 5)


def test_texture_profile_caps_organic_matter():
    z, dz, zi = soil.soil_layers()
    _, _, organic = soil.texture_profile([[30.0, 30.0]], [[20.0, 20.0]], [[300.0, 50.0]], z,
                                         soil.NLEVSOI)  # fmt: skip
    assert organic[0, 0] == 130.0 and organic[0, 19] == 50.0


def test_us_mms_soil_properties_equal_clm():
    """With the texture of every level from the surface dataset, CLM-ml's formulas give
    CLM's porosity, saturated suction, b and saturated conductivity at every soil level
    (equal to CLM's single-precision history)."""
    z, dz, zi = soil.soil_layers()
    sand, clay, organic = soil.texture_profile(MMS_SAND, MMS_CLAY, MMS_ORGANIC, z, soil.NLEVSOI)
    props = soil.soil_properties(sand, clay, organic, [None], z, soil.NLEVSOI)
    for key, clm in CLM_MMS.items():
        np.testing.assert_array_equal(props[key][0, :20].astype(np.float32),
                                      np.asarray(clm, dtype=np.float32), err_msg=key)  # fmt: skip


def test_organic_matter_below_half_a_metre_only_with_a_profile():
    """One texture per column keeps CLM-ml v2's organic matter in the top 0.5 m; a profile
    applies each level's organic matter at every level, as CTSM. With the same texture and
    organic matter everywhere the two agree above 0.5 m and differ below."""
    z, dz, zi = soil.soil_layers()
    one = soil.soil_properties(np.array([31.0]), np.array([19.0]), np.array([40.0]), [None], z,
                               soil.NLEVSOI)  # fmt: skip
    sand, clay, organic = soil.texture_profile([31.0], [19.0], [40.0], z, soil.NLEVSOI)
    per_level = soil.soil_properties(sand, clay, organic, [None], z, soil.NLEVSOI)
    top = z <= 0.5
    for key in ("watsat", "sucsat", "bsw", "hksat", "tkmg", "tkdry", "csol"):
        np.testing.assert_array_equal(per_level[key][0, top], one[key][0, top], err_msg=key)
    deep = (z > 0.5) & (np.arange(len(z)) < soil.NLEVSOI)
    assert np.all(per_level["watsat"][0, deep] > one["watsat"][0, deep])
