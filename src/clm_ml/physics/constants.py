"""Physical constants used by the processes in ``clm_ml.physics``.

For now these are the legacy ``clm_src_main.clm_varcon`` and
``multilayer_canopy.MLclm_varcon`` values, re-exported so the processes name one place; the
definitions move here when the legacy modules go (Phase 7.6).
"""

from clm_src_main.clm_varcon import (  # noqa: F401
    cpice,
    cpliq,
    csol_bedrock,
    denh2o,
    denice,
    grav,
    rpi,
    sb,
    tfrz,
    thk_bedrock,
    tkice,
    tkwat,
    vkc,
)
from multilayer_canopy.MLclm_varcon import (  # noqa: F401
    J_to_umol,
    LcL_max,
    LcL_min,
    Pr0,
    Pr1,
    Pr2,
    beta_neutral_max,
    c2,
    cd,
    chil_max,
    chil_min,
    cpbio,
    cpd,
    cpw,
    cr,
    dewmx,
    emg,
    eta_max,
    fcarbon,
    fwater,
    fwet_exponent,
    interception_fraction,
    kb_max,
    lapse_rate,
    maximum_leaf_wetted_fraction,
    mmdry,
    mmh2o,
    ra_max,
    rgas,
    wind_forc_min,
    z0mg,
)
