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
)
from multilayer_canopy.MLclm_varcon import (  # noqa: F401
    J_to_umol,
    chil_max,
    chil_min,
    cpbio,
    cpd,
    cpw,
    dewmx,
    emg,
    fcarbon,
    fwater,
    fwet_exponent,
    interception_fraction,
    kb_max,
    lapse_rate,
    maximum_leaf_wetted_fraction,
    mmdry,
    mmh2o,
    rgas,
    wind_forc_min,
)
