module UrbanTimeVarType
  ! Stand-in for CLM's UrbanTimeVarType in the harness (CLM's reads urban data streams
  ! through ESMF): only the type, with the two fields that SoilTemperatureMod and
  ! UrbBuildTempOleson2015Mod name. The harness has no urban column and never reads them.
  use shr_kind_mod, only : r8 => shr_kind_r8
  implicit none
  private
  type, public :: urbantv_type
     real(r8), public, pointer :: t_building_max(:) => null()
     real(r8), public, pointer :: p_ac(:) => null()
  end type urbantv_type
end module UrbanTimeVarType
