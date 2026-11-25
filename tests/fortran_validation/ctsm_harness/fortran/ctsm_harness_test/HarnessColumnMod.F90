module HarnessColumnMod

  ! One vegetated soil column for the harness drivers, set up with CTSM's unit-test helpers:
  ! CLM's 20-level soil structure (25 ground levels), the fields that the soil water routines
  ! read, the soil water namelist of the CLM6 defaults and the parameters of
  ! ctsm60_params.c251211.nc. A driver fills the soil properties and the water state of its
  ! case and calls the CLM routine.

  use shr_kind_mod, only : r8 => shr_kind_r8
  use unittestSubgridMod, only : bounds, unittest_subgrid_teardown
  use unittestSimpleSubgridSetupsMod, only : setup_single_veg_patch
  use unittestTimeManagerMod, only : unittest_timemgr_setup, unittest_timemgr_teardown
  use unittestWaterTypeFactory, only : unittest_water_type_factory_type
  use clm_varpar, only : nlevgrnd, nlevsno
  use clm_varctl, only : use_bedrock, use_hydrstress
  use ColumnType, only : col
  use WaterType, only : water_type
  use SoilStateType, only : soilstate_type
  use SoilHydrologyType, only : soilhydrology_type
  use TemperatureType, only : temperature_type
  use EnergyFluxType, only : energyflux_type
  use CanopyStateType, only : canopystate_type
  use SoilWaterRetentionCurveClappHornberg1978Mod, only : &
       soil_water_retention_curve_clapp_hornberg_1978_type
  use SoilWaterMovementMod, only : init_soilwater_movement, params_inst

  implicit none
  private

  integer, parameter, public :: harness_nlevsoi = 20   ! soil levels (hydrologically active)
  integer, parameter, public :: harness_nlevgrnd = 25  ! ground levels

  type, public :: harness_column_type
     integer :: c                     ! the column's index
     integer :: num_hydrologyc        ! one column in the hydrology filter
     integer :: filter_hydrologyc(1)
     integer :: num_urbanc            ! no urban column
     integer :: filter_urbanc(1)
     type(unittest_water_type_factory_type) :: water_factory
     type(water_type) :: water_inst
     type(soilstate_type) :: soilstate_inst
     type(soilhydrology_type) :: soilhydrology_inst
     type(temperature_type) :: temperature_inst
     type(energyflux_type) :: energyflux_inst
     type(canopystate_type) :: canopystate_inst
     type(soil_water_retention_curve_clapp_hornberg_1978_type) :: retention_curve
  end type harness_column_type

  public :: harness_column_setup
  public :: harness_column_teardown

contains

  subroutine harness_column_setup(this, dtime, nbedrock)
    ! The column with a model step of dtime seconds and bedrock below level nbedrock.
    type(harness_column_type), intent(inout) :: this
    integer, intent(in) :: dtime
    integer, intent(in) :: nbedrock
    integer :: j, c, nu
    real(r8) :: dzj(harness_nlevgrnd), zij(0:harness_nlevgrnd)

    call unittest_timemgr_setup(dtime=dtime)
    call this%water_factory%init()
    call this%water_factory%setup_before_subgrid(my_nlevsoi=harness_nlevsoi, &
         nlevgrnd_additional=harness_nlevgrnd-harness_nlevsoi, my_nlevsno=5)
    call setup_single_veg_patch(pft_type=7)
    c = bounds%begc
    this%c = c
    ! One column in the hydrology filter, none in the urban filter
    this%num_hydrologyc = 1
    this%filter_hydrologyc(1) = c
    this%num_urbanc = 0
    this%filter_urbanc(1) = c

    ! CLM's soil layer structure 20SL_8.5m (initVerticalMod)
    do j = 1, 4
       dzj(j) = j*0.02_r8
    end do
    do j = 5, 13
       dzj(j) = dzj(4) + (j-4)*0.04_r8
    end do
    do j = 14, harness_nlevsoi
       dzj(j) = dzj(13) + (j-13)*0.10_r8
    end do
    do j = harness_nlevsoi+1, harness_nlevgrnd
       dzj(j) = dzj(harness_nlevsoi) + (((j-harness_nlevsoi)*25._r8)**1.5_r8)/100._r8
    end do
    zij(0) = 0._r8
    do j = 1, harness_nlevgrnd
       zij(j) = sum(dzj(1:j))
    end do
    col%snl(:) = 0
    do j = 1, harness_nlevgrnd
       col%dz(c,j) = dzj(j)
       col%zi(c,j) = zij(j)
       col%z(c,j) = 0.5_r8*(zij(j-1) + zij(j))
    end do
    col%zi(c,0) = 0._r8
    col%nbedrock(c) = nbedrock

    call this%water_factory%create_water_type(this%water_inst)

    ! Only the fields that the soil water routines read or write are allocated. The values
    ! here are placeholders (a loam, unfrozen); a driver sets its case's values.
    allocate(this%soilstate_inst%watsat_col(c:c,nlevgrnd), &
         this%soilstate_inst%hksat_col(c:c,nlevgrnd), &
         this%soilstate_inst%bsw_col(c:c,nlevgrnd), &
         this%soilstate_inst%sucsat_col(c:c,nlevgrnd), &
         this%soilstate_inst%hk_l_col(c:c,nlevgrnd), &
         this%soilstate_inst%smp_l_col(c:c,nlevgrnd))
    this%soilstate_inst%watsat_col(c,:) = 0.45_r8
    this%soilstate_inst%hksat_col(c,:) = 0.0035_r8
    this%soilstate_inst%bsw_col(c,:) = 6.9_r8
    this%soilstate_inst%sucsat_col(c,:) = 230._r8
    this%soilstate_inst%hk_l_col(c,:) = 0._r8
    this%soilstate_inst%smp_l_col(c,:) = 0._r8
    allocate(this%soilhydrology_inst%num_substeps_col(c:c), &
         this%soilhydrology_inst%zwt_col(c:c), &
         this%soilhydrology_inst%qcharge_col(c:c), &
         this%soilhydrology_inst%icefrac_col(c:c,nlevgrnd), &
         this%soilhydrology_inst%qin_col(c:c,nlevgrnd), &
         this%soilhydrology_inst%qout_col(c:c,nlevgrnd))
    this%soilhydrology_inst%num_substeps_col(c) = 0._r8
    this%soilhydrology_inst%zwt_col(c) = 8._r8
    this%soilhydrology_inst%qcharge_col(c) = 0._r8
    this%soilhydrology_inst%icefrac_col(c,:) = 0._r8
    this%soilhydrology_inst%qin_col(c,:) = 0._r8
    this%soilhydrology_inst%qout_col(c,:) = 0._r8
    allocate(this%temperature_inst%t_soisno_col(c:c,-nlevsno+1:nlevgrnd))
    this%temperature_inst%t_soisno_col(c,:) = 285._r8

    ! CLM6 defaults: soil depth to bedrock, root uptake from plant hydraulic stress, the
    ! adaptive moisture-form solver with a flux at the surface and no flux at the bottom
    ! (the soilwater_movement_inparm group of a CLM6 case's lnd_in)
    use_bedrock = .true.
    use_hydrstress = .true.
    open(newunit=nu, file='lnd_in_harness', status='replace')
    write(nu,'(a)') '&soilwater_movement_inparm'
    write(nu,'(a)') ' dtmin = 60.', ' expensive = 42', ' flux_calculation = 1', &
         ' inexpensive = 1', ' lower_boundary_condition = 2', &
         ' soilwater_movement_method = 1', ' upper_boundary_condition = 1', &
         ' verysmall = 1.e-8', ' xtolerlower = 1.e-2', ' xtolerupper = 1.e-1', '/'
    close(nu)
    call init_soilwater_movement()

    ! Parameters of ctsm60_params.c251211.nc
    params_inst%e_ice = 6._r8

    ! retention_curve needs no construction: the type has no data, and assigning CLM's
    ! (place-holder) constructor's result to it overwrote the memory after it.
  end subroutine harness_column_setup

  subroutine harness_column_teardown(this)
    type(harness_column_type), intent(inout) :: this

    call this%water_factory%teardown(this%water_inst)
    call unittest_subgrid_teardown()
    call unittest_timemgr_teardown()
  end subroutine harness_column_teardown

end module HarnessColumnMod
