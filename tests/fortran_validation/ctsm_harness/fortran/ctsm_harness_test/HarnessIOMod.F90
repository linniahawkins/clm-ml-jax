module HarnessIOMod

  ! Files of the harness drivers. A driver reads its cases from
  ! <harness_dir>/cases/<routine>.nml (one namelist group per case, written by
  ! generate_golden.py) and writes <harness_dir>/out/<routine>.out: a line "case=i", then one
  ! line "name=value" per output, reals with 17 significant digits. A driver of a sequence
  ! writes a line "step=k" before the outputs of each model step of a case.

  use shr_kind_mod, only : r8 => shr_kind_r8
  use HarnessPathsMod, only : harness_dir

  implicit none
  private

  public :: open_cases   ! open a routine's case file, if there is one
  public :: open_output  ! open a routine's output file
  public :: begin_case   ! start the outputs of a case
  public :: begin_step   ! start the outputs of a model step of a case (sequences)
  public :: put          ! write one output

  interface put
     module procedure put_real
     module procedure put_int
     module procedure put_real_array
  end interface put

contains

  subroutine open_cases(routine, unit, found)
    character(len=*), intent(in) :: routine
    integer, intent(out) :: unit
    logical, intent(out) :: found
    character(len=512) :: path

    unit = -1
    path = trim(harness_dir)//'/cases/'//trim(routine)//'.nml'
    inquire(file=trim(path), exist=found)
    if (found) then
       open(newunit=unit, file=trim(path), status='old', action='read')
    else
       write(*,'(a,a)') 'HARNESS no case file, driver skipped: ', trim(path)
    end if
  end subroutine open_cases

  subroutine open_output(routine, unit)
    character(len=*), intent(in) :: routine
    integer, intent(out) :: unit

    open(newunit=unit, file=trim(harness_dir)//'/out/'//trim(routine)//'.out', &
         status='replace', action='write')
  end subroutine open_output

  subroutine begin_case(unit, i)
    integer, intent(in) :: unit, i

    write(unit,'(a,i0)') 'case=', i
  end subroutine begin_case

  subroutine begin_step(unit, k)
    integer, intent(in) :: unit, k

    write(unit,'(a,i0)') 'step=', k
  end subroutine begin_step

  subroutine put_real(unit, name, value)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: value

    write(unit,'(a,a,es25.16e3)') name, '=', value
  end subroutine put_real

  subroutine put_int(unit, name, value)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: name
    integer, intent(in) :: value

    write(unit,'(a,a,i0)') name, '=', value
  end subroutine put_int

  subroutine put_real_array(unit, name, value)
    ! One line per element, "name(j)=value", j from 1 (CLM's level index).
    integer, intent(in) :: unit
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: value(:)
    integer :: j

    do j = 1, size(value)
       write(unit,'(a,a,i0,a,es25.16e3)') name, '(', j, ')=', value(j)
    end do
  end subroutine put_real_array

end module HarnessIOMod
