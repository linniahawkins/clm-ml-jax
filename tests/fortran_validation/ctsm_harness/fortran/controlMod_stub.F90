module controlMod
  ! Stand-in for CLM's controlMod in the harness: only the name of the namelist file, which
  ! SoilWaterMovementMod reads its settings from.
  implicit none
  public
  character(len=256) :: NLFilename = 'lnd_in_harness'
end module controlMod
