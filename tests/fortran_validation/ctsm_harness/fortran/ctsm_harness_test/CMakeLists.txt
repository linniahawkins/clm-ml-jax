set (pfunit_sources
  test_column.pf
  test_retention_curve.pf)

# Sources of the harness that are not tests. HarnessPathsMod.F90 is written by build.py.
set (extra_sources
  HarnessPathsMod.F90
  HarnessIOMod.F90
  HarnessColumnMod.F90)

add_pfunit_ctest(ctsm_harness
  TEST_SOURCES "${pfunit_sources}"
  OTHER_SOURCES "${extra_sources}"
  LINK_LIBRARIES clm csm_share esmf
  EXTRA_FINALIZE unittest_finalize_esmf
  EXTRA_USE unittestInitializeAndFinalize)
