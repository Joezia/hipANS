# Provenance recorded in the benchmark's JSON output.
execute_process(
  COMMAND git -C ${hipANS_SOURCE_DIR} describe --always --dirty --abbrev=40 "--exclude=*"
  OUTPUT_VARIABLE HIPANS_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT HIPANS_GIT_COMMIT)
  set(HIPANS_GIT_COMMIT "unknown")
endif()

hip_add_executable(hipans_decode_bench hipans_decode_bench.cpp)
target_compile_definitions(hipans_decode_bench PRIVATE
  HIPANS_COMMIT_BUILT="${HIPANS_GIT_COMMIT}"
  HIPANS_BUILD_FLAGS="${CMAKE_HIP_FLAGS}")
target_link_libraries(hipans_decode_bench
  gpu_ans
  hipans_utils
  glog::glog
  pthread
)
