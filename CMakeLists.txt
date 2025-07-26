cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

option(PYTBLIS_PYPI_BUILD "Assume we're building from an sdist or for PyPI" ON)

nanobind_add_module(_pytblis_impl src/pytblis.cxx)

find_package(TBLIS REQUIRED)


if(NOT TBLIS_FOUND)
  add_subdirectory(tblis EXCLUDE_FROM_ALL)
  if(${PYTBLIS_PYPI_BUILD})
    set(ENABLE_THREAD_MODEL pthread)
    add_library(TBLIS::tblis-static ALIAS tblis)
  endif()
endif()
# TBLIS_FOUND
if(${PYTBLIS_PYPI_BUILD})
  target_link_libraries(_pytblis_impl PRIVATE TBLIS::tblis-static)
else()
  target_link_libraries(_pytblis_impl PRIVATE TBLIS::tblis)
endif()

set_target_properties(_pytblis_impl PROPERTIES CXX_STANDARD 20)


if(${SKBUILD})
  install(TARGETS _pytblis_impl LIBRARY DESTINATION pytblis)
endif()
