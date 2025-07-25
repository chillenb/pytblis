cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

option(PYTBLIS_PYPI_BUILD "Assume we're building from an sdist or for PyPI" ON)

if(${PYTBLIS_PYPI_BUILD})
  # PyPI doesn't like openmp
  set(ENABLE_THREAD_MODEL pthread)
endif()

nanobind_add_module(_pytblis_impl src/pytblis.cxx)

if(NOT ${PYTBLIS_PYPI_BUILD})
  find_package(TBLIS)
endif()


if(NOT TBLIS_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    tblis
    GIT_REPOSITORY https://github.com/chillenb/tblis.git
    GIT_TAG 38f7a4a06f04c416e4c7ddac69fe6969cf4dac32
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(tblis)
  if(${PYTBLIS_PYPI_BUILD})
    set(ENABLE_THREAD_MODEL pthread)
    target_link_libraries(_pytblis_impl PRIVATE tblis-static)
  else()
    target_link_libraries(_pytblis_impl PRIVATE tblis)
  endif()
else()
# TBLIS_FOUND
target_link_libraries(_pytblis_impl PRIVATE TBLIS::tblis)
endif()

set_target_properties(_pytblis_impl PROPERTIES CXX_STANDARD 20)


if(${SKBUILD})
  install(TARGETS _pytblis_impl LIBRARY DESTINATION pytblis)
endif()
