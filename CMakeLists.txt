cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

option(PYTBLIS_STATIC_LINK "Statically link TBLIS" OFF)

nanobind_add_module(_pytblis_impl src/pytblis.cxx)

find_package(TBLIS)

if(NOT TBLIS_FOUND)
  add_subdirectory(tblis EXCLUDE_FROM_ALL)
  add_library(TBLIS::tblis-static ALIAS tblis-static)
  add_library(TBLIS::tblis ALIAS tblis)
endif()
# TBLIS_FOUND
if(${PYTBLIS_STATIC_LINK})
  target_link_libraries(_pytblis_impl PRIVATE TBLIS::tblis-static)
else()
  target_link_libraries(_pytblis_impl PRIVATE TBLIS::tblis)
endif()

set_target_properties(_pytblis_impl PROPERTIES CXX_STANDARD 20)

if(${SKBUILD})
  install(TARGETS _pytblis_impl LIBRARY DESTINATION pytblis)
endif()
