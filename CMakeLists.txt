cmake_minimum_required(VERSION 3.15...3.26)
include(CMakeDependentOption)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

find_package(
  Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

option(ADD_TBLIS_TO_RPATH
       "Add TBLIS library path to RPATH of the pytblis module" ON)

nanobind_add_module(_pytblis_impl STABLE_ABI src/pytblis.cxx)

find_package(TBLIS)

# If TBLIS is not found, we're going to build it, so may as well static link
cmake_dependent_option(PYTBLIS_STATIC_LINK "Statically link TBLIS" OFF
                       TBLIS_FOUND ON)

if(NOT TBLIS_FOUND)
  add_subdirectory(tblis EXCLUDE_FROM_ALL)
else()

endif()
# TBLIS_FOUND
if(${PYTBLIS_STATIC_LINK})
  message(STATUS "Statically linking TBLIS into pytblis module")
  target_link_libraries(_pytblis_impl PRIVATE TBLIS::tblis-static)
else()
  message(STATUS "Dynamically linking TBLIS")
  target_link_libraries(_pytblis_impl PRIVATE TBLIS::tblis)
  if(${ADD_TBLIS_TO_RPATH})
    get_target_property(tblis_location TBLIS::tblis LOCATION)
    get_filename_component(tblis_parent_dir ${tblis_location} DIRECTORY)
    set_target_properties(_pytblis_impl PROPERTIES INSTALL_RPATH
                                                   ${tblis_parent_dir})
  endif()
endif()

set_target_properties(_pytblis_impl PROPERTIES CXX_STANDARD 20)

if(${SKBUILD})
  install(TARGETS _pytblis_impl LIBRARY DESTINATION pytblis)
endif()
